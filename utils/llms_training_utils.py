import torch
import os
import torch.distributed as dist
from modules.model_gpt import GPTConfig, GPT
from torch.nn.parallel import DistributedDataParallel as DDP
import math
from transformers import AutoModelForCausalLM
import inspect
from modules.model_gpt import WindowMaskGeneration

@torch.no_grad()
def estimate_loss_gpt2(model, config, get_batch, ctx):
    """Estimate loss on multiple samples

    Args:
        model (torch.nn.Module): model
        config (obj): configuration
        get_batch (function): function generating new samples
        ctx (torch.amp.autocast_mode.autocast): pytorch context allowing mixed precision

    Returns:
        mean loss
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y, _ = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def initialize_ddp_process(config):
    """initialize data parallel processing

    Args:
        config (obj): configuration

    Returns:
        config (obj): updated configuration
    """
    # various inits, derived attributes, I/O setup
    print(f"Init DDP process:")
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    device_type = 'cuda' if 'cuda' in config.device else 'cpu' # for later use in torch.autocast
    if ddp:
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(ddp_local_rank)
        dist.init_process_group(backend=config.backend)
        # torch._dynamo.config.optimize_ddp = False
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert config.gradient_accumulation_steps % ddp_world_size == 0, f'gradient_accumulation_steps: {config.gradient_accumulation_steps}\t ddp_world_size: {ddp_world_size}'
        config.gradient_accumulation_steps //= ddp_world_size
        print("World size = ", ddp_world_size)
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device = config.device
        ddp_local_rank = 0
    
    config.ddp = ddp
    config.device = device
    config.ddp_world_size = ddp_world_size
    config.ddp_local_rank = ddp_local_rank
    config.master_process = master_process
    config.seed_offset = seed_offset
    config.device_type = device_type
    return config

def initialize_model_gpt2(config):
    """    
    Optimizer and model initialization either from scratch or checkpoints.
    Models are wrapped into a DDP container for DDP runs.

    Args:
        config (obj)

    Returns:
        model (nn.Module)
        optimizer (nn.optim)
        config (obj) updated configuration
        teacher_model (nn.Module) optional model for distillation
    """
    model_args = dict(batch_size=config.batch_size,
                attention=config.attention,
                quantization_levels_input=config.quantization_levels_input,
                quantization_levels_weights=config.quantization_levels_weights,
                quantization_levels_output=config.quantization_levels_output,
                sliding_window_size=config.sliding_window_size,
                decay_factor=config.decay_factor,
                n_layer=config.n_layer,
                n_head=config.n_head,
                n_embd=config.n_embd,
                block_size=config.block_size,
                bias=config.bias,
                dropout=config.dropout,
                triton=config.triton,
                max_annealing_step=config.max_annealing_step,
                qkv_out_norm=config.qkv_out_norm,
                rope=config.rope,
                LayerScale=config.LayerScale,
                  ) # start with model_args from command line
    
    if config.init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
    elif config.init_from == 'resume':
        print(f"Resuming training from {config.out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(config.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'attention', 'sliding_window_size',
                  'quantization_levels_input', 'quantization_levels_weights', 'quantization_levels_output', 'decay_factor', 'triton', 'max_annealing_step',
                  'qkv_out_norm', 'rope', 'LayerScale']:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)        
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
        config.iter_num = checkpoint['iter_num']
        config.best_val_loss = checkpoint['best_val_loss']
        try:
            config.wandb_run_id = checkpoint['config']['wandb_run_id']
        except:
            config.wandb_run_id = None
    elif config.init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {config.init_from}")
        # initialize from OpenAI GPT-2 weights. We don't override parameters related to network size, as opposed to model_args we can use when from scratch.
        override_args = dict(dropout=config.dropout,
                            attention=config.attention,
                            quantization_levels_input=config.quantization_levels_input,
                            quantization_levels_weights=config.quantization_levels_weights,
                            quantization_levels_output=config.quantization_levels_output,
                            decay_factor=config.decay_factor,
                            batch_size=config.batch_size,
                            sliding_window_size=config.sliding_window_size,
                            triton=config.triton,
                            max_annealing_step=config.max_annealing_step,
                            qkv_out_norm=config.qkv_out_norm,
                            rope=config.rope,
                            LayerScale=config.LayerScale,
                            )
        model = GPT.from_pretrained(config.init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'attention', 'sliding_window_size',
                  'quantization_levels_input', 'quantization_levels_weights', 'quantization_levels_output', 'decay_factor', 'triton', 'max_annealing_step',
                  'qkv_out_norm', 'rope', 'LayerScale']:
            model_args[k] = getattr(model.config, k)
        try:
            config.wandb_run_id = checkpoint['config']['wandb_run_id']
        except:
            config.wandb_run_id = None
            
    config.model_args = model_args
            
    # crop down the model block size if desired, using model surgery
    if config.block_size < model.config.block_size:
        model.crop_block_size(config.block_size)
        model_args['block_size'] = config.block_size # so that the checkpoint will have the right value
        
    if config.teacher:
        teacher_sd = torch.load(f'../saved_models/{config.init_from}.pt', map_location="cpu", weights_only=False)
        config = GPTConfig(**teacher_sd['model_args'])
        teacher_model = GPT(config)
        teacher_model.load_state_dict(teacher_sd['model'])
        teacher_model.to(config.device)
    else:
        teacher_model = None
    
    model.to(config.device)
    
    # compile the model
    if config.compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model) # requires PyTorch 2.0
        print('Model compiled successfully')
        
    # wrap model into DDP container
    if config.ddp:
        model = DDP(model, device_ids=[config.ddp_local_rank],
                    # find_unused_parameters=True
                    )
        
    # optimizer
    raw_model = model.module if config.ddp else model
    optimizer = raw_model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), config.device_type)
    if config.init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, config, teacher_model

# learning rate decay scheduler (cosine with warmup)
def get_lr(it, config):
    """get the curret learning rate according to a scheduler

    Args:
        it (int): iteration number
        config (obj)

    Returns:
        float: learning rate
    """
    # 1) linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config.lr_decay_iters:
        return config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

def calibration_algorithm(config, get_batch, raw_model):
    X, Y = get_batch('train')
    model = model.to(config.device)
    model.train()
    done = False
    error_threshold = 0.0001    
    max_calibration_iter = 100
    alpha_max = 1.0
    alpha_min = 0.1
    alpha_decay_max_step = 50
    alpha_calibration = torch.linspace(alpha_max, alpha_min, alpha_decay_max_step)

    calibration_iter = 0
    with torch.no_grad():   
        # Start calibration procedure     
        for layer in raw_model.transformer.h:
            for scaler_a_b in [layer.attn.q_scaler, layer.attn.k_scaler, layer.attn.v_scaler, layer.attn.output_scaler]:
                scaler_a_b.calibration = True

        while not(done):
            logits, loss = model(X, targets=Y)
            # Init a and b w.r.t computed statistics
            std_errors = []
            mean_errors = []            
            if calibration_iter < alpha_decay_max_step:
                alpha = alpha_calibration[calibration_iter].item()
            else:
                alpha = alpha_min                
            done_list = []
            for l, layer in enumerate(raw_model.transformer.h):
                for param_id, scaler_a_b in enumerate([layer.attn.q_scaler, layer.attn.k_scaler, layer.attn.v_scaler, layer.attn.output_scaler]):
                    for h in range(config.n_head):
                        std_errors += [torch.abs(scaler_a_b.std_after_scale[:, h]-scaler_a_b.target_std[:, h]).squeeze().item()]      
                        if std_errors[-1] > error_threshold:
                            if (scaler_a_b.std_after_scale[:, h]).squeeze().item() != 0.:
                                new_a = scaler_a_b.a[:, h] * scaler_a_b.target_std[:, h] / scaler_a_b.std_after_scale[:, h]
                                scaler_a_b.a[:, h] = alpha * new_a + (1-alpha) * scaler_a_b.a[:, h]
                            done_list += [False]
                        else:
                            done_list += [True]
                    
                        mean_errors += [torch.abs(scaler_a_b.mean_after_scale[:, h]-scaler_a_b.target_mean[:, h]).squeeze().item()]
                        if mean_errors[-1] > error_threshold:
                            new_b = scaler_a_b.b[:, h] + (scaler_a_b.target_mean[:, h] - scaler_a_b.mean_after_scale[:, h])
                            scaler_a_b.b[:, h] = alpha * new_b + (1-alpha) * scaler_a_b.b[:, h]
                            done_list += [False]
                        else:
                            done_list += [True]
            if config.ddp:
                print(f'Rank {config.ddp_local_rank} Calibraton iter {calibration_iter} | Loss: {loss.item():.4f} | error threshold: {error_threshold:.3f}\tnum valid params: {torch.sum(torch.tensor(done_list))}/{len(done_list)}\tstd errors: {torch.sort(torch.tensor(std_errors), descending=True)[0][:3]}\tmean errors: {torch.sort(torch.tensor(mean_errors), descending=True)[0][:3]}')
            else:
                print(f'Calibraton iter {calibration_iter} | Loss: {loss.item():.4f} | error threshold: {error_threshold:.3f}\tnum valid params: {torch.sum(torch.tensor(done_list))}/{len(done_list)}\tstd errors: {torch.sort(torch.tensor(std_errors), descending=True)[0][:3]}\tmean errors: {torch.sort(torch.tensor(mean_errors), descending=True)[0][:3]}')
            calibration_iter += 1
            # assert calibration_iter < max_calibration_iter, f'Calibration algorithm did not converge after {calibration_iter} steps.'
            if torch.all(torch.tensor(done_list)):
                print(f'Calibration finished after {calibration_iter} steps.')
                done = True    
            if calibration_iter > max_calibration_iter:
                f'Calibration algorithm did not converge after {calibration_iter} steps.'
                break
            
    # End calibration procedure
    for layer in raw_model.transformer.h:
        for scaler_a_b in [layer.attn.q_scaler, layer.attn.k_scaler, layer.attn.v_scaler, layer.attn.att_score_scaler, layer.attn.output_scaler]:
            scaler_a_b.calibration = False

    if config.ddp:
        with torch.no_grad():
            loss_accross_process = [torch.zeros(1, dtype=loss.dtype, device=config.device) for _ in range(config.ddp_world_size)]
            dist.all_gather(loss_accross_process, torch.tensor(loss.item(), device=config.device))
            best_calibration_rank = torch.argmin(torch.tensor(loss_accross_process)).item()
            print(f'Best loss on rank {best_calibration_rank}')        
            for layer in raw_model.transformer.h:
                for scaler_a_b in [layer.attn.q_scaler, layer.attn.k_scaler, layer.attn.v_scaler, layer.attn.output_scaler]:
                    dist.broadcast(scaler_a_b.a, src=best_calibration_rank)
                    dist.broadcast(scaler_a_b.b, src=best_calibration_rank)   
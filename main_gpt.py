import os
import time
import torch.distributed as dist
from types import SimpleNamespace
from contextlib import nullcontext
import torch
import torch.nn as nn
from dataloaders import llms_training_loaders
from configs.configurator import return_config
from utils.llms_training_utils import initialize_ddp_process, initialize_model_gpt2, estimate_loss_gpt2, get_lr, calibration_algorithm
import torch._dynamo
torch._dynamo.config.suppress_errors = True
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ["TORCH_DYNAMO_CACHE_SIZE"] = "16"
os.environ["NCCL_TIMEOUT_MS"] = "1200000"

# Load configuration
config_dict = return_config() # this function returns configuration from default, overrided with condig files, and with argv arguments in top priority.
config = SimpleNamespace(**config_dict) # converted to a class object for easier use.

# Initialize data parallel process if it is a DDP run.
config = initialize_ddp_process(config)

# Precision settings:
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
ctx = nullcontext() if config.device_type == 'cpu' else torch.amp.autocast(config.device_type, dtype=ptdtype)

# Count tokens
tokens_per_iter = config.gradient_accumulation_steps * config.ddp_world_size * config.batch_size * config.block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

# Create model checkpoint directory
if config.master_process:
    os.makedirs(config.out_dir, exist_ok=True)

# Manual seed, offset makes sure all parallel process receive a different one
torch.manual_seed(1337 + config.seed_offset)

# Preparing dataset
dataset = getattr(llms_training_loaders, config.dataset)(config)
get_batch = dataset.get_batch

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
config.iter_num = 0
config.best_val_loss = 1e9

# Initialize model and optimizer
model, optimizer, config, teacher_model = initialize_model_gpt2(config)
raw_model = model.module if config.ddp else model # unwrap DDP container if needed

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16'))

# logging
if config.wandb_log and config.master_process:
    import wandb
    mode = "offline" if config.wandb_offline else "online"
    if config.init_from=="resume" and mode=="online" and config.wandb_run_id is not None:
        wandb_run = wandb.init(dir='../', mode=mode, project=config.wandb_project, id=config.wandb_run_id, resume="must")
    else:
        wandb_run = wandb.init(dir='../', mode=mode, project=config.wandb_project, name=config.wandb_run_name, config=config, group=config.wandb_group_name)

# Prepare model scaling factors with calibration algorithm
if (config.attention=='DRAMAttention') and ((config.init_from=='gpt2-LinearDRAMAttention') or (config.init_from=='gpt2-xl-LinearDRAMAttention')):
    calibration_algorithm(config, get_batch, raw_model)
   
# Start training
X, Y, _ = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(config.iter_num, config) if config.decay_lr else config.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if config.iter_num % config.eval_interval == 0 and  config.master_process:
        losses = estimate_loss_gpt2(model, config, get_batch, ctx)
        print(f"step {config.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if config.wandb_log:
            wandb.log({
                "iter": config.iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })

        if (losses['val'] < config.best_val_loss or config.always_save_checkpoint) and (config.iter_num <= config.stop_saving_after):
            config.best_val_loss = losses['val']
            if config.iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': config.model_args,
                    'iter_num': config.iter_num,
                    'best_val_loss': config.best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {config.out_dir}")
                torch.save(checkpoint, os.path.join(config.out_dir, 'ckpt.pt'))
    if config.iter_num == 0 and config.eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(config.gradient_accumulation_steps):
        if config.ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == config.gradient_accumulation_steps - 1)
        with ctx:
            logits, loss_st = model(X, Y)
            if config.teacher:
                teacher_logits, _ = teacher_model(X, Y)
                dist_loss = nn.MSELoss()(logits, teacher_logits)
                # dist_loss = nn.CrossEntropyLoss()(logits/dist_temperature, teacher_logits/dist_temperature)
                # [(10**t ** 2 * nn.KLDivLoss()(F.softmax(logits/10**t, dim=-1), F.softmax(teacher_logits/10**t, dim=-1))).item() for t in list(range(-5, 5))]
                if config.dist_lambda==1:
                    loss = dist_loss
                else:                    
                    loss = config.dist_lambda * dist_loss + (1-config.dist_lambda) * loss_st
            else:
                loss = loss_st
                
            loss = loss / config.gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, _ = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        
    # Print memory usage on RAM
    try:
        from pynvml_utils import nvidia_smi
        if config.ddp:
            print(f"Memory usage on rank {config.ddp_local_rank} is: {nvidia_smi.getInstance().DeviceQuery('memory.used')['gpu'][config.ddp_local_rank]['fb_memory_usage']['used']}")
        else:
            print(f"Memory usage is: {nvidia_smi.getInstance().DeviceQuery('memory.used')['gpu'][0]['fb_memory_usage']['used']}")
    except:
        pass
        
    # clip the gradient
    if config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if config.iter_num % config.log_interval == 0 and config.master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * config.gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(config.batch_size * config.gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {config.iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    config.iter_num += 1
    ### TEST
    for layer in raw_model.transformer.h:
        layer.attn.iter_num = config.iter_num

    local_iter_num += 1

    # termination conditions
    if config.iter_num > config.max_iters:
        break

if config.ddp:
    dist.destroy_process_group()

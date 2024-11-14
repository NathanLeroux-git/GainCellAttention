"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os

# Check if inside a Conda environment
conda_env = os.environ.get('CONDA_DEFAULT_ENV')
if conda_env:
    print("Conda environment:", conda_env)
else:
    print("Not in a Conda environment")


import sys
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
# import tiktoken

from modules.model_gpt import GPTConfig, GPT
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ["TORCH_DYNAMO_CACHE_SIZE"] = "16"
# os.environ["TORCH_LOGS"] = "recompiles"
os.environ["NCCL_TIMEOUT_MS"] = "1200000"
# os.environ["TORCH_LOGS"] = "+dynamo"
# os.environ["TORCHDYNAMO_VERBOSE"] = "1"
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# torch._dynamo.config.optimize_ddp=False
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
nproc_per_node = 4
nnodes = 1
node_rank = 0
master_addr = "0"
master_port = 9901
out_dir = 'checkpoints'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
stop_saving_after = float('inf')
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_offline = False
wandb_project = 'GPT2'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
wandb_group_name = 'tests'
# data
dataset = 'openwebtext'
if os.path.exists('../datasets/texts/'):
    data_dir = '../datasets/texts/'
elif os.path.exists('/p/project1/neuroml/common/datasets'):
    data_dir = '/p/project1/neuroml/common/datasets'
else:
    print("Data directory not found. Exiting...")
    sys.exit(1)
    
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
attention = "CausalSelfAttention"
sliding_window_size = 1024
quantization_levels_input = 2**32
quantization_levels_weights = 2**32
quantization_levels_output = 2**32
decay_factor = 0.
teacher = False
dist_lambda = 0.9
dist_temperature = 2
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configs/configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
print(f"Init DDP process:")
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    dist.init_process_group(backend=backend)
    # torch._dynamo.config.optimize_ddp = False
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0, f'gradient_accumulation_steps: {gradient_accumulation_steps}\t ddp_world_size: {ddp_world_size}'
    gradient_accumulation_steps //= ddp_world_size
    print("World size = ", ddp_world_size)
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

if dataset=="openwebtext":
    # poor man's data loader
    data_dir = os.path.join(data_dir, dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
    
elif dataset=="slimpajama":
    # enc = tiktoken.get_encoding("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(f"Loading SlimPajama-627B dataset...")
    ds = load_dataset("cerebras/SlimPajama-627B", num_proc=8)
    train_data = ds['train']['text']
    val_data = ds['validation']['text']
    
    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length=1024):
            self.tokenizer = tokenizer
            self.texts = texts
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            # Start with the current text
            current_text = self.texts[idx]
            tokens = self.tokenizer(current_text, return_tensors="pt")["input_ids"].squeeze(0)
            tokens = self.tokenizer(current_text,
                                    return_tensors="pt",
                                    truncation=True,
                                    max_length=self.max_length
                                    )["input_ids"].squeeze(0)
            
            # Continue adding tokens from the next texts until we reach max_length
            while tokens.size(0) < self.max_length:
                idx = (idx + 1) % len(self.texts)  # Move to the next text (circular index)
                next_tokens = self.tokenizer(self.texts[idx], return_tensors="pt")["input_ids"].squeeze(0)
                tokens = torch.cat((tokens, next_tokens), dim=0)
            
            # Truncate to max_length
            tokens = tokens[:self.max_length]
            
            # Create targets by shifting tokens to the right by one position
            targets = tokens.clone()
            targets[:-1] = tokens[1:]
            targets[-1] = self.tokenizer.eos_token_id  # Assign EOS token or padding to the last token
            
            return tokens, targets
        
    train_set = TextDataset(train_data, tokenizer, max_length=block_size)
    val_set = TextDataset(val_data, tokenizer, max_length=block_size)

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    
    def get_batch(split):
        dataloader = train_dataloader if split == 'train' else val_dataloader
        x, y = next(iter(dataloader))
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(batch_size=batch_size,
                  attention=attention,
                  quantization_levels_input=quantization_levels_input,
                  quantization_levels_weights=quantization_levels_weights,
                  quantization_levels_output=quantization_levels_output,
                  sliding_window_size=sliding_window_size,
                  decay_factor=decay_factor,
                  n_layer=n_layer,
                  n_head=n_head,
                  n_embd=n_embd,
                  block_size=block_size,
                  bias=bias,
                  vocab_size=50257,
                  dropout=dropout) # start with model_args from command line

def remove_state_dict_prefix(state_dict, prefix='_orig_mod.'):
    new_state_dict = state_dict.copy()
    for (key, value) in state_dict.items():
        if key.startswith(prefix):
            del new_state_dict[key]
            new_state_dict.update({key[len(prefix):]: value})
    return new_state_dict

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50257
    # model_args['vocab_size'] = 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'attention', 'sliding_window_size', 'quantization_levels_input', 'quantization_levels_weights', 'quantization_levels_output', 'decay_factor']:
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
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout,
                         attention=attention,
                         quantization_levels_input=quantization_levels_input,
                         quantization_levels_weights=quantization_levels_weights,
                         quantization_levels_output=quantization_levels_output,
                         decay_factor=decay_factor,
                         batch_size=batch_size,
                         sliding_window_size=sliding_window_size,
                         )
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'attention', 'sliding_window_size', 'quantization_levels_input', 'quantization_levels_weights', 'quantization_levels_output', 'decay_factor']:
        model_args[k] = getattr(model.config, k)
    ### New
    # model_sd = torch.load(f"../saved_models/{init_from}.pt", map_location='cpu')
    # model_ld = model_sd['model']
    # model_ld = remove_state_dict_prefix(model_ld)
    # config_args = model_sd['model_args']
    # [config_args.update({k:v}) for (k, v) in model_args.items()]
    # print(f"Initializing {config_args['attention']} model from weights: {init_from}")
    # config_model = GPTConfig(**config_args)
    # model = GPT(config_model)
    # model.load_state_dict(model_ld, strict=False)
    # crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
    
if teacher:
    teacher_sd = torch.load(f'../saved_models/{init_from}.pt', map_location="cpu")
    config_teacher = GPTConfig(**teacher_sd['model_args'])
    teacher_model = GPT(config_teacher)
    teacher_model_ld = remove_state_dict_prefix(teacher_sd['model'])
    teacher_model.load_state_dict(teacher_model_ld, strict=False)
    teacher_model.to(device)
    if ddp:
        teacher_model = DDP(teacher_model, device_ids=[ddp_local_rank])
    raw_teacher_model = teacher_model.module if ddp else teacher_model # unwrap DDP container if needed
   
model.to(device)

# model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
# optimizer = model.module.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0
    print('Model compiled successfully')

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank],
                # find_unused_parameters=True
                )
    
def distillation_loss(raw_model, raw_teacher_model, logits, teacher_logits):
    # dist_loss = nn.MSELoss()(logits, teacher_logits)
    # dist_loss = nn.CrossEntropyLoss()(logits/dist_temperature, teacher_logits/dist_temperature)
    # [(10**t ** 2 * nn.KLDivLoss()(F.softmax(logits/10**t, dim=-1), F.softmax(teacher_logits/10**t, dim=-1))).item() for t in list(range(-5, 5))]
    
    # Layer local
    dist_loss_local = 0.
    for l in range(model_args['n_layer']):
        # attention_score = raw_model.transformer.h[l].attn.attention_score
        # attention_score_teacher = raw_teacher_model.transformer.h[l].attn.attention_score
        # dist_loss_local += nn.MSELoss()(attention_score, attention_score_teacher)
        
        layer_out = raw_model.transformer.h[l].attn.attention_output
        layer_out_teacher = raw_teacher_model.transformer.h[l].attn.attention_output
        dist_loss_local += nn.MSELoss()(layer_out, layer_out_teacher)                   
        
        # dist_loss_local -= nn.CosineSimilarity(dim=-1)(layer_out, layer_out_teacher).mean()
        
    # dist_loss = dist_loss_local / model_args['n_layer']
    dist_loss = dist_loss_local                
    dist_loss += nn.MSELoss()(logits, teacher_logits)
    return dist_loss

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    
    if teacher:
        teacher_model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            losses_dist = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    teacher_logits, teacher_loss = teacher_model(X, Y)
                    logits, loss = model(X, Y)
                    dist_loss = distillation_loss(raw_model, raw_teacher_model, logits, teacher_logits)                    
                losses[k] = loss.item()
                losses_dist[k] = dist_loss.item()
            out[split] = losses.mean()
            out[split+'_dist'] = losses_dist.mean()
    else:
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
            
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    mode = "offline" if wandb_offline else "online"
    wandb_run = wandb.init(dir='../', mode=mode, project=wandb_project, name=wandb_run_name, config=config, group=wandb_group_name)
    
raw_model = model.module if ddp else model # unwrap DDP container if needed

# Prepare model scaling factors with calibration algorithm
if (attention=='DRAMAttention') and ((init_from.startswith('gpt2-LinearDRAMAttention')) or (init_from=='gpt2-xl-LinearDRAMAttention')):
    X, Y = get_batch('train')
    model = model.to(device)
    model.train()
    done = False
    error_threshold = 0.001    
    max_calibration_iter = 100
    alpha_max = 1.0
    alpha_min = 0.01
    alpha_decay_max_step = 50
    alpha_calibration = torch.linspace(alpha_max, alpha_min, alpha_decay_max_step)

    calibration_iter = 0
    with torch.no_grad():   
        # Start calibration procedure     
        for layer in raw_model.transformer.h:
            for scaler_a_b in [layer.attn.q_scaler, layer.attn.k_scaler, layer.attn.v_scaler, layer.attn.output_scaler]:
                nn.init.constant_(scaler_a_b.a, val=1.0)
                nn.init.constant_(scaler_a_b.b, val=0.0)
                scaler_a_b.calibration = True
                scaler_a_b.save_target_stats = False

        while not(done):
            logits, loss = model(X, Y)
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
                    
                    std_after_scale, mean_after_scale = scaler_a_b.std_after_scale, scaler_a_b.mean_after_scale
                    target_std, target_mean = scaler_a_b.target_std, scaler_a_b.target_mean
                    
                    # std
                    std_errors += [torch.abs(std_after_scale-target_std).item()]   
                    if std_errors[-1] > error_threshold:
                        if std_after_scale != 0.:
                            new_a = scaler_a_b.a * target_std / std_after_scale
                            scaler_a_b.a.fill_(alpha * new_a + (1-alpha) * scaler_a_b.a)
                        done_list += [False]
                    else:
                        done_list += [True]
                        
                    # mean
                    mean_errors += [torch.abs(mean_after_scale-target_mean).item()]
                    if mean_errors[-1] > error_threshold:
                        new_b = scaler_a_b.b + (target_mean - mean_after_scale)
                        scaler_a_b.b.fill_(alpha * new_b + (1-alpha) * scaler_a_b.b)
                        done_list += [False]
                    else:
                        done_list += [True]
            if ddp:
                print(f'Rank {ddp_local_rank} Calibraton iter {calibration_iter} | Loss: {loss.item():.4f} | error threshold: {error_threshold:.4f}\tnum valid params: {torch.sum(torch.tensor(done_list))}/{len(done_list)}\tstd errors: {torch.sort(torch.tensor(std_errors), descending=True)[0][:3]}\tmean errors: {torch.sort(torch.tensor(mean_errors), descending=True)[0][:3]}')
            else:
                print(f'Calibraton iter {calibration_iter} | Loss: {loss.item():.4f} | error threshold: {error_threshold:.4f}\tnum valid params: {torch.sum(torch.tensor(done_list))}/{len(done_list)}\tstd errors: {torch.sort(torch.tensor(std_errors), descending=True)[0][:3]}\tmean errors: {torch.sort(torch.tensor(mean_errors), descending=True)[0][:3]}')
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

    if ddp:
        with torch.no_grad():
            loss_accross_process = [torch.zeros(1, dtype=loss.dtype, device=device) for _ in range(ddp_world_size)]
            dist.all_gather(loss_accross_process, torch.tensor(loss.item(), device=device))
            best_calibration_rank = torch.argmin(torch.tensor(loss_accross_process)).item()
            print(f'Best loss on rank {best_calibration_rank}')        
            for layer in raw_model.transformer.h:
            # for layer in model.module.transformer.h:
                # for scaler_a_b in [layer.attn.q_scaler, layer.attn.k_scaler, layer.attn.v_scaler, layer.attn.att_score_scaler, layer.attn.weight_average_scaler, layer.attn.output_scaler]:
                for scaler_a_b in [layer.attn.q_scaler, layer.attn.k_scaler, layer.attn.v_scaler, layer.attn.output_scaler]:
                    dist.broadcast(scaler_a_b.a, src=best_calibration_rank)
                    dist.broadcast(scaler_a_b.b, src=best_calibration_rank)       
            
# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
    # if iter_num % eval_interval == 0 and master_process and local_iter_num!=0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            if teacher:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "train/dist_loss": losses['train_dist'],
                    "val/dist_loss": losses['val_dist'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                })
            else:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                })

        if (losses['val'] < best_val_loss or always_save_checkpoint) and (iter_num <= stop_saving_after):
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss_st = model(X, Y)
            if teacher:
                with torch.no_grad():
                    teacher_logits, loss_teacher = teacher_model(X, Y)
            
                dist_loss = distillation_loss(raw_model, raw_teacher_model, logits, teacher_logits)  
                if dist_lambda==1:
                    loss = dist_loss
                else:                    
                    loss = dist_lambda * dist_loss + (1-dist_lambda) * loss_st
            else:
                loss = loss_st
                
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    try:
        from pynvml_utils import nvidia_smi
        print(f"Memory usage on rank {ddp_local_rank} is: {nvidia_smi.getInstance().DeviceQuery('memory.used')['gpu'][ddp_local_rank]['fb_memory_usage']['used']}")
    except:
        pass
        
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        if teacher:
            print(f"iter {iter_num}\t loss_st: {loss_st.item():.4f}\t loss_dist: {dist_loss.item():.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        else:
            print(f"iter {iter_num}\t loss: {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")    
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    dist.destroy_process_group()

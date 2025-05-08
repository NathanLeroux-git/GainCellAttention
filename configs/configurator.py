import sys
from ast import literal_eval
import torch
import os

if os.path.exists('../datasets/texts/'):
    data_dir = '../datasets/texts/'
else:
    print("Data directory not found. Exiting...")
    sys.exit(1)

config = dict(
nproc_per_node = 4,
nnodes = 1,
node_rank = 0,
master_addr = "0",
master_port = 9901,
out_dir = 'checkpoints',
eval_interval = 2000,
log_interval = 1,
eval_iters = 200,
eval_only = False, # if True, script exits right after the first eval
always_save_checkpoint = True, # if True, always save a checkpoint after each eval
stop_saving_after = float('inf'),
init_from = 'scratch', # 'scratch' or 'resume', or 'hf' or 'gpt2*'
model_type = 'gpt2',
calibrate = False,
# wandb logging
wandb_log = False, # disabled by default
wandb_offline = False,
wandb_project = 'GPT2',
wandb_run_name = 'gpt2', # 'run' + str(time.time())
wandb_group_name = 'tests',
wandb_run_id = '',
# data
dataset = 'openwebtext',
tokenizer = None, # None if using pre-tokenized data, else 'gpt2', etc.
data_dir = '../datasets/texts/' if os.path.exists('../datasets/texts/') else '/p/project1/neuroml/common/datasets',
gradient_accumulation_steps = 5 * 8, # used to simulate larger batch sizes
batch_size = 12, # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024,
# model
n_layer = 12,
n_head = 12,
n_embd = 768,
dropout = 0.0, # for pretraining 0 is good, for finetuning try 0.1+
bias = False, # do we use bias inside LayerNorm and Linear layers?
attention = "CausalSelfAttention",
triton = True,
sliding_window_size = 1024,
quantization_levels_input = 2**32,
quantization_levels_weights = 2**32,
quantization_levels_output = 2**32,
qkv_out_norm = False,
rope = False,
LayerScale = False,
decay_factor = 0.,
max_annealing_step = 0,
teacher = False,
dist_lambda = 0.9,
dist_temperature = 2,
# adamw optimizer
learning_rate = 6e-4 ,# max learning rate
max_iters = 600000, # total number of training iterations
weight_decay = 1e-1,
beta1 = 0.9,
beta2 = 0.95,
grad_clip = 1.0, # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True, # whether to decay the learning rate
warmup_iters = 2000, # how many steps to warm up for
lr_decay_iters = 600000, # should be ~= max_iters per Chinchilla
min_lr = 6e-5, # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl', # 'nccl', 'gloo', etc.
# system
device = 'cuda', # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16', # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True, # use PyTorch 2.0 to compile the model to be faster
)
# -----------------------------------------------------------------------------
override_config = {}
def return_config():
    for arg in sys.argv[1:]:
        if '=' not in arg:
            # assume it's the name of a config file
            assert not arg.startswith('--')
            config_file = arg
            print(f"Overriding config with {config_file}:")
            # with open(config_file) as f:
            #     print(f.read())
            exec(open(config_file).read(), globals())
            for key, val in override_config.items():
                assert key in config
                config.update({key: val})
                
        else:
            # assume it's a --key=value argument
            assert arg.startswith('--')
            key, val = arg.split('=')
            key = key[2:]
            if key in config:
                try:
                    # attempt to eval it (e.g. if bool, number, or etc)
                    attempt = literal_eval(val)
                except (SyntaxError, ValueError):
                    # if that goes wrong, just use the string
                    attempt = val
                # ensure the types match ok
                assert type(attempt) == type(config[key]), f'key: {key}, type attempt: {type(attempt)}\ttype config: {type(config[key])}'
                print(f"Overriding: {key} = {attempt}")
                config.update({key:attempt})
            else:
                raise ValueError(f"Unknown config key: {key}")
    return config

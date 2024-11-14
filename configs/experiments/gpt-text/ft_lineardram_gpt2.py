out_dir = '../saved_models/checkpoints/gpt2_LinearDRAMAttention'
wandb_log = False
wandb_offline = True
wandb_project = 'owt'
wandb_group_name = 'owt_training'
wandb_run_name = 'gpt2_LinearDRAMAttention'

init_from = 'gpt2'
attention = 'LinearDRAMAttention'

quantization_levels_input=16
quantization_levels_weights=8
quantization_levels_output=32
decay_factor = 1.6e-4

teacher = False
dist_lambda = 0.0
dist_temperature = 1
compile = True

block_size = 1024

batch_size = 16
gradient_accumulation_steps = 120

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 100
eval_iters = 100
log_interval = 1

# weight decay
weight_decay = 1e-1

# lr
learning_rate = 6e-4 # max learning rate

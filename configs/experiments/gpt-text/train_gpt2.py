# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
out_dir = '../saved_models/checkpoints/gpt2_with_scaling_statistics'
wandb_log = False
wandb_project = 'owt'
wandb_run_name='gpt2-CSA_with_scaling_statistics'

init_from = 'gpt2'
attention = 'LinearDRAMAttention'
teacher = False
dist_lambda = 0.0
dist_temperature = 1
compile = True

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520 -> gradient_accumulation_steps * batch_size needs to be 480, gradient_accumulation_steps needs to be a multiple of number of nodes
block_size = 1024

batch_size = 10
gradient_accumulation_steps = 48

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

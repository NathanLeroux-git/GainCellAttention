# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'checkpoints/shakespeare'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'GPT2'
import time
wandb_run_name = 'train-' + str(time.time())
wandb_group_name = 'train_on_shakespeare'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 64
batch_size = 4
block_size = 1024 # context of up to 1024 previous characters

# GPT 2 model
attention = "DRAMWrappedAttention"
# n_layer = 12
# n_head = 12
# n_embd = 768

n_layer = 1
n_head = 12 // 4
n_embd = 768 // 4

dropout = 0.2
vocab_size = 50304

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model

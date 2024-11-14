"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
import os
import sys
dir_name = os.getcwd()
# parent_dir_name = os.path.dirname(dir_name)
sys.path.insert(0, dir_name)

from modules.model_gpt import GPTConfig, GPT

# -----------------------------------------------------------------------------
batch_size = 8
out_dir = 'out' # ignored if init_from is not 'resume'
start = "The olympic games" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configs/configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

### RESTORE
# # model
# if init_from == 'resume':
#     # init from a model saved in a specific directory
#     ckpt_path = os.path.join(out_dir, 'ckpt.pt')
#     checkpoint = torch.load(ckpt_path, map_location=device)
#     gptconf = GPTConfig(**checkpoint['model_args'])
#     model = GPT(gptconf)
#     state_dict = checkpoint['model']
#     unwanted_prefix = '_orig_mod.'
#     for k,v in list(state_dict.items()):
#         if k.startswith(unwanted_prefix):
#             state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
#     model.load_state_dict(state_dict)
# elif init_from.startswith('gpt2') or init_from.startswith('checkpoints'):
#     # init from a given GPT-2 model
#     model = GPT.from_pretrained(init_from, dict(batch_size=batch_size,
#                                                 attention=attention,
#                                                 dropout=0.0,
#                                                 quantization_levels_input=16,
#                                                 quantization_levels_weights=8,
#                                                 quantization_levels_output=32,
#                                                 ))

### TEST
def remove_state_dict_prefix(state_dict, prefix='_orig_mod.'):
    new_state_dict = state_dict.copy()
    for (key, value) in state_dict.items():
        if key.startswith(prefix):
            del new_state_dict[key]
            new_state_dict.update({key[len(prefix):]: value})
    return new_state_dict

root = "/Users/leroux/sEMG/saved_models/"
init_from = "gpt2-DRAMAttention"
assert init_from in ["gpt2", "gpt2-LinearDRAMAttention", "gpt2-DRAMAttention"]
model_sd = torch.load(f'{root}/{init_from}.pt')
config_args = model_sd['model_args']
overide_args = dict(
                   )
[config_args.update({k:overide_args[k]}) if k in overide_args else None for (k, v) in overide_args.items()]

model_ld = model_sd['model']
model_ld = remove_state_dict_prefix(model_ld)

print(f"Initializing {config_args['attention']} model weights: {init_from}")
config = GPTConfig(**config_args)
model = GPT(config)
model.load_state_dict(model_ld, strict=False)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')

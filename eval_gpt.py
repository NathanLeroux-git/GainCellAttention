import torch
import torch.nn as nn
from modules.model_gpt import GPTConfig, GPT

def remove_state_dict_prefix(state_dict, prefix='_orig_mod.'):
    new_state_dict = state_dict.copy()
    for (key, value) in state_dict.items():
        if key.startswith(prefix):
            del new_state_dict[key]
            new_state_dict.update({key[len(prefix):]: value})
    return new_state_dict

device = 'cuda:0'
root = "/Data/pgi-15/common_models/dram_attention_project/"
init_from = "gpt2-DRAMAttention"
assert init_from in ["gpt2", "gpt2-LinearDRAMAttention", "gpt2-DRAMAttention"]
model_sd = torch.load(f'{root}/{init_from}.pt', map_location="cpu")
config_args = model_sd['model_args']
override_args = dict(
                #    attention="LinearDRAMAttention",
                #    n_layer=12,
                #    n_head=12,
                #    n_embd=768,
                #    dropout=0.,
                #    vocab_size=50257,
                #    block_size=1024,
                #    bias=True,
                #    quantization_levels_input=16,
                #    quantization_levels_weights=8,    
                #    quantization_levels_output=32,  
                   )
# [config_args.update({k:override_args[k]}) if k in override_args else None for (k, v) in config_args.items()]
[config_args.update({k:v}) for (k, v) in override_args.items()]

model_ld = model_sd['model']
model_ld = remove_state_dict_prefix(model_ld)

print(f"Initializing {config_args['attention']} model weights: {init_from}")
config = GPTConfig(**config_args)
model = GPT(config)
model.load_state_dict(model_ld, strict=False)
model.to(device)
pass
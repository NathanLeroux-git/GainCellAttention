import os
import sys
dir_name = os.getcwd()
# parent_dir_name = os.path.dirname(dir_name)
sys.path.insert(0, dir_name)
from modules.model_gpt import DRAMAttention
from dataclasses import dataclass
import torch
import numpy as np

@dataclass
class GPTConfig:
    block_size: int = 64
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 1
    n_head: int = 1
    n_embd: int = 64
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    attention: str = "DRAMAttention"
    quantization_levels_input: int = 2**32
    quantization_levels_weights: int = 2**32
    quantization_levels_output: int = 2**32
    sliding_window_size: int = 64
    batch_size: int = 10

args = GPTConfig()
attention = DRAMAttention(args, array_len=64)

## Scaler parameters from saved models
# trained_model = torch.load(f'/Users/leroux/sEMG/saved_models/gpt2-DRAMAttention.pt')['model']
# proj_names = ['q', 'k', 'v']
# with torch.no_grad():
#     for i, scaler in enumerate([attention.q_scaler, attention.k_scaler, attention.v_scaler]):
#         scaler.a.fill_(trained_model[f'_orig_mod.transformer.h.0.attn.{proj_names[i]}_scaler.a'].item())
#         scaler.b.fill_(trained_model[f'_orig_mod.transformer.h.0.attn.{proj_names[i]}_scaler.b'].item())

# Scaler parameters from 1 and 0
with torch.no_grad():
    for i, scaler in enumerate([attention.q_scaler, attention.k_scaler, attention.v_scaler]):
        scaler.a.fill_(1.)
        scaler.b.fill_(0.)
    
    B, H, T, D = args.batch_size, args.n_head, args.block_size, args.n_embd//args.n_head
    
    # start_from_file = None
    start_from_file = "tests_divers/64x64_single_arrays_final_output_pulse_notquant.npz"
    
    if start_from_file is None:
        q = torch.rand(B, H, T, D)
        k = torch.rand(B, H, T, D) * 0.9
        v = torch.rand(B, H, T, D) * 0.9
    else:
        saved_tensors = np.load(start_from_file, allow_pickle=True)
        q = torch.tensor(saved_tensors.f.Q).unsqueeze(1)
        k = torch.tensor(saved_tensors.f.K).unsqueeze(1)
        v = torch.tensor(saved_tensors.f.V).unsqueeze(1)

    _ = attention.attention_forward(q, k, v)
    S, phi_S, output = attention.S, attention.phi_S, attention.output

    q, k, v, S, phi_S, output = q.squeeze(), k.squeeze(), v.squeeze(), S.squeeze(), phi_S.squeeze(), output.squeeze()

    tensors_to_save = {'Q': q.to(torch.float).numpy(),
                    'K': k.to(torch.float).numpy(),
                    'V': v.to(torch.float).numpy(),
                    'S': S.to(torch.float).numpy(),
                    'phi_S': phi_S.to(torch.float).numpy(),
                    'output': output.to(torch.float).numpy(),
    }

file_name = "./tests_divers/64x64_single_arrays_with_minus_one.npz"
np.savez(file_name, **tensors_to_save, allow_pickle=True)

saved_tensors = np.load(file_name, allow_pickle=True)
for (key, value) in saved_tensors.items():
    print(key, value)

pass
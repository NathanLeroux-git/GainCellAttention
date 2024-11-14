import os
import sys
dir_name = os.getcwd()
# parent_dir_name = os.path.dirname(dir_name)
sys.path.insert(0, dir_name)
import torch
import torch.nn as nn
import time
import pynvml
from pynvml_utils import nvidia_smi
import sys
# import asyncio
import numpy as np
from dataclasses import dataclass
'''
How to measure GPU consumption
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0) <- for GPU 0
power = pynvml.nvmlDeviceGetPowerUsage(handle) <- returns power in mW
'''

filename = "./tests_divers/nvidia_4090_gpt2.npy"
is_jetson=False
if is_jetson:
    from jtop import jtop

S = 4   
H = 12
D_h = 64
n_tokens = 1024

lstm_model = False

offset_step = 5
n_steps = 10

device = 0

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(device)
mW_to_W = 1e3

input_size = 50257

from modules.model_gpt import GPT, GPTConfig
sd = torch.load('/Users/leroux/sEMG/saved_models/gpt2.pt')
config_args = sd['model_args']
config = GPTConfig(**config_args)
gpt2_net = GPT(config)
gpt2_net.load_state_dict(sd['model'], strict=False)
params = [v for v in gpt2_net.parameters()]
gpt2_params = sum(p.numel() for p in params)
print(f"gpt2 number of parameters: {gpt2_params}")

# lstm_net = nn.LSTM(
#     input_size=input_size,
#     hidden_size=384,
#     num_layers=12,
#     batch_first = True,
# )
lstm_net = nn.LSTM(
    input_size=input_size,
    hidden_size=512,
    num_layers=12,
    batch_first=True,
)

# wte = gpt2_net.transformer.wte # token encoding
params = [v for v in lstm_net.parameters()]
lstm_params = sum(p.numel() for p in params)
# params = [v for v in wte.parameters()]
# wte_params = sum(p.numel() for p in params)
print(f"lstm number of parameters: {lstm_params}")

if lstm_model:
    net = lstm_net.to(device)  
    # wte = wte.to(device)
else:
    net = gpt2_net.to(device)

mean_latency_per_token = torch.zeros(1)
mean_energy_per_token = torch.zeros(1)

print("tensor build")

for block_size_idx, T in enumerate(torch.tensor([n_tokens])):
    
    if lstm_model:
        x = torch.rand(S, n_tokens, input_size).to(device)
    else:
        x = (torch.rand(S, n_tokens) * input_size).to(torch.long).to(device)

    print("tensor loaded on the device")
    latency_tensor = torch.zeros(n_steps)
    power_tensor = torch.zeros(n_steps)
    energy_tensor = torch.zeros(n_steps)
    # energy_per_multiplication = torch.zeros(n_steps)
    with torch.no_grad():
        # warmup the gpus
        for i in range(offset_step):
            if not(lstm_model):
                for t in range(n_tokens): 
                    net(x[:, :t+1]) 
                # net(x)
            else:
                net(x)
   
            print(f'Offset step {i+1}/{offset_step}', end='\r')  
        
        start = time.time()
        for i in range(n_steps):
            inner_start = time.time()
            
            if not(lstm_model):
                for t in range(n_tokens): 
                    net(x[:, :t+1]) 
                # net(x)
            else:
                net(x)
      
            inner_end = time.time()

            if is_jetson:
                # power = 15 * 1e-3
                memory_usage = np.nan
                with jtop() as jetson:
                    power = jetson.stats["Power TOT"] * 1e-3 / mW_to_W  #  result was in micro W
            else:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / mW_to_W
                memory_usage = nvidia_smi.getInstance().DeviceQuery('memory.used')['gpu'][device]['fb_memory_usage']['used']
       
            latency_tensor[i] = inner_end - inner_start
            power_tensor[i] = power
            energy_tensor[i] = power_tensor[i] * latency_tensor[i]
            # energy_per_multiplication[i] = energy_tensor[i] / (dot_product_nmul + v_prod_nmul)
            if i % 1==0:
                print('GPU power consumption:', f'{power_tensor[i].item():.3f} W | ',
                    # 'Energy per multiplication:', f'{energy_per_multiplication[i].item():.2e} J | ',
                    'Memory usage:', f'{memory_usage:.2f} Mb | ',
                    f'Step {i+1}/{n_steps}', end='\r',
                    )
    end = time.time()
    time_to_end = end - start
    mean_power = power_tensor.mean()
    mean_latency = latency_tensor.mean()
    mean_energy = mean_power * mean_latency

    mean_latency_per_token[block_size_idx] = mean_latency  / S / T
    mean_energy_per_token[block_size_idx] = mean_energy / S / T
    
    print(f'\nblock size: {T}|\tmean latency per token: {mean_latency_per_token[block_size_idx].item():.2e}|\tmean energy per token per head: {mean_energy_per_token[block_size_idx].item():.2e}')
    
print(f'Mean latency per token: {mean_latency_per_token}')
print(f'Mean energy per token per head: {mean_energy_per_token}')

to_save = dict(
    is_jetson=is_jetson,
    lstm_model=lstm_model,
    S = S,
    H = H,
    D_h = D_h,
    N_tokens = n_tokens,
    mean_latency_per_token_per_head=mean_latency_per_token,
    mean_energy_per_token_per_head=mean_energy_per_token,
)

import os

# def get_unique_filename(directory, filename):
#     # Split the filename into name and extension
#     name, ext = os.path.splitext(filename)
#     unique_filename = filename
#     i = 1
    
#     # Check if the file already exists
#     while os.path.exists(os.path.join(directory, unique_filename)):
#         # If it exists, increment the number
#         unique_filename = f"{name}_{i}{ext}"
#         i += 1    
#     return unique_filename

# Function to save dictionary to a text file
def save_dict_to_file(dictionary, filename):
    with open(filename, 'w') as file:
        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")

save_dict_to_file(to_save, filename)
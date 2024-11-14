import os
import sys
import json
dir_name = os.getcwd()
parent_dir_name = os.path.dirname(dir_name)
sys.path.insert(0, parent_dir_name)
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

filename = "nvidia_jetson_attention.json"
is_jetson=True
if is_jetson:
    from jtop import jtop

S = 64
H = 12
D_h = 64
# T_tensor = torch.tensor([1024, 1024*2, 1024*4, 1024*8])
T_tensor = torch.tensor([1024])

sequential_model = True
if sequential_model:
    offset_step = 5
    n_steps = 10
else:
    offset_step = 100
    n_steps = 1000
    
device = 0
# qkv_proj_nmul = S * D * D_h * 3 * H * T
# dot_product_nmul = S * T * T * D_h * H    # should change to M * T * D * H depending wether we want to compare with self attention or sliding-window attention
# v_prod_nmul = S * T * T * D_h * H         # should change to M * T * D * H depending wether we want to compare with self attention or sliding-window attention
# out_proj_nmul = S * H * D_h * D * T
# total_nmul = qkv_proj_nmul + dot_product_nmul + v_prod_nmul + out_proj_nmul
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(device)
mW_to_W = 1e3

class Attention(nn.Module):
    def __init__(self, block_size=1024, window_size=4096):
        super().__init__()
        if block_size > window_size:
            block_size = window_size
        self.window_size = window_size
        self.register_buffer("k_cache", torch.zeros(S, block_size, H, D_h), persistent=False)
        self.register_buffer("v_cache", torch.zeros(S, block_size, H, D_h), persistent=False)
        self.flash_attention = False
        self.kv_cache = True        
    def forward(self, t, qkv=None):
        q, k, v = qkv # (S, T, H, D_h)
        n_samples, seq_len, _, _ = q.shape
        if self.kv_cache:
            t = t % self.window_size # sliding window
            self.k_cache[:n_samples, t:t+seq_len] = k
            self.v_cache[:n_samples, t:t+seq_len] = v
            k = self.k_cache
            v = self.v_cache
            q = q.transpose(1,2)
            k = k.transpose(1,2)
            v = v.transpose(1,2)
        if self.flash_attention:
            nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            k = k.transpose(-1, -2)
            x = torch.matmul(q, k)
            x = torch.matmul(x, v)   
    
mean_latency_per_token = torch.zeros(len(T_tensor))
mean_power_per_token = torch.zeros(len(T_tensor))
mean_energy_per_token = torch.zeros(len(T_tensor))

print("tensor build")
print("calibrating static power...")
static_power = []
for _ in range(10):
    if is_jetson:
        with jtop() as jetson:
            power = jetson.stats["Power TOT"] / mW_to_W  #  result was in micro W
    else:
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / mW_to_W
    time.sleep(0.01)
    static_power += [power]
static_power = torch.mean(torch.tensor(static_power))

for block_size_idx, T in enumerate(T_tensor):
    net = Attention(block_size=T).to(device)
    qkv = torch.rand(3, S, T, H, D_h).to(device)
    
    print("tensor loaded on the device")
    latency_tensor = torch.zeros(n_steps)
    power_tensor = torch.zeros(n_steps)
    energy_tensor = torch.zeros(n_steps)
    # energy_per_multiplication = torch.zeros(n_steps)
    with torch.no_grad():
        # warmup the gpus
        for i in range(offset_step):
            if sequential_model:    
                for t in range(T):
                    net(t=t, qkv=qkv[:,:,t].unsqueeze(2))  
            else:
                net(t=0, qkv=qkv)
   
            print(f'Offset step {i+1}/{offset_step}', end='\r')  
        
        start = time.time()
        for i in range(n_steps):
            inner_start = time.time()
            if sequential_model:  
                for t in range(T):
                    net(t=t, qkv=qkv[:,:,t].unsqueeze(2))  
            else:
                net(t=0, qkv=qkv)
      
            inner_end = time.time()

            if is_jetson:
                # power = 15 * 1e-3
                memory_usage = np.nan
                with jtop() as jetson:
                    power = jetson.stats["Power TOT"] / mW_to_W  #  result was in micro W
            else:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / mW_to_W
                memory_usage = nvidia_smi.getInstance().DeviceQuery('memory.used')['gpu'][device]['fb_memory_usage']['used']
       
            power -= static_power
       
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

    mean_latency_per_token[block_size_idx] = mean_latency / S / T
    mean_power_per_token[block_size_idx] = mean_power / S / T
    mean_energy_per_token[block_size_idx] = mean_energy / S / T
    
    print(f'\nblock size: {T}|\tmean latency per token: {mean_latency_per_token[block_size_idx].item():.2e}|\tmean energy per token: {mean_energy_per_token[block_size_idx].item():.2e}')
    
print(f'Block sizes: {T_tensor}')
print(f'Mean latency per token: {mean_latency_per_token}')
print(f'Mean power per token: {mean_power_per_token}')
print(f'Mean energy per token: {mean_energy_per_token}')

to_save = dict(
    is_jetson=is_jetson,
    is_sequential=sequential_model,
    S = S,
    H = H,
    D_h = D_h,
    N_tokens = T_tensor.item(),
    mean_latency_per_token=mean_latency_per_token.item(),
    mean_power_per_token=mean_power_per_token.item(),
    mean_energy_per_token=mean_energy_per_token.item(),
)

import os

def get_unique_filename(directory, filename):
    # Split the filename into name and extension
    name, ext = os.path.splitext(filename)
    unique_filename = filename
    i = 1
    
    # Check if the file already exists
    while os.path.exists(os.path.join(directory, unique_filename)):
        # If it exists, increment the number
        unique_filename = f"{name}_{i}{ext}"
        i += 1
    
    return unique_filename

directory = "./tests_divers"
unique_filename = get_unique_filename(directory, filename)

# Function to save dictionary to a text file
def save_dict_to_file(dictionary, filename):
    # with open(o
    #     for key, value in dictionary.items():
    #         file.write(f"{key}: {value}\n")
    with open(os.path.join(directory, unique_filename), 'w') as file:
        json.dump(to_save, file)

save_dict_to_file(to_save, unique_filename)
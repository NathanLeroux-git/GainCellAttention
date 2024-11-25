import torch
import torch.nn as nn
from math import pi
import sys, os
dir_name = os.getcwd()
sys.path.insert(0, dir_name)
from modules.model_gpt import GPT, GPTConfig
import numpy as np
from IPython.display import clear_output
from matplotlib import colormaps
import torch.nn.functional as F
from torchmetrics.text import Perplexity
import json

import json
import os

attentions = [
              "DRAMAttention",
              "NLAttention_x3",
              "NLAttention_x5",
              "NLAttention_sigmoid",
              "NLAttention_exponential",              
              ]
init_from = "../saved_models/gpt2_LinearDRAMAttention.pt"
file_name = "./tests_divers/adaptation_results.json"

def get_batch(split, device_id):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device_id, non_blocking=True), y.pin_memory().to(device_id, non_blocking=True)
    else:
        x, y = x.to(device_id), y.to(device_id)
    return x, y

def append_to_json(file_path, new_data):
    # Check if the file already exists
    if os.path.exists(file_path):
        # Open the existing JSON file and load its content
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
                # Ensure data is a list
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                # If the file is empty or malformed, initialize an empty list
                data = []
    else:
        # If the file doesn't exist, start with an empty list
        data = []

    # Append the new data to the list
    data.append(new_data)

    # Write the updated data back to the JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=3)
        # json.dump(data, file, indent=None, separators=(',', ': '))        

def remove_state_dict_prefix(state_dict, prefix='_orig_mod.'):
    new_state_dict = state_dict.copy()
    for (key, value) in state_dict.items():
        if key.startswith(prefix):
            del new_state_dict[key]
            new_state_dict.update({key[len(prefix):]: value})
    return new_state_dict

error_threshold = 0.0001
max_calibration_iter = 100
alpha_max = 1.0
alpha_min = 0.1
alpha_decay_max_step = 50
batch_size = 16
data_dir = os.path.join('../datasets/texts/', "openwebtext")
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
device_type = 'cuda'
device = 'cuda:1'
block_size = 1024
X_calib, Y_calib = get_batch('train', device)

for exp_id, attention in enumerate(attentions):
    print(f"Now testing {attention}:")
    exp_name = f"{attention} adapted from gpt2-LinearDRAMAttention"

    # Model to compare
    model_sd = torch.load(init_from, map_location='cpu')
    model_ld = model_sd['model']
    model_ld = remove_state_dict_prefix(model_ld)
    config_args = model_sd['model_args']
    override_args = dict(
                        batch_size=batch_size,
                        attention=attention,
                    )
    [config_args.update({k:v}) for (k, v) in override_args.items()]
    print(f"Initializing {config_args['attention']} model from weights: {init_from}")
    config = GPTConfig(**config_args)
    model = GPT(config)
    model.load_state_dict(model_ld, strict=True)

    ### Now model to calibrate
    model = model.to(device)
    model.train()
    done = False
    alpha_calibration = torch.linspace(alpha_max, alpha_min, alpha_decay_max_step)
    n_head = config_args['n_head']
    # Take colors at regular intervals spanning the colormap.    
    calibration_iter = 0
    converged = True

    with torch.no_grad():        
        for layer in model.transformer.h:
            for scaler_a_b in [layer.attn.q_scaler, layer.attn.k_scaler, layer.attn.v_scaler, layer.attn.output_scaler]:
                nn.init.constant_(scaler_a_b.a, val=1.0)
                nn.init.constant_(scaler_a_b.b, val=0.0)
                scaler_a_b.calibration = True
                scaler_a_b.save_target_stats = False
        
        losses = []
        while not(done):
            _, loss_train = model(X_calib, targets=Y_calib)
            model.eval()
            X, Y =  get_batch('val', device)
            _, loss= model(X, targets=Y)
            model.train()
            losses += [loss.item()]
            # Init a and b w.r.t computed statistics
            std_errors = []
            mean_errors = []            
            if calibration_iter < alpha_decay_max_step:
                alpha = alpha_calibration[calibration_iter].item()
            else:
                alpha = alpha_min                
            done_list = []
            for l, layer in enumerate(model.transformer.h):
                for param_id, scaler_a_b in enumerate([layer.attn.q_scaler, layer.attn.k_scaler, layer.attn.v_scaler, layer.attn.output_scaler]):
                    for h in range(n_head):
                        # print(f"l{l} h{h} param_id{param_id} std {scaler_a_b.std_after_scale[:, h]}")
                        std_errors += [torch.abs(scaler_a_b.std_after_scale[:, h]-scaler_a_b.target_std[:, h]).squeeze().item()]      
                        if std_errors[-1] > error_threshold:
                            if (scaler_a_b.std_after_scale[:, h]).squeeze().item() != 0.:
                                new_a = scaler_a_b.a[:, h] * scaler_a_b.target_std[:, h] / scaler_a_b.std_after_scale[:, h]
                                scaler_a_b.a[:, h] = alpha * new_a + (1-alpha) * scaler_a_b.a[:, h]
                            else:
                                pass
                            done_list += [False]
                        else:
                            done_list += [True]
                    
                        mean_errors += [torch.abs(scaler_a_b.mean_after_scale[:, h]-scaler_a_b.target_mean[:, h]).squeeze().item()]
                        if mean_errors[-1] > error_threshold:
                            new_b = scaler_a_b.b[:, h] + (scaler_a_b.target_mean[:, h] - scaler_a_b.mean_after_scale[:, h])
                            scaler_a_b.b[:, h] = alpha * new_b + (1-alpha) * scaler_a_b.b[:, h]
                            done_list += [False]
                        else:
                            done_list += [True]
                            
            print(f'Calibraton iter {calibration_iter} | Loss: {loss.item():.4f} | Ppl: {np.exp(loss.item()):.4f} | error threshold: {error_threshold:.2e}\tnum valid params: {torch.sum(torch.tensor(done_list))}/{len(done_list)}\tstd errors: {torch.sort(torch.tensor(std_errors), descending=True)[0][:3]}\tmean errors: {torch.sort(torch.tensor(mean_errors), descending=True)[0][:3]}')
            calibration_iter += 1
            if calibration_iter > max_calibration_iter:
                print(f'Calibration algorithm did not converge after {calibration_iter} steps.')
                converged = False
                break
            if torch.all(torch.tensor(done_list)):
                print(f'Calibration finished after {calibration_iter} steps.')
                done = True          
    print(f"Losses tensor: {torch.tensor(losses)}")
    print(f"Ppl tensor: {torch.exp(torch.tensor(losses))}")

    for layer in model.transformer.h:
        for scaler_a_b in [layer.attn.q_scaler, layer.attn.k_scaler, layer.attn.v_scaler, layer.attn.output_scaler]:
            scaler_a_b.calibration = False

    # final evaluation on eval set
    final_loss = []
    final_ppl = []
    n_eval = 100
    model.eval()
    with torch.no_grad():
        for i in range(n_eval):
            X, Y = get_batch('eval', device)  
            _, loss = model(X, targets=Y)
            final_loss += [loss.item()]
            final_ppl += [loss.exp().item()]
            print(f"final evaluation step {i}/{n_eval}\tloss: {loss.item():.3f}\tppl: {loss.exp().item():.3f}")

        final_loss = np.array(final_loss)
        final_ppl = np.array(final_ppl)
        
        mean_loss = final_loss.mean()
        mean_ppl = final_ppl.mean()

        std_loss = final_loss.std()
        std_ppl = final_ppl.std()
        print(f"mean_loss: {mean_loss:.2f}\t mean_ppl: {mean_ppl:.2f}\t std_loss: {std_loss:.2f}\t std_ppl: {std_ppl:.2f}\t")

    from datetime import datetime

    # datetime object containing current date and time
    now = datetime.now()
    
    print("now =", now)

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    new_result = {
        "name": exp_name,
        "date": dt_string,
        "batch_size": batch_size,
        "error_threshold": error_threshold,
        "max_calibration_iter": max_calibration_iter,
        "alpha_max": alpha_max,
        "alpha_min": alpha_min,
        "alpha_decay_max_step": alpha_decay_max_step,
        "mean_loss_eval": mean_loss,
        "mean_ppl_eval": mean_ppl,
        "std_loss_eval": std_loss,
        "std_ppl_eval": std_ppl,
        "converged": converged,
        "calibration_iters": calibration_iter,
        "loss_tensor": losses,
        "ppl_tensor": torch.exp(torch.tensor(losses)).tolist()    
    }

    append_to_json(file_name, new_result)
"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""
import os
import sys
import numpy as np
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import torchaudio
import transformers
from transformers.modeling_outputs import CausalLMOutput


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, args, flash=True):
        super().__init__()
        assert args.n_embd % args.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(args.n_embd, 3 * args.n_embd, bias=args.bias)
        # output projection
        self.c_proj = nn.Linear(args.n_embd, args.n_embd, bias=args.bias)
        # # KV caching
        # self.kv_cache = args.kv_cache
        # if self.kv_cache:
        #     self.register_buffer("k_cache", torch.zeros(args.batch_size, args.block_size, args.n_head, args.n_embd // args.n_head), persistent=False)
        #     self.register_buffer("v_cache", torch.zeros(args.batch_size, args.block_size, args.n_head, args.n_embd // args.n_head), persistent=False)

        # regularization
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.dropout = args.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and flash
        if not self.flash and args.attention=="CausalSelfAttention":
            # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(args.block_size, args.block_size))
                                        .view(1, 1, args.block_size, args.block_size),
                                        persistent=False)

    def forward(self, x: torch.tensor):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        q = q.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        v = v.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        
        q = q.transpose(1, 2) # (B, nh, T, hs)
        k = k.transpose(1, 2) # (B, nh, T, hs)
        v = v.transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class SlidingWindowAttention(CausalSelfAttention):
    def __init__(self, args):
        super().__init__(args, flash=False)
        self.attention_type = args.attention
        self.block_size = args.block_size
        # Masking
        self.register_buffer('mask', WindowMaskGeneration(n_tokens=args.block_size, n_memory=args.block_size, chunk_size=args.sliding_window_size).view(1, 1, args.block_size, args.block_size), persistent=False) 
        self.sliding_window_size = args.sliding_window_size

        # Vanilla dot-product softmax similarity
        self.mask_value_1 = 0.
        self.mask_value_2 = -np.inf
        self.similarity = torch.matmul
        self.weight_average = torch.matmul 
        trainable_a_b = False
        save_target_stats = False
        self.q_scaler = range_scaler(shape=(1, args.n_head, 1, 1), trainable_a_b=trainable_a_b, save_target_stats=save_target_stats) # To mitigate the effect of clamping between 0 and 1
        self.k_scaler = range_scaler(shape=(1, args.n_head, 1, 1), trainable_a_b=trainable_a_b, save_target_stats=save_target_stats)
        self.v_scaler = range_scaler(shape=(1, args.n_head, 1, 1), trainable_a_b=trainable_a_b, save_target_stats=save_target_stats)
        self.att_score_scaler = range_scaler(shape=(1,), a_init=(self.n_embd // self.n_head)**-0.5, trainable_a_b=False, save_target_stats=False)
        self.weight_average_scaler = range_scaler(shape=(1,), trainable_a_b=False, save_target_stats=False)
        self.output_scaler = range_scaler(shape=(1, args.n_head, 1, 1), trainable_a_b=trainable_a_b, save_target_stats=save_target_stats)
        
        self.quantization = quantization_custom.apply
        self.apply_input_quantization = True if args.quantization_levels_input!=2**32 else False
        self.apply_weights_quantization = True if args.quantization_levels_weights!=2**32 else False
        self.apply_output_quantization = True if args.quantization_levels_output!=2**32 else False
        self.quantization_levels_input = args.quantization_levels_input
        self.quantization_levels_weights = args.quantization_levels_weights
        self.quantization_levels_output = args.quantization_levels_output

        self.input_clamping_bounds = [-np.inf, +np.inf] if not(self.apply_input_quantization) else [-1., 1.]
        self.weights_clamping_bounds = [-np.inf, +np.inf] if not(self.apply_weights_quantization) else [-1., 1.]
        self.output_clamping_bounds = [-np.inf, +np.inf] if not(self.apply_output_quantization) else [-1., 1.]
        
        self.read_noise_qk =  nn.Identity()
        self.read_noise_Wv = nn.Identity()
        self.decay = nn.Identity()
        self.NL = nn.Softmax(dim=-1)
        
        self.array_length = args.block_size

        apply_bin_count = False
        self.bins_count_q = BinsCount(apply_bin_count, self.quantization_levels_input)
        self.bins_count_k = BinsCount(apply_bin_count, self.quantization_levels_weights)
        self.bins_count_v = BinsCount(apply_bin_count, self.quantization_levels_weights)
        self.bins_count_A = BinsCount(apply_bin_count, self.quantization_levels_input, mask=self.mask)
        self.bins_count_out = BinsCount(apply_bin_count, self.quantization_levels_output) # nn.Identity()

    def forward(self, x: torch.tensor, qkv: torch.tensor=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        
        if qkv is None:
            q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
            q = q.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
            v = v.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
            
            # if start_pos < self.sliding_window_size:
            #     start_pos_col = 0
            # else:
            #     start_pos_col = start_pos-self.sliding_window_size+1   
                         
            # if self.kv_cache:
            #     self.k_cache[:B, start_pos:start_pos+T] = k
            #     self.v_cache[:B, start_pos:start_pos+T] = v
            #     k = self.k_cache[:B, start_pos_col:start_pos_col+self.sliding_window_size]
            #     v = self.v_cache[:B, start_pos_col:start_pos_col+self.sliding_window_size]
            
            q = q.transpose(1, 2) # (B, nh, T, hs)
            k = k.transpose(1, 2) # (B, nh, T, hs)
            v = v.transpose(1, 2) # (B, nh, T, hs)
            
        else:
            q, k, v = qkv
            
        # y = self.attention_forward(q, k, v)       
        y = checkpoint(self.attention_forward, q, k, v, use_reentrant=False)
            
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

    def attention_forward(self, q, k, v):
        B, H, T, D = q.shape            
        k = k.transpose(2,3) # (n_samples, n_heads, head_dim, n_tokens)
        
        q = self.q_scaler(q)
        k = self.k_scaler(k)
        v = self.v_scaler(v)
        q = torch.clamp(q, self.input_clamping_bounds[0], self.input_clamping_bounds[1])
        k = torch.clamp(k, self.weights_clamping_bounds[0], self.weights_clamping_bounds[1])
        v = torch.clamp(v, self.weights_clamping_bounds[0], self.weights_clamping_bounds[1])
        q = self.bins_count_q(self.quantization(q, self.apply_input_quantization, self.quantization_levels_input, self.input_clamping_bounds[0], self.input_clamping_bounds[1]))
        k = self.bins_count_k(self.quantization(k, self.apply_weights_quantization, self.quantization_levels_weights, self.weights_clamping_bounds[0], self.weights_clamping_bounds[1]))
        v = self.bins_count_v(self.quantization(v, self.apply_weights_quantization, self.quantization_levels_weights, self.weights_clamping_bounds[0], self.weights_clamping_bounds[1]))
        
        if isinstance(self.decay, nn.Identity):
            x = self.similarity(q, k)
        else:
            x = self.similarity(q, k, self.decay.decay_mask[:, :, :T, :T])
        
        x = x.masked_fill(self.mask[:, :, :T, :T] == 0, self.mask_value_1)   # masking before the read noise is important so that the noise is not computed respected to masked values
        x = self.att_score_scaler(x, apply_mask=True) # (n_samples, n_heads, n_tokens, n_tokens)
        x = self.read_noise_qk(x)
    
        x = x.masked_fill(self.mask[:, :, :T, :T] == 0, self.mask_value_2)   # Need to mask again because self.att_score_scaler adds a scalar to all elements            
        x = self.NL(x)
       
        x = self.bins_count_A(x)
        x = torch.clamp(x, self.input_clamping_bounds[0], self.input_clamping_bounds[1])        
        x = self.attn_dropout(x) # B, H, T, T

        # Padding required when sequence length not multiple of array length
        n_tiles = (T + self.array_length - 1) // self.array_length
        x = torch.cat((x, torch.zeros(B, H, T, int(n_tiles*self.array_length-T)).to(x.device)), dim=-1) # padded to the next multiple of array_length
        v = torch.cat((v, torch.zeros(B, H, int(n_tiles*self.array_length-T), D).to(x.device)), dim=-2) # padded to the next multiple of array_length
        
        n_columns = x.shape[-1]
          
        x = x.view(B, H, T, n_tiles, n_columns//n_tiles).transpose(2, 3) # B, H, n_tiles, T, M/n_tiles
        v = v.view(B, H, n_tiles, n_columns//n_tiles, D) # B, H, n_tiles, T/n_tiles, D
        if isinstance(self.decay, nn.Identity):
            x = self.weight_average(x, v)
        else:
            decay_mask = self.decay.decay_mask[:, :, :T, :n_columns].view(1, 1, T, n_tiles, n_columns//n_tiles).transpose(2, 3) # B, H, n_tiles, T, T/n_tiles
            x = self.weight_average(x, v, decay_mask)
        
        x = self.bins_count_out(x)
        x = self.read_noise_Wv(x)
        x = self.weight_average_scaler(x)
        
        x = torch.clamp(x, self.output_clamping_bounds[0], self.output_clamping_bounds[1])        
        x = self.quantization(x, self.apply_output_quantization, self.quantization_levels_output, self.output_clamping_bounds[0], self.output_clamping_bounds[1])
        
        x = torch.sum(x, dim=2) # B, H, T, D
        x = self.output_scaler(x)        
        return x    
    
class LinearDRAMAttention(SlidingWindowAttention):
    def __init__(self, args, array_len=64):
        super().__init__(args)
        # # Masking
        # mask = WindowMaskGeneration(n_tokens=args.block_size, n_memory=args.block_size, chunk_size=args.sliding_window_size).view(1, 1, args.block_size, args.block_size)      
        trainable_a_b = True
        # target_previous_model_stats = False
        save_target_stats = True
        
        self.NL = nn.Identity()
        self.mask_value_1 = 0.
        self.mask_value_2 = 0.
        # self.NL = nn.Identity()
        
        # self.q_scaler = range_scaler(a_init=1., b_init=0.0, trainable_a_b=trainable_a_b, target_previous_model_stats=target_previous_model_stats, save_target_stats=save_target_stats) # To mitigate the effect of clamping between 0 and 1. Floating point is mendatory for a_init and b_init.
        # self.k_scaler = range_scaler(a_init=1., b_init=0.45, trainable_a_b=trainable_a_b, target_previous_model_stats=target_previous_model_stats, save_target_stats=save_target_stats)
        # self.v_scaler = range_scaler(a_init=1., b_init=0.45, trainable_a_b=trainable_a_b, target_previous_model_stats=target_previous_model_stats, save_target_stats=save_target_stats)              
        # self.att_score_scaler = range_scaler(a_init=1/80, b_init=0.0, range_a=[1/80, 1/80], range_b=[0., 0.], trainable_a_b=False, target_previous_model_stats=save_target_stats, save_target_stats=save_target_stats) 
        # self.weight_average_scaler = range_scaler(a_init=1/40, b_init=0.0, range_a=[1/40, 1/40], range_b=[0., 0.], trainable_a_b=False, target_previous_model_stats=False, save_target_stats=False) 
        # self.output_scaler = range_scaler(a_init=1., b_init=0., trainable_a_b=trainable_a_b, target_previous_model_stats=target_previous_model_stats, save_target_stats=save_target_stats)

        amp_coefficient_attn = 19.3
        self.similarity = offset_weights_matmul_QK(offset_input=0.0, offset_weight=0.45, amp_coefficient=amp_coefficient_attn) # amp coefficient corresponding to the regression fitting
        amp_coefficient_weight_average = 19.3
        self.weight_average = offset_weights_matmul_AV(offset_input=0.0, offset_weight=0.45, amp_coefficient=amp_coefficient_weight_average) # amp coefficient corresponding to the regression fitting 
        self.att_score_scaler = range_scaler(shape=(1,), a_init=1/80, b_init=0.0, range_a=[1/80, 1/80], range_b=[0., 0.], trainable_a_b=False, save_target_stats=False, mask=None) # need mask=self.mask if want to operate statistics saving (save_target_stats=True)
        self.weight_average_scaler = range_scaler(shape=(1,), a_init=1/40, b_init=0.0, range_a=[1/40, 1/40], range_b=[0., 0.], trainable_a_b=False, save_target_stats=False) 
        self.decay = decay_mask(self.mask, decay_factor=args.decay_factor)
        
        self.q_scaler = range_scaler(shape=(1, args.n_head, 1, 1), a_init=1., b_init=0.0, trainable_a_b=trainable_a_b, save_target_stats=save_target_stats) # To mitigate the effect of clamping between 0 and 1. Floating point is mendatory for a_init and b_init.
        self.k_scaler = range_scaler(shape=(1, args.n_head, 1, 1), a_init=1., b_init=0.45, trainable_a_b=trainable_a_b, save_target_stats=save_target_stats)
        self.v_scaler = range_scaler(shape=(1, args.n_head, 1, 1), a_init=1., b_init=0.45, trainable_a_b=trainable_a_b, save_target_stats=save_target_stats)              
        self.output_scaler = range_scaler(shape=(1, args.n_head, 1, 1), a_init=1., b_init=0., trainable_a_b=trainable_a_b, save_target_stats=save_target_stats)
        
        self.input_clamping_bounds = [0.0, 1.0]
        self.weights_clamping_bounds = [0., 0.9]
        self.output_clamping_bounds = [-1., 1.0]

        self.read_noise_qk = nn.Identity()
        self.read_noise_Wv = nn.Identity()

        self.array_length = array_len
    
    def forward(self, x: torch.tensor):
        return super().forward(x)
    
class decay_mask(nn.Module):
    def __init__(self, mask: torch.tensor, decay_factor: float):
        super().__init__()
        mask = torch.cumsum(mask, dim=-2) - 1
        decay = np.exp(-decay_factor)
        self.register_buffer('decay_mask', decay ** mask, persistent=False)
        
    def forward(self, x):
        x *= self.decay_mask
        return x
    
class offset_weights_matmul_QK(nn.Module):
    def __init__(self, offset_input: float, offset_weight: float, amp_coefficient:float):
        super().__init__()
        self.offset_input = offset_input
        self.offset_weight = offset_weight
        self.amp_coefficient = amp_coefficient
    def forward(self, x: torch.tensor, weights: torch.tensor, decay_mask: torch.tensor):
        x = self.amp_coefficient * torch.matmul(x-self.offset_input, weights-self.offset_weight)
        x = x * decay_mask
        return x
    
class offset_weights_matmul_AV(nn.Module):
    def __init__(self, offset_input: float, offset_weight: float, amp_coefficient:float):
        super().__init__()
        self.offset_input = offset_input
        self.offset_weight = offset_weight
        self.amp_coefficient = amp_coefficient
    def forward(self, x: torch.tensor, weights: torch.tensor, decay_mask: torch.tensor):
        x = x * decay_mask
        x = self.amp_coefficient * torch.matmul(x-self.offset_input, weights-self.offset_weight)
        return x
    
class DRAMAttention(SlidingWindowAttention):
    def __init__(self, args, array_len=64):
        super().__init__(args)
        # Masking (same as SlidingWindowAttention)
        # mask = WindowMaskGeneration(n_tokens=args.block_size, n_memory=args.block_size, chunk_size=args.sliding_window_size).view(1, 1, args.block_size, args.block_size)        
        # Circuits constrains
        
        # Clamped nonlinear scaled attention
        self.mask_value_1 = 0.
        self.mask_value_2 = 0.
        self.NL = nn.Identity()
        
        self.similarity = DRAM_MAC_temporal_encoding_surrogate_QK.apply
        self.weight_average = DRAM_MAC_temporal_encoding_surrogate_AV.apply
        # self.similarity = DRAM_MAC_temporal_encoding()
        # self.weight_average = DRAM_MAC_temporal_encoding()

        trainable_a_b = True
        save_target_stats=False
        
        nonlinear_amp = 19.3 # Linear fit coefficient from DRAM temporal encoding
        
        self.q_scaler = range_scaler(shape=(1, args.n_head, 1, 1), a_init=1., b_init=0.0, trainable_a_b=trainable_a_b, save_target_stats=save_target_stats) # To mitigate the effect of clamping between 0 and 1
        self.k_scaler = range_scaler(shape=(1, args.n_head, 1, 1), a_init=1., b_init=0.45, trainable_a_b=trainable_a_b, save_target_stats=save_target_stats)
        self.v_scaler = range_scaler(shape=(1, args.n_head, 1, 1), a_init=1., b_init=0.45, trainable_a_b=trainable_a_b, save_target_stats=save_target_stats)
        # self.att_score_scaler = range_scaler(shape=(1,), a_init=1/80, b_init=0.0, range_a=[1/80, 1/80], range_b=[0., 0.], trainable_a_b=False, save_target_stats=save_target_stats, mask=self.mask)
        self.att_score_scaler = range_scaler(shape=(1,), a_init=1/80, b_init=0.0, range_a=[1/80, 1/80], range_b=[0., 0.], trainable_a_b=False, save_target_stats=False, mask=None) # need mask=self.mask if want to operate statistics saving (or save_target_stats=True)
        self.weight_average_scaler = range_scaler(shape=(1,), a_init=1/40, b_init=0.0, range_a=[1/40, 1/40], range_b=[0., 0.], trainable_a_b=False, save_target_stats=False) 
        self.output_scaler = range_scaler(shape=(1, args.n_head, 1, 1), a_init=1 / nonlinear_amp, b_init=0., trainable_a_b=trainable_a_b, save_target_stats=save_target_stats)
        self.input_clamping_bounds = [0.0, 1.0]
        self.weights_clamping_bounds = [0., 0.9]
        self.output_clamping_bounds = [-1., 1.0]

        self.read_noise_qk = nn.Identity()
        self.read_noise_Wv = nn.Identity()
        self.decay = decay_mask(self.mask, decay_factor=args.decay_factor)
        self.array_length = array_len

    def forward(self, x: torch.tensor):
        return super().forward(x)
    
def WindowMaskGeneration(n_tokens: int, n_memory: int, chunk_size: int):
    mask = torch.zeros(n_tokens, n_memory)
    for i in range(n_tokens):
        if i<chunk_size:
            mask[i, :i+1] = 1.
        else:
            mask[i, i-chunk_size+1:i+1] = 1.
    return mask
    
class range_scaler(nn.Module):
    def __init__(self, shape, a_init=1., b_init=0., save_target_stats=False, trainable_a_b=False, range_a=[1e-20, np.inf], range_b=[-np.inf, np.inf], mask=None):
        super().__init__()    
        self.range_a, self.range_b = range_a, range_b   
        self.register_parameter('a', nn.Parameter(torch.ones(shape)*a_init, requires_grad=trainable_a_b))
        self.register_parameter('b',  nn.Parameter(torch.ones(shape)*b_init, requires_grad=trainable_a_b))
        self.alpha = 0.1
        self.register_buffer('std_after_scale', torch.ones(shape), persistent=False) # with persistent=False, the parameter won't be transmitted from a model to another
        self.register_buffer('mean_after_scale', torch.zeros(shape), persistent=False)
        self.register_buffer('target_std', torch.zeros(shape), persistent=True) # with persistent=True, the parameter will be transmitted from a model to another
        self.register_buffer('target_mean', torch.zeros(shape), persistent=True)
        self.calibration = False
        self.save_target_stats = save_target_stats
        if mask is not None:
            self.register_buffer("mask", mask)

    def forward(self, x, apply_mask=False):
        a_, b_ = torch.clamp(self.a, min=self.range_a[0], max=self.range_a[1]), torch.clamp(self.b, min=self.range_b[0], max=self.range_b[1])        
        x = a_ * x + b_
        with torch.no_grad():
            if self.calibration:
                if apply_mask:
                    x_ = torch.masked_select(x, self.mask==1)
                else:
                    x_ = x
                self.std_after_scale.copy_(x_.std(dim=[0, 2, 3], keepdim=True))
                self.mean_after_scale.copy_(x_.mean(dim=[0, 2, 3], keepdim=True))
            if self.save_target_stats:
                if apply_mask:
                    x_ = torch.masked_select(x, self.mask)
                else:
                    x_ = x
                self.target_std.copy_(self.alpha * x_.std(dim=[0, 2, 3], keepdim=True) + (1-self.alpha) * self.target_std) # Save statistics to use for another model
                self.target_mean.copy_(self.alpha * x_.mean(dim=[0, 2, 3], keepdim=True) + (1-self.alpha) * self.target_mean)
        return x    
        
class ReadNoise(nn.Module):
    def __init__(self, size: list, level: float=0.1, mask: torch.tensor=None):
        super().__init__()
        self.level = level
        self.alpha = 0.1
        self.std_dev = None
        self.register_buffer('noise', torch.randn(size), persistent=False)
        self.register_buffer('snr', 10*np.log10(1/self.alpha)*torch.ones(size[:-1]), persistent=False)
        self.n_samples = size[0]
        if mask is not None:
            self.noise[mask.expand(size[0], size[1], -1, -1)==0] = 0.
        
    def forward(self, x):
        if not(self.training):
            with torch.no_grad():            
                n_samples, n_heads, n_tokens, n_features = x.shape      
                x = torchaudio.functional.add_noise(waveform=x,
                                                        noise=self.noise[torch.randperm(self.n_samples)[:n_samples]][:, :, :, torch.randperm(n_features)],
                                                        snr=self.snr[:n_samples]
                                                        )
        return x
    


class quantization_custom(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, apply, n_levels, Vmin, Vmax):
        """
        this function simulaties the DAC operation by quantizing the input Inp to resolution bits
        """
        if apply:
            q_step = (Vmax-Vmin)/(n_levels-1)
            return q_step * torch.round(x/q_step)
        else:
            return x
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None, None, None, None
    
class BinsCount(nn.Module):
    def __init__(self, apply, n_levels, range=None, mask=None):
        super().__init__()
        self.apply_quantization = apply
        if apply:
            # self.range = [0., 1.]
            self.range = range
            if self.range is not None: # custom bining
                Vmin, Vmax = self.range
                q_step = (Vmax-Vmin)/(n_levels-1)
                self.register_n_levels = n_levels
                self.density = torch.tensor(0., device='cpu')
                self.bins_edges = torch.linspace(Vmin-q_step/2, Vmax+q_step/2, n_levels+1, device='cpu')
                self.bins_values = torch.arange(Vmin, Vmax+q_step, q_step, device='cpu')
                self.bins = self.bins_edges
            else:                      # number of bins
                self.bins_edges = torch.Tensor([0.])
                self.bins_values = torch.Tensor([0.])
                self.bins = n_levels
                self.data = torch.Tensor([0.])
            self.mask = mask
            
    def forward(self, x):
        if self.apply_quantization:
            with torch.no_grad():                
                if self.mask is not None:
                    B, H, seq_len, context_len = x.shape
                    x_ = x[self.mask.expand(B, H, -1, -1)[:, :, :seq_len, :context_len]!=0].clone()
                    self.density, bins_edges = torch.histogram(x_.to('cpu'), self.bins)
                    self.density /= x_.numel() # density
                    self.data = x_.flatten().to('cpu')                    
                else:                    
                    self.density, bins_edges = torch.histogram(x.to('cpu'), self.bins)
                    self.density /= x.numel() # density
                    self.data = x.flatten().to('cpu')                    
                if self.range is None:
                    self.bins_edges = bins_edges
                    self.bins_values = (bins_edges[1:] + bins_edges[:-1]) / 2
        return x

class DRAM_MAC_temporal_encoding(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y, QV_mul, decay_mask):
        c = [0.17393044,  0.15653739,  0.14088365,  0.12679529,  5.51975209,  4.96777688,  4.4709992,  -1.44776001, -1.30298401, 46.05483778]
        max_order = int((2 * len(c))**0.5 - 1)
        x_max = 0.9
        offset = 0.45
        y = y - offset
        idx = 0 
        for i in range(max_order+1):
            for j in range(max_order - i + 1):
                if QV_mul:
                    if idx == 0:
                        basis_sum = torch.matmul(x, y.pow(i)*x_max**j*c[idx]) * decay_mask
                    else:
                        basis_sum.add_(torch.matmul(x, y.pow(i)*x_max**j*c[idx]) * decay_mask)
                else:
                    if idx == 0:
                        basis_sum = torch.matmul(x * decay_mask.pow(i), y.pow(i)*x_max**j*c[idx])
                    else:
                        basis_sum.add_(torch.matmul(x * decay_mask.pow(i), y.pow(i)*x_max**j*c[idx]))
                idx += 1     
        return basis_sum
    
class DRAM_MAC_temporal_encoding_surrogate_QK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, decay_mask):
        ctx.save_for_backward(x, y, decay_mask)
        c = [0.17393044,  0.15653739,  0.14088365,  0.12679529,  5.51975209,  4.96777688,  4.4709992,  -1.44776001, -1.30298401, 46.05483778]
        max_order = int((2 * len(c))**0.5 - 1)
        x_max = 0.9
        offset = 0.45
        y = y - offset
        idx = 0 
        for i in range(max_order+1):
            for j in range(max_order - i + 1):
                if idx == 0:
                    basis_sum = torch.matmul(x, y.pow(i)*x_max**j*c[idx]) * decay_mask.pow(i)
                else:
                    basis_sum.add_(torch.matmul(x, y.pow(i)*x_max**j*c[idx]) * decay_mask.pow(i))
                idx += 1     
        return basis_sum
    
    @staticmethod
    def backward(ctx, grad_output): # Linear fit: z = a*(x-offset)*(y_offset) + b
        a = 19.3
        offset = 0.45
        x, y, decay_mask = ctx.saved_tensors
        y = y - offset
        grad_x = a * torch.matmul(grad_output.clone() * decay_mask, y.transpose(-1, -2).to(grad_output.dtype))
        grad_y = a * torch.matmul(x.transpose(-1, -2).to(grad_output.dtype), grad_output.clone() * decay_mask)
        return grad_x, grad_y, None, None
    
class DRAM_MAC_temporal_encoding_surrogate_AV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, decay_mask):
        ctx.save_for_backward(x, y, decay_mask)
        c = [0.17393044,  0.15653739,  0.14088365,  0.12679529,  5.51975209,  4.96777688,  4.4709992,  -1.44776001, -1.30298401, 46.05483778]
        max_order = int((2 * len(c))**0.5 - 1)
        x_max = 0.9
        offset = 0.45
        y = y - offset
        idx = 0 
        for i in range(max_order+1):
            for j in range(max_order - i + 1):
                if idx == 0:
                    basis_sum = torch.matmul(x * decay_mask.pow(i), y.pow(i)*x_max**j*c[idx])
                else:
                    basis_sum.add_(torch.matmul(x * decay_mask.pow(i), y.pow(i)*x_max**j*c[idx]))
                idx += 1     
        return basis_sum
    
    @staticmethod
    def backward(ctx, grad_output): # Linear fit: z = a*(x-offset)*(y_offset) + b
        a = 19.3
        offset = 0.45
        x, y, decay_mask = ctx.saved_tensors
        y = y - offset
        grad_x = a * torch.matmul(grad_output.clone(), y.transpose(-1, -2).to(grad_output.dtype)) * decay_mask
        grad_y = a * torch.matmul((x * decay_mask).transpose(-1, -2).to(grad_output.dtype), grad_output.clone())
        return grad_x, grad_y, None, None
    
class mask_to_value(nn.Module):
    def __init__(self, mask: torch.tensor, value: float=float('-inf')):
        super().__init__()
        self.value = value
        self.register_buffer('mask', mask, persistent=False)
        self.n_tokens = mask.shape[-2]
        
    def forward(self, x, t):
        n_samples, n_heads, sequence_length, context_length = x.shape
        x = x.masked_fill(self.mask[:, :, :sequence_length, :context_length] == 0, self.value)  
        return x

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        # self.attn = getattr(getattr(__import__(__name__), os.path.basename(__file__)[:-len(".py")]), config.attention)(config)
        self.attn = eval(config.attention)(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
def remove_state_dict_prefix(state_dict, prefix='_orig_mod.'):
    new_state_dict = state_dict.copy()
    for (key, value) in state_dict.items():
        if key.startswith(prefix):
            del new_state_dict[key]
            new_state_dict.update({key[len(prefix):]: value})
    return new_state_dict


from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

class DRAMGPT2BaseConfig(PretrainedConfig):
# class DRAMGPT2Config(PretrainedConfig):
    model_type = None # "dram-gpt2" # "dram_gpt2"

    def __init__(
        self,
        block_size: int = 1024,
        vocab_size: int = 50304, # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        bias: bool = True, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        attention: str = "CausalSelfAttention",
        quantization_levels_input: int = 2**32,
        quantization_levels_weights: int = 2**32,
        quantization_levels_output: int = 2**32,
        decay_factor: float = 0.,
        sliding_window_size: int = 1024,
        batch_size: int = 32,
        **kwargs,
    ):
        # assert model_type in ["gpt2", "gpt2-xl", "gpt2-LinearDRAMAttention", "gpt2-DRAMAttention"]
        # self.model_type = model_type
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.attention = attention
        self.quantization_levels_input = quantization_levels_input
        self.quantization_levels_weights = quantization_levels_weights
        self.quantization_levels_output = quantization_levels_output
        self.decay_factor = decay_factor
        self.sliding_window_size = sliding_window_size
        self.batch_size = batch_size
        super().__init__(**kwargs)


def remove_state_dict_prefix(state_dict, prefix='_orig_mod.'):
    new_state_dict = state_dict.copy()
    for (key, value) in state_dict.items():
        if key.startswith(prefix):
            del new_state_dict[key]
            new_state_dict.update({key[len(prefix):]: value})
    return new_state_dict


def load_model(root, dram_model_type, batch_size=None, device=None, additional_suffix=None):
    assert dram_model_type.startswith('dram-')
    assert dram_model_type in DRAM_GPT2_MODEL_DICT.keys()
    # model_type = dram_model_type[len('dram-1_'):]
    
    # # name structure : dram-1_gpt-2-DRAMAttention_adopted...
    # model_type = dram_model_type.split('_')[1]
    init_from = dram_model_type[len("dram-"):]

    # model_cls = DRAM_GPT2_MODEL_DICT[model_type]
    # config_cls = model_cls.config_class

    
    # # huggingface_models = ['gpt2-medium', 'gpt2-large']
    # # local_models = ['gpt2', 'gpt2-from-scratch', 'gpt2-xl', 'gpt2-LinearDRAMAttention', 'gpt2-xl-LinearDRAMAttention', 'gpt2-DRAMAttention']
    # local_models = 
    # assert model_type in local_models

    # override_args = override_args or {} # default to empty dict
    # # only dropout can be overridden see more notes below
    # # assert all(k == 'dropout' for k in override_args)
    # if model_type in huggingface_models:
    #     raise ValueError("Huggingface models are not supported for DRAM-GPT2")

    model_sd = torch.load(os.path.join(root, f"{init_from}{additional_suffix or ''}.pt"), map_location="cpu")
    print("model_sd.keys():", model_sd.keys())
    sd_hf = model_sd['model']
    config_args = model_sd['model_args']
    overide_args = dict(
        batch_size=batch_size or 1,
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
    [config_args.update({k:overide_args[k]}) if k in overide_args else None for (k, v) in config_args.items()]

    model_cls = DRAM_GPT2_MODEL_DICT[dram_model_type]
    config_cls = model_cls.config_class

    config = config_cls(**config_args)
    model = model_cls(config)

    # [sd_hf.update({k: torch.tensor(0)}) for k in sd_hf if k.endswith('.iter_num')] # We want to reinitialize this parameter
    with torch.no_grad():
        sd_hf = remove_state_dict_prefix(sd_hf)
        model.load_state_dict(sd_hf, strict=False)
        # model.load_state_dict(sd_hf)
    if device is not None:
        model.to(device)
    return model


class DRAMGPT2Base(PreTrainedModel):
    config_class = None # DRAMGPT2Config

    def __init__(self, config):
        super().__init__(config)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.tensor, attention_mask=None, targets: torch.tensor=None):
        idx = input_ids
        if attention_mask is not None:
            print("WARNING: attention_mask not supported in this model, it will be ignored")
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for b, block in enumerate(self.transformer.h):
            x = block(x)
            # print(b, torch.isnan(x).sum())
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = self.lm_head(x[:, :, :]) # note: using list [-1] to preserve the time dim
            loss = None

        return CausalLMOutput(loss, logits)

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, dram_model_type: str, override_args=None, **kwargs):
        # NOTE: HARDCODED MODEL PATHS
        if "gpt2-xl" in dram_model_type:
            model_path = "/Data/pgi-15/common_models/dram_attention_project/ncs_rebuttal/"
        else:
            model_path = "/Data/pgi-15/common_models/dram_attention_project/ncs_submission/"
            
        batch_size_model = kwargs.pop("batch_size_model", None)
        additional_suffix = kwargs.pop("additional_suffix", None)
        model = load_model(model_path, dram_model_type, batch_size=batch_size_model, device=None, additional_suffix=additional_suffix) # 'cuda')
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        idx_next = idx
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class DRAMGPT2Config(DRAMGPT2BaseConfig):
    model_type = "dram-1_gpt2_public"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class DRAMGPT2FromScratchConfig(DRAMGPT2BaseConfig):
    model_type = "dram-2_gpt2-from-scratch"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class DRAMGPT2_XL_DRAMAttention_Config(DRAMGPT2BaseConfig):
    model_type = "dram-gpt2-xl-DRAMAttention"

    def __init__(self, **kwargs):
        attention = kwargs.pop('attention', "DRAMAttention")
        assert attention == "DRAMAttention", f"attention must be DRAMAttention, got '{attention}'"
        super().__init__(attention=attention, **kwargs)

class DRAMGPT2_LinearDRAMAttention_Config(DRAMGPT2BaseConfig):
    model_type = "dram-3_gpt2-LinearDRAMAttention"

    def __init__(self, **kwargs):
        attention = kwargs.pop('attention', "LinearDRAMAttention")
        assert attention == "LinearDRAMAttention", f"attention must be LinearDRAMAttention, got '{attention}'"
        super().__init__(attention=attention, **kwargs)

class DRAMGPT2_XL_LinearDRAMAttention_Config(DRAMGPT2BaseConfig):
    model_type = "dram-gpt2-xl-LinearDRAMAttention"

    def __init__(self, **kwargs):
        attention = kwargs.pop('attention', "LinearDRAMAttention")
        assert attention == "LinearDRAMAttention", f"attention must be LinearDRAMAttention, got '{attention}'"
        super().__init__(attention=attention, **kwargs)

class DRAMGPT2_DRAMAttention_Config(DRAMGPT2BaseConfig):
    model_type = "dram-gpt2-DRAMAttention"

    def __init__(self, **kwargs):
        attention = kwargs.pop('attention', "DRAMAttention")
        assert attention == "DRAMAttention", f"attention must be DRAMAttention, got '{attention}'"
        super().__init__(attention=attention, **kwargs)


class DRAMGPT2_DRAMAttention_adaptNoFinet_Config(DRAMGPT2_DRAMAttention_Config):
    model_type = "dram-4_gpt2-DRAMAttention_adapted_from_LinearDRAMAttention_no_fine-tuning"


class DRAMGPT2_DRAMAttention_adaptWithFinet_Config(DRAMGPT2_DRAMAttention_Config):
    model_type = "dram-5_gpt2-DRAMAttention_adapted_from_LinearDRAMAttention+fine-tuning"

class DRAMGPT2(DRAMGPT2Base):
    config_class = DRAMGPT2Config

class DRAMGPT2FromScratch(DRAMGPT2Base):
    config_class = DRAMGPT2FromScratchConfig

class DRAMGPT2_XL_DRAMAttention(DRAMGPT2Base):
    config_class = DRAMGPT2_XL_DRAMAttention_Config

class DRAMGPT2_LinearDRAMAttention(DRAMGPT2Base):
    config_class = DRAMGPT2_LinearDRAMAttention_Config

class DRAMGPT2_XL_LinearDRAMAttention(DRAMGPT2Base):
    config_class = DRAMGPT2_XL_LinearDRAMAttention_Config

class DRAMGPT2_DRAMAttention(DRAMGPT2Base):
    config_class = DRAMGPT2_DRAMAttention_Config

class DRAMGPT2_DRAMAttention_adaptNoFinet(DRAMGPT2Base):
    config_class = DRAMGPT2_DRAMAttention_adaptNoFinet_Config

class DRAMGPT2_DRAMAttention_adaptWithFinet(DRAMGPT2Base):
    config_class = DRAMGPT2_DRAMAttention_adaptWithFinet_Config


DRAM_GPT2_MODEL_DICT : dict[str, DRAMGPT2Base] = {
    "dram-1_gpt2_public": DRAMGPT2,
    "dram-2_gpt2-from-scratch": DRAMGPT2FromScratch,
    "dram-3_gpt2-LinearDRAMAttention": DRAMGPT2_LinearDRAMAttention,
    "dram-4_gpt2-DRAMAttention_adapted_from_LinearDRAMAttention_no_fine-tuning": DRAMGPT2_DRAMAttention_adaptNoFinet,
    "dram-5_gpt2-DRAMAttention_adapted_from_LinearDRAMAttention+fine-tuning": DRAMGPT2_DRAMAttention_adaptWithFinet,
    "dram-gpt2-xl-LinearDRAMAttention": DRAMGPT2_XL_LinearDRAMAttention,
    "dram-gpt2-xl-DRAMAttention": DRAMGPT2_XL_DRAMAttention,
}
    

def register_dram_gpt2_hf_models():
    # In order to not look for cached files but just read local config
    transformers.utils.import_utils._peft_available = False
    
    for model_name in [
        "dram-1_gpt2_public", 
        "dram-2_gpt2-from-scratch", 
        "dram-3_gpt2-LinearDRAMAttention", 
        "dram-4_gpt2-DRAMAttention_adapted_from_LinearDRAMAttention_no_fine-tuning", 
        "dram-5_gpt2-DRAMAttention_adapted_from_LinearDRAMAttention+fine-tuning",
        "dram-gpt2-xl-LinearDRAMAttention",
        "dram-gpt2-xl-DRAMAttention"
    ]:
        model_cls = DRAM_GPT2_MODEL_DICT[model_name]
        config_cls = model_cls.config_class
        AutoConfig.register(model_name, config_cls)
        AutoModel.register(config_cls, model_cls)
        AutoModelForCausalLM.register(config_cls, model_cls)

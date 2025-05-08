from transformers import AutoModelForCausalLM
import torch
import os
import sys
dir_name = os.getcwd()
sys.path.insert(0, dir_name)

model_name = sys.argv[0]

model = AutoModelForCausalLM.from_pretrained('openai-community/' + model_name)
torch.save(model.state_dict(), '../saved_models/' + model_name + '.pt')

# checkpoint_path = '../saved_models/checkpoints/DRAMFlashAttention_ft_from_smallm_no_trainable_b_qkv_LN_out_RMS_norm_out_quant_497_linearDRAMFlashAttention_stop_10000'
# checkpoint_path = os.path.join(checkpoint_path,
#                             'ckpt.pt')
# safe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint_path)
# torch.serialization.add_safe_globals(safe_globals)
# local_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

# del local_checkpoint['optimizer']

# output_dir = '/Data/pgi-15/common_models/dram_attention_project/ncs_rebuttal/'
# new_name = 'smollm_DRAMAttention_13000_iters.pt'

# torch.save(local_checkpoint, os.path.join(output_dir, new_name))

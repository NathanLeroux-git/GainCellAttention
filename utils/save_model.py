from transformers import AutoModelForCausalLM
import torch
import os
import sys
dir_name = os.getcwd()
sys.path.insert(0, dir_name)

model_name = sys.argv[1]
model_name = model_name[2:]

model = AutoModelForCausalLM.from_pretrained('openai-community/' + model_name)
torch.save(model.state_dict(), '../saved_models/' + model_name + '.pt')

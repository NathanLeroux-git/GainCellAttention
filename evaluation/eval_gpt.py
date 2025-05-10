import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from modules.model_gpt_eval import register_dram_gpt2_hf_models
register_dram_gpt2_hf_models()

import sys
import torch
import torch.nn as nn

from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()

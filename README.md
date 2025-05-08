# GainCellAttention

This repository contains the codes used to train the models discussed in the article :

N. Leroux, P.-P. Manea, C. Sudarshan, J. Finkbeiner, S. Siegel, J. P. Strachan, and E. Neftci, Analog In-Memory Computing Attention Mechanism for Fast and Energy-Efficient Large Language Models, arXiv:2409.19315.
https://arxiv.org/abs/2409.19315.

## Update 06/05/2025.
New FlashAttention Triton implementation adapted for gain cells simulations (flash_attention/flash_attn/DRAM_flash_attn_triton.py). \
Use arguments --attention=LinearDRAMFlashAttention or --attention=DRAMFlashAttention \
Improved main_gpt.py \
Added utils/llms_training_utils.py and dataloaders/llms_training_loaders.py \
New experiment to train 1.5B parameters gpt2-xl (see configs/experiments/gpt-text/train_gpt2-xl.py) \
Added utils/save_model.py --gpt2 to save models locally automatically

## Installation guide (estimated time < 1h)

### Environement pre-requisit: Linux, python > 3.0, pip > 22.0
### Create project directory and virtual environement and activate

mkdir new_project \
cd new_project \
mkdir ./venv/ \
python -m venv ./venv/gpt2 \
git clone https://github.com/NathanLeroux-git/GainCellAttention.git \
cd GainCellAttention/ \
source ../venv/gpt2/bin/activate

### Install Pytorch (done with torch 2.2.2, need to adapt cuda version depending on system)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

### Install other required packages
pip install -r requirements.txt

### Preparing the datasets
python dataloaders/texts/shakespeare/prepare.py \
python dataloaders/texts/openwebtext/prepare.py

### Download and save gpt models locally
mkdir -p ../saved_models/ \
python utils/save_model.py --gpt2 (or --gpt2-xl)

## Training the models (total estimated time: ~4 days on 8 Nvidia H100 80 Gb GPUs, ~16 days on 8 Nvidia RTX 4090 24 Gb)
main_gpt.py automatically saved the results on wandb.ai (use --wandb_log=True and --wandb_offline=False). The trained models are checkpointed in "../saved_models/checkpoints".

### run training example on a tiny Shakespeare texts dataset
python -m main_gpt ./configs/experiments/gpt-text/finetune_shakespeare.py --wandb_log=True --wandb_offline=False --batch_size=8

### run training intermediate model on OpenWebText
Distributed run with 4 GPUs on 3000 iterations to train the intermediate gain cells model fine-tuned from gpt2:

python -m torch.distributed.run --nproc_per_node 4 main_gpt.py ./configs/experiments/gpt-text/ft_lineardram_gpt2.py --wandb_log=True --wandb_offline=False --init_from='gpt2' --stop_saving_after=3000. --max_iters=3001 --out_dir='../saved_models/checkpoints/gpt2_LinearDRAMAttention'

### fine-tuning the final model
After training the intermediate model, we need to change the saved model name to ../saved_models/gpt2-LinearDRAMAttention.pt. For instance:

cp ../saved_models/checkpoints/gpt2_LinearDRAMAttention/ckpt.pt ../saved_models/gpt2-LinearDRAMAttention.pt

Finally, we can fine-tune the intermediate model on the final gain cells model (the adaptation algorithm is operated by main_gpt.py):

python -m torch.distributed.run --nproc_per_node 4 main_gpt.py ./configs/experiments/gpt-text/ft_dram_gpt2.py --wandb_log=True --wandb_offline=False --init_from='gpt2-LinearDRAMAttention' --stop_saving_after=13000. --max_iters=13001 --out_dir='../saved_models/checkpoints/gpt2_DRAMAttention'

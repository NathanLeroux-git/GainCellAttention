# GainCellAttention

## Installation guide

### Environement pre-requisit: Linux, python > 3.0, pip > 22.0
### Create project directory and virtual environement and activate

mkdir new_project \
cd new_project \
mkdir ./venv/ \
python -m venv ./venv/gpt2 \
git clone https://github.com/NathanLeroux-git/GainCellAttention.git \
cd GainCellAttention/ \
source ../venv/gpt2/bin/activate \

### Install Pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

### Install required packages
pip install -r requirements.txt

### run training example on a tiny Shakespeare texts dataset
First need to run the command: \
python datasets/texts/shakespeare/prepare.py to prepare the tokenized text, and then run with: \
python -m main_gpt ./configs/experiments/gpt-text/finetune_shakespeare.py --wandb_log=False --wandb_offline=False


### run training on OpenWebText
First need to run the command: \
python datasets/texts/openwebtext/prepare.py to prepare the tokenized text \
python -m torch.distributed.run --nproc_per_node 4 main_gpt.py ./configs/experiments/gpt-text/ft_lineardram_gpt2.py --wandb_log=True --wandb_offline=False

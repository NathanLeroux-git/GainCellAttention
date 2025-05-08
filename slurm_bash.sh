#!/bin/bash -x
#SBATCH --ntasks=1
#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --partition=pgi15
#SBATCH --gpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32

#SBATCH --output=../slurm/slurm-out.%j
#SBATCH --error=../slurm/slurm-err.%j

source ../venv/gpt2/bin/activate

### Tests
batch_size=4
gradient_accumulation_steps=16
n_gpus=4
run_name="test"
output_dir="../saved_models/checkpoints/${run_name}"
srun --cpu-bind=no python -m torch.distributed.run --nproc_per_node $n_gpus main_gpt.py configs/experiments/gpt-text/ft_lineardram_gpt2.py --out_dir="${output_dir}" --wandb_run_name="${run_name}" --wandb_log=False --wandb_offline=False --batch_size=${batch_size} --gradient_accumulation_steps=${gradient_accumulation_steps} --stop_saving_after=200. --max_iters=201 --attention=LinearDRAMFlashAttention
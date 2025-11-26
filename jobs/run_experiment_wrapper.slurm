#!/bin/bash
#SBATCH --job-name=visreas-exp
#SBATCH -D ./
#SBATCH --output=visreas-exp_job_%j.log
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/data/ktroyan
# #SBATCH --partition=gpu              # does the site require one to be specified?
# #SBATCH --account=MY_PROJECT         # does the site require one to be specified?

# Activate environment
source .venv/bin/activate 2>/dev/null || true

# Debug info
echo "Running on node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES from SLURM: $CUDA_VISIBLE_DEVICES"
nvidia-smi || true

# Set WANDB API key (better to do it like this here for SLURM)
export WANDB_API_KEY=$(cat ~/.secrets/wandb_api_key)

# Call existing script in SLURM mode and forward all args
bash ./jobs/run_experiment_nohup.sh --slurm "$@"

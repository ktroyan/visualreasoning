#!/bin/bash
#SBATCH --job-name=visreas-exp
#SBATCH -D ./
#SBATCH --output=slurm_%x_%j.out          # SLURM's own log (with custom file)
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                       # change to gpu:2, etc. if needed
#SBATCH --output=visreas-exp_%j.log          # Output log file
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/data/ktroyan

# ---- Activate environment ----
# For uv with a local .venv
source .venv/bin/activate 2>/dev/null || true

# Debug info
echo "Running on node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES from SLURM: $CUDA_VISIBLE_DEVICES"
nvidia-smi || true

# ---- Call existing bash script in SLURM mode and forward all args ----
bash ./jobs/run_experiment_nohup.sh --slurm "$@"

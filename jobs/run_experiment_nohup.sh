#!/bin/bash

# Default values for variables
GPU_ID=1
DATA_ENV="BEFOREARC"
STUDY="sample-efficiency"
SETTING="exp_setting_1"
EXPERIMENT="experiment_1"
MAX_EPOCHS=10
BACKBONE="vit"
HEAD="mlp"
WANDB_SWEEP_ENABLED=false
ADDI_LOG_NAME=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu_id) GPU_ID="$2"; shift 2 ;;
        --data_env) DATA_ENV="$2"; shift 2 ;;
        --study) STUDY="$2"; shift 2 ;;
        --setting) SETTING="$2"; shift 2 ;;
        --experiment) EXPERIMENT="$2"; shift 2 ;;
        --max_epochs) MAX_EPOCHS="$2"; shift 2 ;;
        --backbone) BACKBONE="$2"; shift 2 ;;
        --head) HEAD="$2"; shift 2 ;;
        --sweep_enabled) WANDB_SWEEP_ENABLED="$2"; shift 2 ;;
        --add_log_name) ADDI_LOG_NAME="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set CUDA device
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Construct log filename
LOG_FILE="output_${DATA_ENV}_${STUDY}_${SETTING}_${EXPERIMENT}_${BACKBONE}_${HEAD}${ADDI_LOG_NAME}.log"

# Run the command in the background using nohup
nohup uv run experiment.py \
    base.gpu_id="${GPU_ID}" \
    base.data_env="${DATA_ENV}" \
    experiment.study="${STUDY}" \
    experiment.setting="${SETTING}" \
    experiment.name="${EXPERIMENT}" \
    training.max_epochs="${MAX_EPOCHS}" \
    model.backbone="${BACKBONE}" \
    model.head="${HEAD}" \
    wandb.sweep.enabled="${WANDB_SWEEP_ENABLED}" \
    > "${LOG_FILE}" 2>&1 &

# Report success
echo "Launched experiment with PID $!"
echo "Logs are being written to: ${LOG_FILE}"

## Run the bash script
# For example:
# bash ./jobs/run_experiment_nohup.sh --data_env "BEFOREARC" --study "sample-efficiency" --setting "exp_setting_1" --experiment "experiment_16" --max_epochs 10 --backbone "vit" --head "mlp" --sweep_enabled false --gpu_id 0 --add_log_name ""
#!/bin/bash

# Initialize variables as empty string (i.e., no defaults)
GPU_ID=""
DATA_ENV=""
STUDY=""
SETTING=""
EXPERIMENT=""
MAX_EPOCHS=""
BACKBONE=""
HEAD=""
WANDB_SWEEP_ENABLED=""
ADDI_LOG_NAME=""

# Parse CLI arguments
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

# Set CUDA device only if provided
if [[ -n "$GPU_ID" ]]; then
    export CUDA_VISIBLE_DEVICES=${GPU_ID}
fi

# Construct log filename
LOG_FILE="output_${DATA_ENV}_${STUDY}_${SETTING}_${EXPERIMENT}_${BACKBONE}_${HEAD}_${ADDI_LOG_NAME}.log"

# Construct command
CMD="nohup uv run experiment.py"

[[ -n "$GPU_ID" ]] && CMD+=" base.gpu_id=\"${GPU_ID}\""
[[ -n "$DATA_ENV" ]] && CMD+=" base.data_env=\"${DATA_ENV}\""
[[ -n "$STUDY" ]] && CMD+=" experiment.study=\"${STUDY}\""
[[ -n "$SETTING" ]] && CMD+=" experiment.setting=\"${SETTING}\""
[[ -n "$EXPERIMENT" ]] && CMD+=" experiment.name=\"${EXPERIMENT}\""
[[ -n "$MAX_EPOCHS" ]] && CMD+=" training.max_epochs=\"${MAX_EPOCHS}\""
[[ -n "$BACKBONE" ]] && CMD+=" model.backbone=\"${BACKBONE}\""
[[ -n "$HEAD" ]] && CMD+=" model.head=\"${HEAD}\""
[[ -n "$WANDB_SWEEP_ENABLED" ]] && CMD+=" wandb.sweep.enabled=\"${WANDB_SWEEP_ENABLED}\""

CMD+=" > \"${LOG_FILE}\" 2>&1 &"

# Run command
eval $CMD

# Echo messages
echo "Launched experiment with PID $!"
echo "Logs are being written to: ${LOG_FILE}"

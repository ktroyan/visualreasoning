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
VISUAL_TOKENS_ENABLED=""
APE_ENABLED=""
APE_TYPE=""
APE_MIXER=""
OPE_ENABLED=""
RPE_ENABLED=""
RPE_TYPE=""
NUM_REG_TOKENS=""
USE_TASK_EMBEDDING=""
USE_GEN_TEST_SET=""
VALIDATE_IN_AND_OUT_DOMAIN=""
WANDB_SWEEP_ENABLED=""
WANDB_SWEEP_CONFIG=""
ADDI_LOG_NAME=""

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu_id) GPU_ID="$2"; shift 2 ;;
        --data_env) DATA_ENV="$2"; shift 2 ;;
        --study) STUDY="$2"; shift 2 ;;
        --setting) SETTING="$2"; shift 2 ;;
        --experiment) EXPERIMENT="$2"; shift 2 ;;
        --use_gen_test_set) USE_GEN_TEST_SET="$2"; shift 2 ;;
        --validate_in_and_out_domain) VALIDATE_IN_AND_OUT_DOMAIN="$2"; shift 2 ;;
        --max_epochs) MAX_EPOCHS="$2"; shift 2 ;;
        --backbone) BACKBONE="$2"; shift 2 ;;
        --head) HEAD="$2"; shift 2 ;;
        --visual_tokens_enabled) VISUAL_TOKENS_ENABLED="$2"; shift 2 ;;
        --ape_enabled) APE_ENABLED="$2"; shift 2 ;;
        --ape_type) APE_TYPE="$2"; shift 2 ;;
        --ape_mixer) APE_MIXER="$2"; shift 2 ;;
        --ope_enabled) OPE_ENABLED="$2"; shift 2 ;;
        --rpe_enabled) RPE_ENABLED="$2"; shift 2 ;;
        --rpe_type) RPE_TYPE="$2"; shift 2 ;;
        --num_reg_tokens) NUM_REG_TOKENS="$2"; shift 2 ;;
        --use_task_embedding) USE_TASK_EMBEDDING="$2"; shift 2 ;;
        --sweep_enabled) WANDB_SWEEP_ENABLED="$2"; shift 2 ;;
        --sweep_config) WANDB_SWEEP_CONFIG="$2"; shift 2 ;;
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

# Generate timestamp: format day-month-hour_min
TIMESTAMP=$(date +"%d-%m-%H_%M")

# Determine log file name
if [[ -n "$WANDB_SWEEP_CONFIG" ]]; then
    SWEEP_BASENAME="${WANDB_SWEEP_CONFIG##*/}"              # remove directory path
    SWEEP_NO_EXT="${SWEEP_BASENAME%%.*}"                    # remove file extension
    SWEEP_SAFE_NAME="${WANDB_SWEEP_CONFIG//\//_}"           # replace "/" with "_"
    SWEEP_SAFE_NAME="${SWEEP_SAFE_NAME%%.*}"                # remove extension from whole path
    LOG_FILE="output_${SWEEP_SAFE_NAME}_${TIMESTAMP}.log"
else
    LOG_FILE="output_${DATA_ENV}_${STUDY}_${SETTING}_${EXPERIMENT}_${BACKBONE}_${HEAD}_${ADDI_LOG_NAME}_${TIMESTAMP}.log"
fi

# Construct command
CMD="nohup uv run experiment.py"

[[ -n "$GPU_ID" ]] && CMD+=" base.gpu_id=\"${GPU_ID}\""
[[ -n "$DATA_ENV" ]] && CMD+=" base.data_env=\"${DATA_ENV}\""
[[ -n "$STUDY" ]] && CMD+=" experiment.study=\"${STUDY}\""
[[ -n "$SETTING" ]] && CMD+=" experiment.setting=\"${SETTING}\""
[[ -n "$EXPERIMENT" ]] && CMD+=" experiment.name=\"${EXPERIMENT}\""
[[ -n "$USE_GEN_TEST_SET" ]] && CMD+=" data.use_gen_test_set=\"${USE_GEN_TEST_SET}\""
[[ -n "$VALIDATE_IN_AND_OUT_DOMAIN" ]] && CMD+=" data.validate_in_and_out_domain=\"${VALIDATE_IN_AND_OUT_DOMAIN}\""
[[ -n "$MAX_EPOCHS" ]] && CMD+=" training.max_epochs=\"${MAX_EPOCHS}\""
[[ -n "$BACKBONE" ]] && CMD+=" model.backbone=\"${BACKBONE}\""
[[ -n "$HEAD" ]] && CMD+=" model.head=\"${HEAD}\""
[[ -n "$VISUAL_TOKENS_ENABLED" ]] && CMD+=" model.visual_tokens.enabled=\"${VISUAL_TOKENS_ENABLED}\""
[[ -n "$APE_ENABLED" ]] && CMD+=" model.ape.enabled=\"${APE_ENABLED}\""
[[ -n "$APE_TYPE" ]] && CMD+=" model.ape.ape_type=\"${APE_TYPE}\""
[[ -n "$APE_MIXER" ]] && CMD+=" model.ape.mixer=\"${APE_MIXER}\""
[[ -n "$OPE_ENABLED" ]] && CMD+=" model.ope.enabled=\"${OPE_ENABLED}\""
[[ -n "$RPE_ENABLED" ]] && CMD+=" model.rpe.enabled=\"${RPE_ENABLED}\""
[[ -n "$RPE_TYPE" ]] && CMD+=" model.rpe.rpe_type=\"${RPE_TYPE}\""
[[ -n "$NUM_REG_TOKENS" ]] && CMD+=" model.num_reg_tokens=\"${NUM_REG_TOKENS}\""
[[ -n "$USE_TASK_EMBEDDING" ]] && CMD+=" model.task_embedding.enabled=\"${USE_TASK_EMBEDDING}\""
[[ -n "$WANDB_SWEEP_ENABLED" ]] && CMD+=" wandb.sweep.enabled=\"${WANDB_SWEEP_ENABLED}\""
[[ -n "$WANDB_SWEEP_CONFIG" ]] && CMD+=" wandb.sweep.config=\"${WANDB_SWEEP_CONFIG}\""

CMD+=" > \"${LOG_FILE}\" 2>&1 &"

# Run command
eval $CMD

# Echo messages
echo "Launched experiment with PID $!"
echo "Logs are being written to: ${LOG_FILE}"
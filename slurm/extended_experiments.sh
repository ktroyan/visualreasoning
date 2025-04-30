#!/bin/bash

# Define options
data_env="BEFOREARC"
task_embedding_options=("false" "true")
sweep_types=("comp" "sysgen")
experiment_ids=("es1" "es2" "es3" "es4" "es5")

# Lowercase version of data_env for file paths
data_env_lc=$(echo "$data_env" | tr '[:upper:]' '[:lower:]')

# WANDB config
wandb_project="VisReas-project-${data_env}-sweep"
wandb_entity="sagerpascal"

# Build configs dynamically
configs=()

for task_embedding in "${task_embedding_options[@]}"; do
  for sweep_type in "${sweep_types[@]}"; do
    for exp_id in "${experiment_ids[@]}"; do
      config="base.data_env=${data_env} \
model.task_embedding.enabled=${task_embedding} \
wandb.sweep.config=configs/sweeps/${data_env_lc}/${sweep_type}_${exp_id}.yaml \
wandb.wandb_project_name=${wandb_project} \
wandb.wandb_entity_name=${wandb_entity}"
      configs+=("$config")
    done
  done
done

# Submit jobs
for config in "${configs[@]}"; do
  echo sbatch run_experiment.submit \
    data.use_gen_test_set=true \
    data.validate_in_and_out_domain=true \
    wandb.sweep.enabled=true \
    $config
  # sleep 61 # optional: avoid timestamp collision
done

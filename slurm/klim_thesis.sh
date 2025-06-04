#!/bin/bash

# List of experiment configurations
configs=(
  "BEFOREARC compositionality exp_setting_3 experiment_3 2025"
  "BEFOREARC sys-gen exp_setting_5 experiment_5 2025"
)

# Iterate over each configuration and submit the job
for config in "${configs[@]}"; do
  read -r data_env study setting name seed <<< "$config"

  sbatch run_experiment.submit \
    wandb.sweep.enabled=false \
    wandb.wandb_entity_name=VisReas-ETHZ \
    wandb.wandb_project_name=VisReas-project-${data_env}-llada-final \
    base.data_env=${data_env} \
    experiment.study=${study} \
    experiment.setting=${setting} \
    experiment.name=${name} \
    base.seed=${seed}
done

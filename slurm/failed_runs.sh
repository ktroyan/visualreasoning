#!/bin/bash

# List of experiment configurations
configs=(
  "BEFOREARC sys-gen exp_setting_1 experiment_3"
  "BEFOREARC sys-gen exp_setting_2 experiment_3"
  "BEFOREARC sys-gen exp_setting_3 experiment_2"
  "BEFOREARC sys-gen exp_setting_3 experiment_4"
  "BEFOREARC sys-gen exp_setting_4 experiment_3"
  "BEFOREARC sys-gen exp_setting_5 experiment_4"
)

# Iterate over each configuration and submit the job
for config in "${configs[@]}"; do
  read -r data_env study setting name <<< "$config"

  sbatch run_experiment.submit \
    wandb.sweep.enabled=false \
    wandb.wandb_entity_name=VisReas-ETHZ \
    wandb.wandb_project_name=VisReas-project-${data_env}-llada-2 \
    base.data_env=${data_env} \
    experiment.study=${study} \
    experiment.setting=${setting} \
    experiment.name=${name}
done

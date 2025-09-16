#!/bin/bash

# List of experiment configurations
configs=(
  "BEFOREARC compositionality exp_setting_1 experiment_4 1997"
  "BEFOREARC compositionality exp_setting_2 experiment_4 1997"
  "BEFOREARC compositionality exp_setting_3 experiment_4 1997"
  "BEFOREARC compositionality exp_setting_1 experiment_4 42"
  "BEFOREARC compositionality exp_setting_2 experiment_4 42"
  "BEFOREARC compositionality exp_setting_3 experiment_4 42"
  "BEFOREARC compositionality exp_setting_1 experiment_4 2025"
  "BEFOREARC compositionality exp_setting_2 experiment_4 2025"
  "BEFOREARC compositionality exp_setting_3 experiment_4 2025"
)

# Iterate over each configuration and submit the job
for config in "${configs[@]}"; do
  read -r data_env study setting name seed <<< "$config"

  sbatch run_experiment.submit \
    wandb.sweep.enabled=false \
    wandb.wandb_entity_name=VisReas-ETHZ \
    wandb.wandb_project_name=VisReas-project-${data_env}-llada-new-comgen \
    base.data_env=${data_env} \
    experiment.study=${study} \
    experiment.setting=${setting} \
    experiment.name=${name} \
    base.seed=${seed}
done

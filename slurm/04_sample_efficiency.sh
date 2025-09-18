#!/bin/bash

# List of experiment configurations
configs=(
  "BEFOREARC sample-efficiency exp_setting_1 experiment_1 1997"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_2 1997"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_3 1997"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_4 1997"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_5 1997"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_6 1997"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_7 1997"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_8 1997"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_9 1997"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_10 1997"

  "BEFOREARC sample-efficiency exp_setting_2 experiment_1 1997"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_2 1997"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_3 1997"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_4 1997"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_5 1997"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_6 1997"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_7 1997"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_8 1997"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_9 1997"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_10 1997"

  "BEFOREARC sample-efficiency exp_setting_3 experiment_1 1997"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_2 1997"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_3 1997"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_4 1997"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_5 1997"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_6 1997"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_7 1997"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_8 1997"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_9 1997"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_10 1997"

  "BEFOREARC sample-efficiency exp_setting_4 experiment_1 1997"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_2 1997"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_3 1997"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_4 1997"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_5 1997"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_6 1997"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_7 1997"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_8 1997"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_9 1997"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_10 1997"
  
  ## 42
  "BEFOREARC sample-efficiency exp_setting_1 experiment_1 42"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_2 42"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_3 42"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_4 42"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_5 42"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_6 42"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_7 42"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_8 42"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_9 42"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_10 42"

  "BEFOREARC sample-efficiency exp_setting_2 experiment_1 42"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_2 42"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_3 42"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_4 42"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_5 42"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_6 42"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_7 42"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_8 42"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_9 42"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_10 42"

  "BEFOREARC sample-efficiency exp_setting_3 experiment_1 42"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_2 42"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_3 42"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_4 42"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_5 42"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_6 42"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_7 42"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_8 42"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_9 42"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_10 42"

  "BEFOREARC sample-efficiency exp_setting_4 experiment_1 42"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_2 42"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_3 42"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_4 42"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_5 42"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_6 42"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_7 42"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_8 42"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_9 42"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_10 42"
  
  
  ## 2025
  "BEFOREARC sample-efficiency exp_setting_1 experiment_1 2025"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_2 2025"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_3 2025"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_4 2025"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_5 2025"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_6 2025"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_7 2025"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_8 2025"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_9 2025"
  "BEFOREARC sample-efficiency exp_setting_1 experiment_10 2025"

  "BEFOREARC sample-efficiency exp_setting_2 experiment_1 2025"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_2 2025"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_3 2025"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_4 2025"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_5 2025"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_6 2025"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_7 2025"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_8 2025"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_9 2025"
  "BEFOREARC sample-efficiency exp_setting_2 experiment_10 2025"

  "BEFOREARC sample-efficiency exp_setting_3 experiment_1 2025"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_2 2025"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_3 2025"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_4 2025"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_5 2025"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_6 2025"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_7 2025"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_8 2025"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_9 2025"
  "BEFOREARC sample-efficiency exp_setting_3 experiment_10 2025"

  "BEFOREARC sample-efficiency exp_setting_4 experiment_1 2025"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_2 2025"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_3 2025"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_4 2025"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_5 2025"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_6 2025"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_7 2025"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_8 2025"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_9 2025"
  "BEFOREARC sample-efficiency exp_setting_4 experiment_10 2025"
)

# Iterate over each configuration and submit the job
for config in "${configs[@]}"; do
  read -r data_env study setting name seed <<< "$config"

  sbatch run_experiment.submit \
    wandb.sweep.enabled=false \
    wandb.wandb_entity_name=VisReas-ETHZ \
    wandb.wandb_project_name=VisReas-project-${data_env}-llada-se \
    base.data_env=${data_env} \
    experiment.study=${study} \
    experiment.setting=${setting} \
    experiment.name=${name} \
    base.seed=${seed}
done

#!/bin/bash

# List of experiment configurations
configs=(
  "BEFOREARC compositionality exp_setting_1 experiment_1 grid_size_30 1997"
  "BEFOREARC compositionality exp_setting_1 experiment_1 grid_size_40 1997"
  "BEFOREARC compositionality exp_setting_1 experiment_1 grid_size_50 1997"
  "BEFOREARC compositionality exp_setting_1 experiment_2 grid_size_30 1997"
  "BEFOREARC compositionality exp_setting_1 experiment_2 grid_size_40 1997"
  "BEFOREARC compositionality exp_setting_1 experiment_2 grid_size_50 1997"
  "BEFOREARC compositionality exp_setting_1 experiment_3 grid_size_30 1997"
  "BEFOREARC compositionality exp_setting_1 experiment_3 grid_size_40 1997"
  "BEFOREARC compositionality exp_setting_1 experiment_3 grid_size_50 1997"
  "BEFOREARC compositionality exp_setting_1 experiment_4 grid_size_30 1997"
  "BEFOREARC compositionality exp_setting_1 experiment_4 grid_size_40 1997"
  "BEFOREARC compositionality exp_setting_1 experiment_4 grid_size_50 1997"
  "BEFOREARC compositionality exp_setting_1 experiment_5 grid_size_30 1997"
  "BEFOREARC compositionality exp_setting_1 experiment_5 grid_size_40 1997"
  "BEFOREARC compositionality exp_setting_1 experiment_5 grid_size_50 1997"

  "BEFOREARC compositionality exp_setting_3 experiment_1 grid_size_30 1997"
  "BEFOREARC compositionality exp_setting_3 experiment_1 grid_size_40 1997"
  "BEFOREARC compositionality exp_setting_3 experiment_1 grid_size_50 1997"
  "BEFOREARC compositionality exp_setting_3 experiment_2 grid_size_30 1997"
  "BEFOREARC compositionality exp_setting_3 experiment_2 grid_size_40 1997"
  "BEFOREARC compositionality exp_setting_3 experiment_2 grid_size_50 1997"
  "BEFOREARC compositionality exp_setting_3 experiment_3 grid_size_30 1997"
  "BEFOREARC compositionality exp_setting_3 experiment_3 grid_size_40 1997"
  "BEFOREARC compositionality exp_setting_3 experiment_3 grid_size_50 1997"
  "BEFOREARC compositionality exp_setting_3 experiment_4 grid_size_30 1997"
  "BEFOREARC compositionality exp_setting_3 experiment_4 grid_size_40 1997"
  "BEFOREARC compositionality exp_setting_3 experiment_4 grid_size_50 1997"
  "BEFOREARC compositionality exp_setting_3 experiment_5 grid_size_30 1997"
  "BEFOREARC compositionality exp_setting_3 experiment_5 grid_size_40 1997"
  "BEFOREARC compositionality exp_setting_3 experiment_5 grid_size_50 1997"

  "BEFOREARC sys-gen exp_setting_5 experiment_1 grid_size_30 1997"
  "BEFOREARC sys-gen exp_setting_5 experiment_1 grid_size_40 1997"
  "BEFOREARC sys-gen exp_setting_5 experiment_1 grid_size_50 1997"
  "BEFOREARC sys-gen exp_setting_5 experiment_2 grid_size_30 1997"
  "BEFOREARC sys-gen exp_setting_5 experiment_2 grid_size_40 1997"
  "BEFOREARC sys-gen exp_setting_5 experiment_2 grid_size_50 1997"
  "BEFOREARC sys-gen exp_setting_5 experiment_3 grid_size_30 1997"
  "BEFOREARC sys-gen exp_setting_5 experiment_3 grid_size_40 1997"
  "BEFOREARC sys-gen exp_setting_5 experiment_3 grid_size_50 1997"
  "BEFOREARC sys-gen exp_setting_5 experiment_4 grid_size_30 1997"
  "BEFOREARC sys-gen exp_setting_5 experiment_4 grid_size_40 1997"
  "BEFOREARC sys-gen exp_setting_5 experiment_4 grid_size_50 1997"
  "BEFOREARC sys-gen exp_setting_5 experiment_5 grid_size_30 1997"
  "BEFOREARC sys-gen exp_setting_5 experiment_5 grid_size_40 1997"
  "BEFOREARC sys-gen exp_setting_5 experiment_5 grid_size_50 1997"
)

# Iterate over each configuration and submit the job
for config in "${configs[@]}"; do
  read -r data_env study setting name specifics seed <<< "$config"

  sbatch run_experiment_bigger.submit \
    wandb.sweep.enabled=false \
    wandb.wandb_entity_name=VisReas-ETHZ \
    wandb.wandb_project_name=VisReas-project-${data_env}-llada-grid-size \
    base.data_env=${data_env} \
    experiment.study=${study} \
    experiment.setting=${setting} \
    experiment.name=${name} \
    experiment.exp_specifics=${specifics} \
    base.seed=${seed}
done

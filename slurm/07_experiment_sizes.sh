#!/bin/bash

# List of experiment configurations
# Format: "data_env study setting name specifics max_samples max_epochs seed"
configs=(
  # --- Config 1: 1M Samples, 10 Epochs ---
  # Seed 1997
  "BEFOREARC compositionality exp_setting_1 experiment_0_1M scaling_1M 1000000 10 1997"
  "BEFOREARC compositionality exp_setting_2 experiment_0_1M scaling_1M 1000000 10 1997"
  "BEFOREARC compositionality exp_setting_3 experiment_0_1M scaling_1M 1000000 10 1997"
  # Seed 42
  "BEFOREARC compositionality exp_setting_1 experiment_0_1M scaling_1M 1000000 10 42"
  "BEFOREARC compositionality exp_setting_2 experiment_0_1M scaling_1M 1000000 10 42"
  "BEFOREARC compositionality exp_setting_3 experiment_0_1M scaling_1M 1000000 10 42"
  # Seed 2025
  "BEFOREARC compositionality exp_setting_1 experiment_0_1M scaling_1M 1000000 10 2025"
  "BEFOREARC compositionality exp_setting_2 experiment_0_1M scaling_1M 1000000 10 2025"
  "BEFOREARC compositionality exp_setting_3 experiment_0_1M scaling_1M 1000000 10 2025"

  # --- Config 2: 100k Samples, 100 Epochs ---
  # Seed 1997
  "BEFOREARC compositionality exp_setting_1 experiment_0_100k scaling_100k 100000 100 1997"
  "BEFOREARC compositionality exp_setting_2 experiment_0_100k scaling_100k 100000 100 1997"
  "BEFOREARC compositionality exp_setting_3 experiment_0_100k scaling_100k 100000 100 1997"
  # Seed 42
  "BEFOREARC compositionality exp_setting_1 experiment_0_100k scaling_100k 100000 100 42"
  "BEFOREARC compositionality exp_setting_2 experiment_0_100k scaling_100k 100000 100 42"
  "BEFOREARC compositionality exp_setting_3 experiment_0_100k scaling_100k 100000 100 42"
  # Seed 2025
  "BEFOREARC compositionality exp_setting_1 experiment_0_100k scaling_100k 100000 100 2025"
  "BEFOREARC compositionality exp_setting_2 experiment_0_100k scaling_100k 100000 100 2025"
  "BEFOREARC compositionality exp_setting_3 experiment_0_100k scaling_100k 100000 100 2025"

  # --- Config 3: 250k Samples, 50 Epochs ---
  # Seed 1997
  "BEFOREARC compositionality exp_setting_1 experiment_0_250k scaling_250k 250000 50 1997"
  "BEFOREARC compositionality exp_setting_2 experiment_0_250k scaling_250k 250000 50 1997"
  "BEFOREARC compositionality exp_setting_3 experiment_0_250k scaling_250k 250000 50 1997"
  # Seed 42
  "BEFOREARC compositionality exp_setting_1 experiment_0_250k scaling_250k 250000 50 42"
  "BEFOREARC compositionality exp_setting_2 experiment_0_250k scaling_250k 250000 50 42"
  "BEFOREARC compositionality exp_setting_3 experiment_0_250k scaling_250k 250000 50 42"
  # Seed 2025
  "BEFOREARC compositionality exp_setting_1 experiment_0_250k scaling_250k 250000 50 2025"
  "BEFOREARC compositionality exp_setting_2 experiment_0_250k scaling_250k 250000 50 2025"
  "BEFOREARC compositionality exp_setting_3 experiment_0_250k scaling_250k 250000 50 2025"

  # --- Config 4: 500k Samples, 20 Epochs ---
  # Seed 1997
  "BEFOREARC compositionality exp_setting_1 experiment_0_500k scaling_500k 500000 20 1997"
  "BEFOREARC compositionality exp_setting_2 experiment_0_500k scaling_500k 500000 20 1997"
  "BEFOREARC compositionality exp_setting_3 experiment_0_500k scaling_500k 500000 20 1997"
  # Seed 42
  "BEFOREARC compositionality exp_setting_1 experiment_0_500k scaling_500k 500000 20 42"
  "BEFOREARC compositionality exp_setting_2 experiment_0_500k scaling_500k 500000 20 42"
  "BEFOREARC compositionality exp_setting_3 experiment_0_500k scaling_500k 500000 20 42"
  # Seed 2025
  "BEFOREARC compositionality exp_setting_1 experiment_0_500k scaling_500k 500000 20 2025"
  "BEFOREARC compositionality exp_setting_2 experiment_0_500k scaling_500k 500000 20 2025"
  "BEFOREARC compositionality exp_setting_3 experiment_0_500k scaling_500k 500000 20 2025"
)

# Iterate over each configuration and submit the job
for config in "${configs[@]}"; do
  # UPDATED: Added max_samples and max_epochs to the read command
  read -r data_env study setting name specifics max_samples max_epochs seed <<< "$config"

  sbatch run_experiment.submit \
    wandb.sweep.enabled=false \
    wandb.wandb_entity_name=VisReas-ETHZ \
    wandb.wandb_project_name=VisReas-project-${data_env}-llada-exp-scaling \
    base.data_env=${data_env} \
    experiment.study=${study} \
    experiment.setting=${setting} \
    experiment.name=${name} \
    experiment.exp_specifics=${specifics} \
    data.max_train_samples=${max_samples} \
    training.max_epochs=${max_epochs} \
    base.seed=${seed}
done
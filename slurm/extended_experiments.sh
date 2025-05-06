#!/bin/bash

# Choose which environments to run: options are "BEFOREARC", "REARC", or "BOTH"
run_env="BOTH"

# Define common parameters
wandb_entity="VisReas-ETHZ"

# Submit jobs
if [[ "$run_env" == "BEFOREARC" || "$run_env" == "BOTH" ]]; then
  data_env="BEFOREARC"
  data_env_lc=$(echo "$data_env" | tr '[:upper:]' '[:lower:]')
  wandb_project="VisReas-project-${data_env}-llada"
  sweep_types=("comp" "sysgen")
  experiment_settings=("es1" "es2" "es3" "es4" "es5")

  for sweep_type in "${sweep_types[@]}"; do
    for setting in "${experiment_settings[@]}"; do
      # Only enable task embedding for compositionality in BEFOREARC

      config="base.data_env=${data_env} \
wandb.sweep.config=configs/sweeps/${data_env_lc}/${sweep_type}_${setting}.yaml \
wandb.wandb_project_name=${wandb_project} \
wandb.wandb_entity_name=${wandb_entity}"
      echo "Submitting BEFOREARC: $config"
      sbatch run_experiment.submit \
         wandb.sweep.enabled=true \
        $config
      sleep 61
    done
  done
fi

if [[ "$run_env" == "REARC" || "$run_env" == "BOTH" ]]; then
  data_env="REARC"
  data_env_lc=$(echo "$data_env" | tr '[:upper:]' '[:lower:]')
  wandb_project="VisReas-project-${data_env}-llada"
  sweep_types=("se" "sysgen")
  experiment_settings=("es1")

  for sweep_type in "${sweep_types[@]}"; do
    for setting in "${experiment_settings[@]}"; do
      config="base.data_env=${data_env} \
wandb.sweep.config=configs/sweeps/${data_env_lc}/${sweep_type}_${setting}.yaml \
wandb.wandb_project_name=${wandb_project} \
wandb.wandb_entity_name=${wandb_entity}"
      echo "Submitting REARC ($sweep_type): $config"
      sbatch run_experiment.submit \
        wandb.sweep.enabled=true \
        data.train_batch_size=32 \
        data.val_batch_size=32 \
        data.test_batch_size=32 \
        $config
      sleep 61
    done
  done
fi

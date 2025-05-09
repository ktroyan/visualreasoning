#!/bin/bash

wandb_entity="VisReas-ETHZ"
data_env="BEFOREARC"
data_env_lc=$(echo "$data_env" | tr '[:upper:]' '[:lower:]')
wandb_project="VisReas-project-${data_env}-llada-final"

sweep_type="comp"
experiment_settings=("es3" "es4" "es5")
for setting in "${experiment_settings[@]}"; do
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

sweep_type="sysgen"
experiment_settings=("es1" "es2" "es3" "es4" "es5")
for setting in "${experiment_settings[@]}"; do
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

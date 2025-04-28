#!/bin/bash

declare -a configs=(
  "base.data_env=BEFOREARC model.task_embedding.enabled=false"
  "base.data_env=BEFOREARC model.task_embedding.enabled=true"
  #"base.data_env=REARC backbone_network.diffusion.steps=32"
  #"base.data_env=REARC backbone_network.diffusion.steps=32 backbone_network.diffusion.sage_thinking=true"
)

for config in "${configs[@]}"; do
  sbatch run_experiment.submit wandb.sweep.enabled=false $config
  sleep 2 # avoid folders/runs to have same name (as it is based on timestamp)
done

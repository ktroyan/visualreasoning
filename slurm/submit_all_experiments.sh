#!/bin/bash

declare -a configs=(
  "base.data_env=REARC"
  #"base.data_env=REARC backbone_network.diffusion.steps=32"
  #"base.data_env=REARC backbone_network.diffusion.steps=32 backbone_network.diffusion.sage_thinking=true"
)

for config in "${configs[@]}"; do
  sbatch run_experiment.submit wandb.sweep.enabled=false experiment.name=experiment_4 $config
done

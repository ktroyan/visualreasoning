#!/bin/bash

declare -a configs=(
  "base.data_env=REARC backbone_network.diffusion.steps=1"
  "base.data_env=REARC backbone_network.diffusion.steps=8"
  "base.data_env=REARC backbone_network.diffusion.steps=16"
  "base.data_env=REARC backbone_network.diffusion.steps=32"
  "base.data_env=REARC backbone_network.diffusion.steps=64"
  "base.data_env=REARC backbone_network.diffusion.steps=128"
)

for config in "${configs[@]}"; do
  sbatch run_inference.submit wandb.sweep.enabled=true wandb.wandb_project_name=VisReas-project-inference $config
done

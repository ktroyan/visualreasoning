#!/bin/bash

declare -a configs=(
  ""
  "backbone_network.diffusion.steps=32"
  "backbone_network.diffusion.steps=64"
  "backbone_network.diffusion.steps=128"
  "backbone_network.diffusion.sage_thinking=true"
  "backbone_network.diffusion.steps=32 backbone_network.diffusion.sage_thinking=true"
  "backbone_network.diffusion.steps=64 backbone_network.diffusion.sage_thinking=true"
  "backbone_network.diffusion.steps=128 backbone_network.diffusion.sage_thinking=true"
)

for config in "${configs[@]}"; do
  sbatch run_experiment.submit training.max_epochs=20 wandb.sweep.enabled=false experiment.name=experiment_4 $config
done

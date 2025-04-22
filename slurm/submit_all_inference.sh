#!/bin/bash

declare -a configs=(
  "base.data_env=REARC inference.inference_model_ckpt=/cluster/home/sage/visualreasoning/REARC/experiments/experiment_19_04_10_09/epoch=19-step=31260.ckpt backbone_network.diffusion.steps=1"
  "base.data_env=REARC inference.inference_model_ckpt=/cluster/home/sage/visualreasoning/REARC/experiments/experiment_19_04_10_09/epoch=19-step=31260.ckpt backbone_network.diffusion.steps=8"
  "base.data_env=REARC inference.inference_model_ckpt=/cluster/home/sage/visualreasoning/REARC/experiments/experiment_19_04_10_09/epoch=19-step=31260.ckpt backbone_network.diffusion.steps=16"
  "base.data_env=REARC inference.inference_model_ckpt=/cluster/home/sage/visualreasoning/REARC/experiments/experiment_19_04_10_09/epoch=19-step=31260.ckpt backbone_network.diffusion.steps=32"
  "base.data_env=REARC inference.inference_model_ckpt=/cluster/home/sage/visualreasoning/REARC/experiments/experiment_19_04_10_09/epoch=19-step=31260.ckpt backbone_network.diffusion.steps=64"
  "base.data_env=REARC inference.inference_model_ckpt=/cluster/home/sage/visualreasoning/REARC/experiments/experiment_19_04_10_09/epoch=19-step=31260.ckpt backbone_network.diffusion.steps=128"
)

for config in "${configs[@]}"; do
  sbatch run_inference.submit wandb.sweep.enabled=true wandb.wandb_project_name=VisReas-project-inference $config
done

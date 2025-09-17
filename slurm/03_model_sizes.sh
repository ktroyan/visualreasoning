#!/bin/bash

# List of experiment configurations
configs=(
#  "BEFOREARC compositionality exp_setting_1 experiment_1 48 96 2 4 4 4 1997"
#  "BEFOREARC compositionality exp_setting_1 experiment_2 48 96 2 4 4 4 1997"
#  "BEFOREARC compositionality exp_setting_1 experiment_3 48 96 2 4 4 4 1997"
#  "BEFOREARC compositionality exp_setting_1 experiment_4 48 96 2 4 4 4 1997"
#  "BEFOREARC compositionality exp_setting_1 experiment_5 48 96 2 4 4 4 1997"
#  "BEFOREARC compositionality exp_setting_3 experiment_1 48 96 2 4 4 4 1997"
#  "BEFOREARC compositionality exp_setting_3 experiment_2 48 96 2 4 4 4 1997"
#  "BEFOREARC compositionality exp_setting_3 experiment_3 48 96 2 4 4 4 1997"
#  "BEFOREARC compositionality exp_setting_3 experiment_4 48 96 2 4 4 4 1997"
#  "BEFOREARC compositionality exp_setting_3 experiment_5 48 96 2 4 4 4 1997"
#  "BEFOREARC sys-gen exp_setting_5 experiment_1 48 96 2 4 4 4 1997"
#  "BEFOREARC sys-gen exp_setting_5 experiment_2 48 96 2 4 4 4 1997"
#  "BEFOREARC sys-gen exp_setting_5 experiment_3 48 96 2 4 4 4 1997"
#  "BEFOREARC sys-gen exp_setting_5 experiment_4 48 96 2 4 4 4 1997"
#  "BEFOREARC sys-gen exp_setting_5 experiment_5 48 96 2 4 4 4 1997"
#
#  "BEFOREARC compositionality exp_setting_1 experiment_1 128 256 4 4 4 6 1997"
#  "BEFOREARC compositionality exp_setting_1 experiment_2 128 256 4 4 4 6 1997"
#  "BEFOREARC compositionality exp_setting_1 experiment_3 128 256 4 4 4 6 1997"
#  "BEFOREARC compositionality exp_setting_1 experiment_4 128 256 4 4 4 6 1997"
#  "BEFOREARC compositionality exp_setting_1 experiment_5 128 256 4 4 4 6 1997"
#  "BEFOREARC compositionality exp_setting_3 experiment_1 128 256 4 4 4 6 1997"
#  "BEFOREARC compositionality exp_setting_3 experiment_2 128 256 4 4 4 6 1997"
#  "BEFOREARC compositionality exp_setting_3 experiment_3 128 256 4 4 4 6 1997"
#  "BEFOREARC compositionality exp_setting_3 experiment_4 128 256 4 4 4 6 1997"
#  "BEFOREARC compositionality exp_setting_3 experiment_5 128 256 4 4 4 6 1997"
#  "BEFOREARC sys-gen exp_setting_5 experiment_1 128 256 4 4 4 6 1997"
#  "BEFOREARC sys-gen exp_setting_5 experiment_2 128 256 4 4 4 6 1997"
#  "BEFOREARC sys-gen exp_setting_5 experiment_3 128 256 4 4 4 6 1997"
#  "BEFOREARC sys-gen exp_setting_5 experiment_4 128 256 4 4 4 6 1997"
#  "BEFOREARC sys-gen exp_setting_5 experiment_5 128 256 4 4 4 6 1997"
#
#  "BEFOREARC compositionality exp_setting_1 experiment_1 384 512 4 8 8 8 1997"
#  "BEFOREARC compositionality exp_setting_1 experiment_2 384 512 4 8 8 8 1997"
#  "BEFOREARC compositionality exp_setting_1 experiment_3 384 512 4 8 8 8 1997"
#  "BEFOREARC compositionality exp_setting_1 experiment_4 384 512 4 8 8 8 1997"
#  "BEFOREARC compositionality exp_setting_1 experiment_5 384 512 4 8 8 8 1997"
#  "BEFOREARC compositionality exp_setting_3 experiment_1 384 512 4 8 8 8 1997"
#  "BEFOREARC compositionality exp_setting_3 experiment_2 384 512 4 8 8 8 1997"
#  "BEFOREARC compositionality exp_setting_3 experiment_3 384 512 4 8 8 8 1997"
#  "BEFOREARC compositionality exp_setting_3 experiment_4 384 512 4 8 8 8 1997"
#  "BEFOREARC compositionality exp_setting_3 experiment_5 384 512 4 8 8 8 1997"
#  "BEFOREARC sys-gen exp_setting_5 experiment_1 384 512 4 8 8 8 1997"
#  "BEFOREARC sys-gen exp_setting_5 experiment_2 384 512 4 8 8 8 1997"
#  "BEFOREARC sys-gen exp_setting_5 experiment_3 384 512 4 8 8 8 1997"
#  "BEFOREARC sys-gen exp_setting_5 experiment_4 384 512 4 8 8 8 1997"
#  "BEFOREARC sys-gen exp_setting_5 experiment_5 384 512 4 8 8 8 1997"



  ## Seed 42
  "BEFOREARC compositionality exp_setting_1 experiment_1 48 96 2 4 4 4 42"
  "BEFOREARC compositionality exp_setting_1 experiment_2 48 96 2 4 4 4 42"
  "BEFOREARC compositionality exp_setting_1 experiment_3 48 96 2 4 4 4 42"
  "BEFOREARC compositionality exp_setting_1 experiment_4 48 96 2 4 4 4 42"
  "BEFOREARC compositionality exp_setting_1 experiment_5 48 96 2 4 4 4 42"
  "BEFOREARC compositionality exp_setting_3 experiment_1 48 96 2 4 4 4 42"
  "BEFOREARC compositionality exp_setting_3 experiment_2 48 96 2 4 4 4 42"
  "BEFOREARC compositionality exp_setting_3 experiment_3 48 96 2 4 4 4 42"
  "BEFOREARC compositionality exp_setting_3 experiment_4 48 96 2 4 4 4 42"
  "BEFOREARC compositionality exp_setting_3 experiment_5 48 96 2 4 4 4 42"
  "BEFOREARC sys-gen exp_setting_5 experiment_1 48 96 2 4 4 4 42"
  "BEFOREARC sys-gen exp_setting_5 experiment_2 48 96 2 4 4 4 42"
  "BEFOREARC sys-gen exp_setting_5 experiment_3 48 96 2 4 4 4 42"
  "BEFOREARC sys-gen exp_setting_5 experiment_4 48 96 2 4 4 4 42"
  "BEFOREARC sys-gen exp_setting_5 experiment_5 48 96 2 4 4 4 42"

  "BEFOREARC compositionality exp_setting_1 experiment_1 128 256 4 4 4 6 42"
  "BEFOREARC compositionality exp_setting_1 experiment_2 128 256 4 4 4 6 42"
  "BEFOREARC compositionality exp_setting_1 experiment_3 128 256 4 4 4 6 42"
  "BEFOREARC compositionality exp_setting_1 experiment_4 128 256 4 4 4 6 42"
  "BEFOREARC compositionality exp_setting_1 experiment_5 128 256 4 4 4 6 42"
  "BEFOREARC compositionality exp_setting_3 experiment_1 128 256 4 4 4 6 42"
  "BEFOREARC compositionality exp_setting_3 experiment_2 128 256 4 4 4 6 42"
  "BEFOREARC compositionality exp_setting_3 experiment_3 128 256 4 4 4 6 42"
  "BEFOREARC compositionality exp_setting_3 experiment_4 128 256 4 4 4 6 42"
  "BEFOREARC compositionality exp_setting_3 experiment_5 128 256 4 4 4 6 42"
  "BEFOREARC sys-gen exp_setting_5 experiment_1 128 256 4 4 4 6 42"
  "BEFOREARC sys-gen exp_setting_5 experiment_2 128 256 4 4 4 6 42"
  "BEFOREARC sys-gen exp_setting_5 experiment_3 128 256 4 4 4 6 42"
  "BEFOREARC sys-gen exp_setting_5 experiment_4 128 256 4 4 4 6 42"
  "BEFOREARC sys-gen exp_setting_5 experiment_5 128 256 4 4 4 6 42"

  "BEFOREARC compositionality exp_setting_1 experiment_1 384 512 4 8 8 8 42"
  "BEFOREARC compositionality exp_setting_1 experiment_2 384 512 4 8 8 8 42"
  "BEFOREARC compositionality exp_setting_1 experiment_3 384 512 4 8 8 8 42"
  "BEFOREARC compositionality exp_setting_1 experiment_4 384 512 4 8 8 8 42"
  "BEFOREARC compositionality exp_setting_1 experiment_5 384 512 4 8 8 8 42"
  "BEFOREARC compositionality exp_setting_3 experiment_1 384 512 4 8 8 8 42"
  "BEFOREARC compositionality exp_setting_3 experiment_2 384 512 4 8 8 8 42"
  "BEFOREARC compositionality exp_setting_3 experiment_3 384 512 4 8 8 8 42"
  "BEFOREARC compositionality exp_setting_3 experiment_4 384 512 4 8 8 8 42"
  "BEFOREARC compositionality exp_setting_3 experiment_5 384 512 4 8 8 8 42"
  "BEFOREARC sys-gen exp_setting_5 experiment_1 384 512 4 8 8 8 42"
  "BEFOREARC sys-gen exp_setting_5 experiment_2 384 512 4 8 8 8 42"
  "BEFOREARC sys-gen exp_setting_5 experiment_3 384 512 4 8 8 8 42"
  "BEFOREARC sys-gen exp_setting_5 experiment_4 384 512 4 8 8 8 42"
  "BEFOREARC sys-gen exp_setting_5 experiment_5 384 512 4 8 8 8 42"


  # Seed 2025
  "BEFOREARC compositionality exp_setting_1 experiment_1 48 96 2 4 4 4 2025"
  "BEFOREARC compositionality exp_setting_1 experiment_2 48 96 2 4 4 4 2025"
  "BEFOREARC compositionality exp_setting_1 experiment_3 48 96 2 4 4 4 2025"
  "BEFOREARC compositionality exp_setting_1 experiment_4 48 96 2 4 4 4 2025"
  "BEFOREARC compositionality exp_setting_1 experiment_5 48 96 2 4 4 4 2025"
  "BEFOREARC compositionality exp_setting_3 experiment_1 48 96 2 4 4 4 2025"
  "BEFOREARC compositionality exp_setting_3 experiment_2 48 96 2 4 4 4 2025"
  "BEFOREARC compositionality exp_setting_3 experiment_3 48 96 2 4 4 4 2025"
  "BEFOREARC compositionality exp_setting_3 experiment_4 48 96 2 4 4 4 2025"
  "BEFOREARC compositionality exp_setting_3 experiment_5 48 96 2 4 4 4 2025"
  "BEFOREARC sys-gen exp_setting_5 experiment_1 48 96 2 4 4 4 2025"
  "BEFOREARC sys-gen exp_setting_5 experiment_2 48 96 2 4 4 4 2025"
  "BEFOREARC sys-gen exp_setting_5 experiment_3 48 96 2 4 4 4 2025"
  "BEFOREARC sys-gen exp_setting_5 experiment_4 48 96 2 4 4 4 2025"
  "BEFOREARC sys-gen exp_setting_5 experiment_5 48 96 2 4 4 4 2025"

  "BEFOREARC compositionality exp_setting_1 experiment_1 128 256 4 4 4 6 2025"
  "BEFOREARC compositionality exp_setting_1 experiment_2 128 256 4 4 4 6 2025"
  "BEFOREARC compositionality exp_setting_1 experiment_3 128 256 4 4 4 6 2025"
  "BEFOREARC compositionality exp_setting_1 experiment_4 128 256 4 4 4 6 2025"
  "BEFOREARC compositionality exp_setting_1 experiment_5 128 256 4 4 4 6 2025"
  "BEFOREARC compositionality exp_setting_3 experiment_1 128 256 4 4 4 6 2025"
  "BEFOREARC compositionality exp_setting_3 experiment_2 128 256 4 4 4 6 2025"
  "BEFOREARC compositionality exp_setting_3 experiment_3 128 256 4 4 4 6 2025"
  "BEFOREARC compositionality exp_setting_3 experiment_4 128 256 4 4 4 6 2025"
  "BEFOREARC compositionality exp_setting_3 experiment_5 128 256 4 4 4 6 2025"
  "BEFOREARC sys-gen exp_setting_5 experiment_1 128 256 4 4 4 6 2025"
  "BEFOREARC sys-gen exp_setting_5 experiment_2 128 256 4 4 4 6 2025"
  "BEFOREARC sys-gen exp_setting_5 experiment_3 128 256 4 4 4 6 2025"
  "BEFOREARC sys-gen exp_setting_5 experiment_4 128 256 4 4 4 6 2025"
  "BEFOREARC sys-gen exp_setting_5 experiment_5 128 256 4 4 4 6 2025"

  "BEFOREARC compositionality exp_setting_1 experiment_1 384 512 4 8 8 8 2025"
  "BEFOREARC compositionality exp_setting_1 experiment_2 384 512 4 8 8 8 2025"
  "BEFOREARC compositionality exp_setting_1 experiment_3 384 512 4 8 8 8 2025"
  "BEFOREARC compositionality exp_setting_1 experiment_4 384 512 4 8 8 8 2025"
  "BEFOREARC compositionality exp_setting_1 experiment_5 384 512 4 8 8 8 2025"
  "BEFOREARC compositionality exp_setting_3 experiment_1 384 512 4 8 8 8 2025"
  "BEFOREARC compositionality exp_setting_3 experiment_2 384 512 4 8 8 8 2025"
  "BEFOREARC compositionality exp_setting_3 experiment_3 384 512 4 8 8 8 2025"
  "BEFOREARC compositionality exp_setting_3 experiment_4 384 512 4 8 8 8 2025"
  "BEFOREARC compositionality exp_setting_3 experiment_5 384 512 4 8 8 8 2025"
  "BEFOREARC sys-gen exp_setting_5 experiment_1 384 512 4 8 8 8 2025"
  "BEFOREARC sys-gen exp_setting_5 experiment_2 384 512 4 8 8 8 2025"
  "BEFOREARC sys-gen exp_setting_5 experiment_3 384 512 4 8 8 8 2025"
  "BEFOREARC sys-gen exp_setting_5 experiment_4 384 512 4 8 8 8 2025"
  "BEFOREARC sys-gen exp_setting_5 experiment_5 384 512 4 8 8 8 2025"

)

# Iterate over each configuration and submit the job
for config in "${configs[@]}"; do
  read -r data_env study setting name embed hidden ratio nheads nkvheads layers seed <<< "$config"

  sbatch run_experiment.submit \
    wandb.sweep.enabled=false \
    wandb.wandb_entity_name=VisReas-ETHZ \
    wandb.wandb_project_name=VisReas-project-${data_env}-llada-model-size \
    base.data_env=${data_env} \
    experiment.study=${study} \
    experiment.setting=${setting} \
    experiment.name=${name} \
    backbone_network.embed_dim=${embed} \
    backbone_network.mlp_hidden_size=${hidden} \
    backbone_network.mlp_ratio=${ratio} \
    backbone_network.n_heads=${nheads} \
    backbone_network.n_kv_heads=${nkvheads} \
    backbone_network.n_layers=${layers} \
    base.seed=${seed}
done

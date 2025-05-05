#!/bin/bash

# Choose which environments to run: options are "BEFOREARC", "REARC", or "BOTH"
run_env="BOTH"

# Define common parameters
task_embedding_options=("false" "true")
wandb_entity="sagerpascal"

# Submit jobs
if [[ "$run_env" == "BEFOREARC" || "$run_env" == "BOTH" ]]; then
  data_env="BEFOREARC"
  data_env_lc=$(echo "$data_env" | tr '[:upper:]' '[:lower:]')
  wandb_project="VisReas-project-${data_env}-sweep-thinking"
  sweep_types=("comp" "sysgen")
  experiment_settings=("es1" "es2" "es3" "es4" "es5")

  for task_embedding in "${task_embedding_options[@]}"; do
    for sweep_type in "${sweep_types[@]}"; do
      for setting in "${experiment_settings[@]}"; do
        config="base.data_env=${data_env} \
model.task_embedding.enabled=${task_embedding} \
wandb.sweep.config=configs/sweeps/${data_env_lc}/${sweep_type}_${setting}.yaml \
wandb.wandb_project_name=${wandb_project} \
wandb.wandb_entity_name=${wandb_entity}"
        echo "Submitting BEFOREARC: $config"
        sbatch run_experiment.submit \
          backbone_network.diffusion.sage_thinking=true \
          data.use_gen_test_set=true \
          data.validate_in_and_out_domain=true \
          wandb.sweep.enabled=true \
          $config
        sleep 61
      done
    done
  done
fi

if [[ "$run_env" == "REARC" || "$run_env" == "BOTH" ]]; then
  data_env="REARC"
  data_env_lc=$(echo "$data_env" | tr '[:upper:]' '[:lower:]')
  wandb_project="VisReas-project-${data_env}-sweep-thinking"
  sweep_types=("se" "sysgen")
  experiment_settings=("es1")

  for task_embedding in "${task_embedding_options[@]}"; do
    for sweep_type in "${sweep_types[@]}"; do
      for setting in "${experiment_settings[@]}"; do
        use_gen_test_set="true"
        validate_in_and_out_domain="true"
        if [[ "$sweep_type" == "se" ]]; then
          use_gen_test_set="false"
          validate_in_and_out_domain="false"
        fi

        config="base.data_env=${data_env} \
model.task_embedding.enabled=${task_embedding} \
wandb.sweep.config=configs/sweeps/${data_env_lc}/${sweep_type}_${setting}.yaml \
wandb.wandb_project_name=${wandb_project} \
wandb.wandb_entity_name=${wandb_entity}"
        echo "Submitting REARC ($sweep_type): $config"
        sbatch run_experiment.submit \
          backbone_network.diffusion.sage_thinking=true \
          data.use_gen_test_set=$use_gen_test_set \
          data.validate_in_and_out_domain=$validate_in_and_out_domain \
          wandb.sweep.enabled=true \
          data.train_batch_size=32 \
          data.val_batch_size=32 \
          data.test_batch_size=32 \
          $config
        sleep 61
      done
    done
  done
fi

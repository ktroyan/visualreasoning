import math
import os
import time
from omegaconf import OmegaConf
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb
import matplotlib
import yaml
from typing import Any, Dict, List, Tuple

# Personal codebase dependencies
import data
import models
import training
import inference
from utility.utils import delete_folder_content, log_config_dict, get_complete_config, generate_timestamped_experiment_name, save_model_metadata_for_ckpt, copy_folder
from utility.logging import logger

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('medium')    # 'high'


def main() -> None:
    """ Main function to run an experiment """

    logger.info("*** Experiment started ***")
    exp_start_time = time.time()

    # Get all the config arguments for a regular experiment run
    config, config_dict = get_complete_config()

    # Setup experiment folders
    experiment_name_timestamped = generate_timestamped_experiment_name("experiment")
    experiment_folder = config.experiment.experiments_dir + f"/{experiment_name_timestamped}"
    os.makedirs(experiment_folder, exist_ok=True)

    # Initialize WandB project run tracking
    run = wandb.init(
        project=config.wandb.wandb_project_name,    # ignored if using sweeps
        entity=config.wandb.wandb_entity_name,      # ignored if using sweeps
        dir=experiment_folder,
        name=experiment_name_timestamped,
        )
    
    if config.wandb.sweep.enabled:
        # Merge the current sweep config arguments with the complete default config arguments
        sweep_config = OmegaConf.create(dict(wandb.config))  # get the sweep config for the current run
        config, config_dict = get_complete_config(sweep_config) # use the sweep config to overwrite parameters (but before CLI arguments)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.base.gpu_id)

    wandb.config = config_dict  # set to wandb the config to use through the program; if sweep enabled, update the wandb.config with the merged config

    # Log the complete and actual config used for the experiment
    log_config_dict(config, "*** All arguments contained in the config dict ***")

    # Seed everything for reproducibility
    if config.base.seed is not None:
        pl.seed_everything(config.base.seed)

    # Data chosen
    data_module = vars(data)[config.base.data_module]
    datamodule = data_module(config.data, config.model)   # initialize the data module
    logger.info(f"Data module instantiated. Now showing the total number of samples per dataloader:\n{datamodule}\n")

    # Get the image size from the datamodule. Useful for the model backbone
    image_size = datamodule.image_size
    logger.info(f"Image size considered (with padding): {image_size}")

    # Model chosen
    model_module = vars(models)[config.base.model_module]
    model = model_module(config.base, config.model, config.data, config.backbone_network, config.head_network, image_size, experiment_folder)   # initialize the model module
    logger.trace(f"Model chosen for training: {model}")
    
    # Save the model metadata for future checkpoint use
    save_model_metadata_for_ckpt(experiment_folder, model)

    # Initialize the experiment logger
    if config.experiment.exp_logger == 'wandb':
        exp_logger = WandbLogger(project=config.wandb.wandb_project_name, name=experiment_name_timestamped, save_dir=experiment_folder, log_model=config.wandb.log_model)
    else:
        logger.warning(f"Experiment logger {config.experiment.exp_logger} not recognized. The experiment logger is set to Null. Otherwise, choose 'wandb'.")
        exp_logger = None


    # Training
    trainer, best_model, best_model_ckpt, train_results = training.main(config,
                                                                        experiment_folder,
                                                                        datamodule,
                                                                        model,
                                                                        exp_logger
                                                                        )


    # Testing
    all_test_results = inference.main(config,
                                      experiment_folder,
                                      datamodule,
                                      model=best_model,
                                      model_ckpt_path=None, # we use the best model obtained during training, so no need to specify a checkpoint path
                                      exp_logger=exp_logger
                                      )
    

    # End the wandb run
    run.finish()

    # Time taken for the experiment
    log_message = "*** Experiment ended ***\n"
    exp_elapsed_time = time.time() - exp_start_time
    log_message += f"\nTotal experiment time: \n{exp_elapsed_time} seconds ~=\n{exp_elapsed_time/60} minutes ~=\n{exp_elapsed_time/(60*60)} hours"
    logger.info(log_message)

    logger.warning(all_test_results.keys())
    logger.warning(all_test_results['test_results'].keys())
    logger.warning(all_test_results)

    # Save the results and config arguments that we are the most interested to check quickly when experimenting
    exp_results_dict = {
        'experiments_dir': config.experiment.experiments_dir,
        'exp_name': experiment_name_timestamped,
        'dataset_dir': config.data.dataset_dir,
        'exp_duration': exp_elapsed_time,
        'data_module': config.base.data_module,
        'model_module': config.base.model_module,
        'network_backbone': config.model.backbone,
        'network_head': config.model.head,
        'model_ckpt': config.training.model_ckpt_path,
        'max_epochs': config.training.max_epochs,
        'train_batch_size': config.data.train_batch_size,
        'val_batch_size': config.data.val_batch_size,
        'test_batch_size': config.data.test_batch_size,
        'lr': config.model.training_hparams.lr,
        'optimizer': config.model.training_hparams.optimizer,
        'scheduler_type': config.model.training_hparams.scheduler.type,
        'scheduler_interval': config.model.training_hparams.scheduler.interval,
        'scheduler_frequency': config.model.training_hparams.scheduler.frequency,
        'seed': config.base.seed,
    }

    # Update the experiment results dict with the training results
    exp_results_dict.update({k:v for k,v in train_results.items()})
    
    # Update the experiment results dict with the test results
    exp_results_dict.update({k:v for k,v in all_test_results['test_results']['test_results_global_avg'].items()})
    exp_results_dict.update({k:v for k,v in all_test_results['test_results']['test_results_per_task_avg'].items()})

    if config.data.use_gen_test_set:
        exp_results_dict.update({k:v for k,v in all_test_results['gen_test_results']['gen_test_results_global_avg'].items()})
        exp_results_dict.update({k:v for k,v in all_test_results['gen_test_results']['gen_test_results_per_task_avg'].items()})
    
    output_dict_df = pd.DataFrame([exp_results_dict])
    os.makedirs(config.experiment.exp_summary_results_dir, exist_ok=True)
    csv_path = os.path.join(config.experiment.exp_summary_results_dir, 'all_results_summary.csv')
    write_header = not os.path.exists(csv_path)
    output_dict_df.to_csv(csv_path, sep=';', mode='a', index=False, header=write_header)

def count_grid_combinations_for_sweep_jobs(sweep_config):
    """
    Count the number of combinations of the grid search for the sweep jobs.
    This is used to determine how many jobs to run for the sweep if the num_sweeps parameter is set to -1 in the sweep config.
    """
    def extract_values(d):
        """
        Recursively extract all lists of 'values' from nested 'parameters' dicts because of OmegaConf style.
        """
        count_list = []
        if isinstance(d, dict):
            for key, value in d.items():
                if key == 'values' and isinstance(value, list):
                    count_list.append(len(value))
                elif isinstance(value, dict):
                    count_list.extend(extract_values(value))
        return count_list

    # Extract 'parameters' section
    param_section = sweep_config.get('parameters', {})

    # Get all counts/lengths of value lists
    all_counts = extract_values(param_section)

    # Compute total combinations as product of all value list lengths
    total_combinations = math.prod(all_counts)

    return total_combinations

if __name__ == '__main__':
    logger.info(f"experiment.py process ID: {os.getpid()}")

    matplotlib.use('Agg')   # prevent the matplotlib GUI pop-ups from stealing focus

    # Get all the config arguments. This is needed to get the arguments that decide on whether to run sweeps or not and for how many sweeps
    config, _ = get_complete_config()

    if config.wandb.sweep.enabled:

        # Start sweep time
        sweep_start_time = time.time()
        logger.info(f"*** Sweep started ***")

        # Load the sweep config yaml file
        with open(config.wandb.sweep.config, 'r') as f:
            sweep_config = yaml.safe_load(f)

        # Setup WandB sweep
        sweep_id = wandb.sweep(sweep=sweep_config,
                               project=config.wandb.wandb_project_name)

        # Get number of sweep jobs to run for this sweep
        if config.wandb.sweep.num_sweeps == -1:
            # If num_sweeps is -1, run all the possible jobs as per the grid search
            num_sweeps = count_grid_combinations_for_sweep_jobs(sweep_config)
        else:
            # If num_sweeps is specified, run that many jobs
            num_sweeps = config.wandb.sweep.num_sweeps
        
        logger.info(f"Number of sweep jobs to run: {num_sweeps}")

        # Start sweep job
        wandb.agent(sweep_id, 
                    function=main,
                    count=num_sweeps)
        
        # End sweep time
        sweep_elapsed_time = time.time() - sweep_start_time
        logger.info(f"*** Sweep ended ***\nTotal sweep time: \n{sweep_elapsed_time} seconds ~=\n{sweep_elapsed_time/60} minutes ~=\n{sweep_elapsed_time/(60*60)} hours")

    else:
        main()

import math
import os
import time
from omegaconf import OmegaConf
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
from utility.utils import log_config_dict, get_complete_config, generate_timestamped_experiment_name, save_model_metadata_for_ckpt, get_paper_model_name
from utility.logging import logger

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('medium')    # 'high'

def write_experiment_results_logs(config, experiment_folder, paper_experiment_results, paper_model_name):
    # Save the experiment results relevant to the paper
    results_file_name = (
        f"{config.base.data_env.lower()}_"
        f"{config.experiment.study.replace('-', '')}_"
        f"{config.experiment.setting.replace('exp_setting_', 'es')}_"
        f"{config.experiment.name.replace('experiment_', 'exp')}_"
        f"{paper_model_name}_"
        # f"{config.model.head}_"
        "results.log"
    )

    experiment_results_file_path = os.path.join(experiment_folder, results_file_name)

    # Write results with metadata
    with open(experiment_results_file_path, 'w') as f:

        # Log experiment metadata
        f.write("*** Experiment ***\n")
        f.write(f"Data Environment: {config.base.data_env}\n")
        f.write(f"Study: {config.experiment.study}\n")
        f.write(f"Experiment Setting: {config.experiment.setting}\n")
        f.write(f"Experiment Name: {config.experiment.name}\n\n")

        # Log model name
        f.write(f"*** Model ***\n")
        full_model_name = f"{paper_model_name} + {config.model.head}"   # model name including network head
        f.write(f"Model: {full_model_name}\n\n")

        # Log test results
        f.write("*** Results ***")
        for key, value in paper_experiment_results.items():
            f.write(f"\n{key}:\n")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    f.write(f"  {sub_key}: {sub_value}\n")
            else:
                f.write(f"  {value}\n")

        # Log full config at the end of the log file
        f.write("\n\n*** Config ***\n")
        f.write(OmegaConf.to_yaml(config))
        f.write("\n")
    
    logger.info(f"Experiment results saved to: {experiment_results_file_path}")

    return experiment_results_file_path

def main() -> None:
    """ Main function to run an experiment """

    logger.info("*** Experiment started ***")
    exp_start_time = time.time()

    # Get all the config arguments for a regular/single experiment run
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

    # Explicitly save the study, setting, experiment name and paper model name to make sure they exist table columns
    run.summary["study"] = config.experiment.study
    run.summary["setting"] = config.experiment.setting
    run.summary["experiment_name"] = config.experiment.name

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


    ## Training results
    # Get the value for each epoch key (in the dict "/metrics")
    paper_train_results = {k.replace("metrics/", ""): v for k, v in train_results["metrics"].items() if "epoch" in k}

    ## Test results
    # For each key in all_test_results, compute the mean of the array of values and store it
    paper_test_results = {k: v.mean() for k, v in all_test_results['test_results'].items()}
    if 'gen_test_results' in all_test_results:
        paper_test_results.update({k: v.mean() for k, v in all_test_results['gen_test_results'].items()})

    ## All paper experiment results
    paper_experiment_results = {
        "train_results": paper_train_results,
        "test_results": paper_test_results
    }

    # Get the model name for the paper
    paper_model_name = get_paper_model_name(config)
    exp_logger.experiment.log({"paper_model_name": paper_model_name})  # log the model name for the paper to wandb
    run.summary["model_name"] = paper_model_name

    # Save the experiment results relevant to the paper
    experiment_results_file_path = write_experiment_results_logs(config, experiment_folder, paper_experiment_results, paper_model_name)
    exp_logger.experiment.save(experiment_results_file_path)    # save the experiment results file to wandb; similar to wandb.save()

    # End the wandb run
    run.finish()

    # Time taken for the experiment
    log_message = "*** Experiment ended ***\n"
    exp_elapsed_time = time.time() - exp_start_time
    log_message += f"\nTotal experiment time: \n{exp_elapsed_time} seconds ~=\n{exp_elapsed_time/60} minutes ~=\n{exp_elapsed_time/(60*60)} hours"
    logger.info(log_message)



def count_grid_combinations_for_sweep_jobs(sweep_config):
    """
    Count the number of combinations of the grid search for the sweep jobs.
    This is used to determine how many jobs to run for the sweep if the num_jobs parameter is set to -1 in the sweep config.
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

    # Prevent the matplotlib GUI pop-ups from stealing focus
    matplotlib.use('Agg')

    # Get all the config arguments.
    # This is needed to get the arguments that decide on whether to run sweeps or not
    config, _ = get_complete_config()

    if config.experiment.dev_run:
        os.environ["WANDB_MODE"] = "disabled"
        logger.info("WandB disabled for this (development) run.")

    if config.wandb.sweep.enabled:
        # Multiple experiment runs (i.e., experiments sweep)

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
        if config.wandb.sweep.num_jobs == -1:
            # Run all the jobs (i.e., all the possible combinations as per the grid search)
            num_jobs = count_grid_combinations_for_sweep_jobs(sweep_config)
        else:
            # Run num_jobs jobs
            num_jobs = config.wandb.sweep.num_jobs
        
        logger.info(f"Number of sweep jobs to run: {num_jobs}")

        # Start sweep
        wandb.agent(sweep_id,
                    function=main,
                    count=num_jobs)
        
        # End sweep time
        sweep_elapsed_time = time.time() - sweep_start_time
        logger.info(f"*** Sweep ended ***\nTotal sweep time: \n{sweep_elapsed_time} seconds ~=\n{sweep_elapsed_time/60} minutes ~=\n{sweep_elapsed_time/(60*60)} hours")

    else:
        # Single experiment run
        main()

import os
import time
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb
import matplotlib
from typing import Any, Dict, List, Tuple

# Personal codebase dependencies
import data
import models
import training
import inference
from utility.utils import log_config_dict, get_complete_config, generate_timestamped_experiment_name, save_model_metadata_for_ckpt, copy_folder
from utility.logging import logger

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('medium')


def main() -> None:
    """ Main function to run an experiment """

    logger.info("*** Experiment started ***")
    exp_start_time = time.time()

    # Get and log all the config arguments
    config, _ = get_complete_config()
    log_config_dict(config, "*** All arguments contained in the config dict ***")

    # Seed everything for reproducibility
    if config.base.seed is not None:
        pl.seed_everything(config.base.seed)

    # Setup experiment folders
    experiment_name_timestamped = generate_timestamped_experiment_name(config.experiment.name)
    experiment_folder = config.experiment.experiments_dir + f"/{experiment_name_timestamped}"
    os.makedirs(experiment_folder, exist_ok=True)

    # Data chosen
    data_module = vars(data)[config.base.data_module]
    datamodule = data_module(config.data)   # initialize the data with the data config
    logger.info(f"Data module instantiated. Now showing the total number of samples per dataloader:\n{datamodule}\n")

    # Get the image size from the datamodule. Useful for the model backbone
    image_size = datamodule.image_size
    logger.info(f"Max. image size considered (with padding): {image_size}")

    # Model chosen
    model_module = vars(models)[config.base.model_module]
    model = model_module(config.base, config.model, config.backbone_network, config.head_network, image_size)   # initialize the model with the model and network configs
    logger.trace(f"Model chosen for training: {model}")
    
    # Save the model metadata for future checkpoint use
    save_model_metadata_for_ckpt(experiment_folder, model)

    # Initialize WandB project tracking with config config
    run = wandb.init(
        project=config.wandb.wandb_project_name,
        entity=config.wandb.wandb_entity_name,
        dir=experiment_folder,
        name=experiment_name_timestamped,
        config=config
        )

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
                                                                         exp_logger)


    # Testing
    all_test_results = inference.main(config, 
                                  datamodule, 
                                  model=best_model,
                                  model_ckpt_path=None, # we use the best model found during training, so no need to specify a checkpoint path
                                  exp_logger=exp_logger)
    

    # End the wandb run
    run.finish()

    # Time taken for the experiment
    log_message = "*** Experiment ended ***\n"
    exp_elapsed_time = time.time() - exp_start_time
    log_message += f"\nTotal experiment time: \n{exp_elapsed_time} seconds ~=\n{exp_elapsed_time/60} minutes ~=\n{exp_elapsed_time/(60*60)} hours"
    logger.info(log_message)

    # Save the figures produced in the /figs folder during the experiment to the experiment folder
    experiment_figs_folder = f"{experiment_folder}/figs"
    os.makedirs(experiment_figs_folder, exist_ok=True)
    copy_folder("./figs", experiment_figs_folder)   # copy everything in the /figs folder to the current experiment folder

    # Save the results and config arguments that we are the most interested to check quickly when experimenting
    exp_results_dict = {
        'experiments_dir': config.experiment.experiments_dir,
        'exp_name': experiment_name_timestamped,
        'dataset_dir': config.data.dataset_dir,
        'exp_duration': exp_elapsed_time,
        'exp_logger': config.experiment.exp_logger,
        'data_module': config.base.data_module,
        'model_module': config.base.model_module,
        'network_backbone': config.model.backbone,
        'network_head': config.model.head,
        'model_ckpt': config.training.model_ckpt_path,
        'backbone_ckpt': config.training.backbone_ckpt_path,
        'freeze_backbone': config.training.freeze_backbone,
        'best_val_acc': train_results['best_val_acc'],
        'best_epoch': train_results['best_val_epoch'],
        'max_epochs': config.training.max_epochs,
        'train_batch_size': config.data.train_batch_size,
        'val_batch_size': config.data.val_batch_size,
        'test_batch_size': config.data.test_batch_size,
        'task_embedding': config.model.task_embedding,
        'lr': config.model.training_hparams.lr,
        'optimizer': config.model.training_hparams.optimizer,
        'scheduler': config.model.training_hparams.scheduler,
        'seed': config.base.seed,
    }
    
    exp_results_dict.update({k:v for k,v in all_test_results['test_results']['test_results_global_avg'].items()})
    exp_results_dict.update({k:v for k,v in all_test_results['test_results']['test_results_per_task_avg'].items()})
    # exp_results_dict.update({k:v for k,v in all_test_results['test_results_per_task'].items()})

    if config.data.use_gen_test_set:
        exp_results_dict.update({k:v for k,v in all_test_results['gen_test_results']['gen_test_results_global_avg'].items()})
        exp_results_dict.update({k:v for k,v in all_test_results['gen_test_results']['gen_test_results_per_task_avg'].items()})
        # exp_results_dict.update({k:v for k,v in all_test_results['gen_test_results_per_task'].items()})
    
    output_dict_df = pd.DataFrame([exp_results_dict])
    os.makedirs(config.experiment.exp_summary_results_dir, exist_ok=True)
    csv_path = os.path.join(config.experiment.exp_summary_results_dir, 'all_results_summary.csv')
    write_header = not os.path.exists(csv_path)
    output_dict_df.to_csv(csv_path, sep=';', mode='a', index=False, header=write_header)


if __name__ == '__main__':
    logger.info(f"experiment.py process ID: {os.getpid()}")

    matplotlib.use('Agg')   # prevent the matplotlib GUI pop-ups from stealing focus

    main()

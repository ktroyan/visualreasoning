import os
import time

import matplotlib
import pandas as pd
import torch
import pytorch_lightning as pl
import yaml

import wandb
from pytorch_lightning.loggers.wandb import WandbLogger

# Personal codebase dependencies
import data
import models
from utility.utils import get_complete_config, log_config_dict, get_model_from_ckpt, observe_input_output_images, \
    process_test_results, generate_timestamped_experiment_name, copy_folder
from utility.rearc.utils import check_train_test_contamination as check_rearc_train_test_contamination
from utility.logging import logger


torch.set_float32_matmul_precision('medium')    # 'high'
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def main(config, datamodule, model=None, model_ckpt_path=None, exp_logger=None):

    logger.info("*** Inference started ***")
    inference_start_time = time.time()

    # Trainer (for inference/testing)
    trainer = pl.Trainer(num_nodes=1,   # number of gpu nodes for distributed training
                         logger=exp_logger,
                         devices=config.base.n_gpus,
                         accelerator='auto',
                         enable_progress_bar=True,
                        )

    # Model
    if model is not None:
        log_message = f"A specific (most likely trained) model instance from class {model.__class__} was given for inference.\n\n"
        # log_message += f"Model: {model}\n"
        log_message += "No model checkpoint was given directly."
        logger.info(log_message)
    
    elif model_ckpt_path is not None:
        model = get_model_from_ckpt(model_ckpt_path)
        log_message = f"Model loaded from checkpoint for inference: {model_ckpt_path}\n\n"
        # log_message += f"Model: {model}\n"
        logger.info(log_message)

    else:
        raise ValueError("No model instance or model checkpoint path was given for inference.")


    logger.info(f"All hyperparameters of the model used for inference:\n{model.hparams} \n")

    # Sanity check for data contamination between train and test dataloaders
    if config.inference.check_data_contamination:
        train_dataloader = datamodule.train_dataloader()
        test_dataloader = datamodule.test_dataloader()[0] if isinstance(datamodule.test_dataloader(), list) else datamodule.test_dataloader()
        check_rearc_train_test_contamination(train_dataloader, test_dataloader)


    # Testing
    trainer.test(model=model, datamodule=datamodule, verbose=True)  # NOTE: if more than one test dataloader was created in the datamodule, all the test dataloaders will be used for testing
   

    # Additional logging and plotting if needed
    if config.inference.inference_verbose == 1 and config.base.data_env in ["REARC", "BEFOREARC"]:
        # TODO: Implement for other data environments

        logger.debug(f"Test predictions: {model.test_preds}")
        logger.debug(f"Test targets: {model.test_targets}")

        test_dataloader = datamodule.test_dataloader()[0] if isinstance(datamodule.test_dataloader(), list) else datamodule.test_dataloader()
        observe_input_output_images(dataloader=test_dataloader, batch_id=0, n_samples=4, split="test")

        if config.data.use_gen_test_set:
            logger.debug(f"Sys-gen test predictions: {model.gen_test_preds}")
            logger.debug(f"Sys-gen test targets: {model.gen_test_targets}")

            gen_test_dataloader = datamodule.test_dataloader()[1]
            observe_input_output_images(dataloader=gen_test_dataloader, batch_id=0, n_samples=4, split="gen_test")            

    # Process the test results for better logging
    test_results = model.test_results
    processed_test_results = process_test_results(config, test_results, test_type="test", exp_logger=None)

    if config.data.use_gen_test_set:
        gen_test_results = model.gen_test_results
        processed_gen_test_results = process_test_results(config, gen_test_results, test_type="gen_test", exp_logger=None)
    else:
        processed_gen_test_results = {}

    all_test_results = {'test_results': processed_test_results, 'gen_test_results': processed_gen_test_results}

    # End of inference
    log_message = "*** Inference ended ***\n"
    inference_elapsed_time = time.time() - inference_start_time
    log_message += f"\nTotal inference time: \n{inference_elapsed_time} seconds ~=\n{inference_elapsed_time/60} minutes ~=\n{inference_elapsed_time/(60*60)} hours"
    logger.info(log_message)

    return all_test_results


def run_inference_from_main():

    # Get and log all the config arguments
    config, _ = get_complete_config()
    log_config_dict(config)

    logger.info("*** Experiment started ***")
    exp_start_time = time.time()

    # Setup experiment folders
    experiment_name_timestamped = generate_timestamped_experiment_name("experiment")
    experiment_folder = config.experiment.experiments_dir + f"/{experiment_name_timestamped}"
    os.makedirs(experiment_folder, exist_ok=True)

    # Initialize WandB project run tracking
    run = wandb.init(
        project=config.wandb.wandb_project_name,  # ignored if using sweeps
        entity=config.wandb.wandb_entity_name,  # ignored if using sweeps
        dir=experiment_folder,
        name=experiment_name_timestamped,
    )
    wandb_subfolder = "/" + wandb.run.id if wandb.run is not None else ""

    # Seed everything for reproducibility
    if config.base.seed is not None:
        pl.seed_everything(config.base.seed)

    # Data chosen
    data_module = vars(data)[config.base.data_module]
    datamodule = data_module(config.data, config.model)  # initialize the data with the data config
    logger.info(f"Data module instantiated. Now showing the total number of samples per dataloader:\n{datamodule}\n")

    # Get the image size from the datamodule. Useful for the model backbone
    image_size = datamodule.image_size
    logger.info(f"Image size considered (with padding): {image_size}")

    # Model chosen
    model_ckpt = config.inference.inference_model_ckpt
    if model_ckpt is not None and model_ckpt != "":
        logger.info(f"Model checkpoint path chosen for inference: {model_ckpt}")
        model = None
    else:
        model_module = vars(models)[config.base.model_module]
        model = model_module(config.base, config.model, config.backbone_network, config.head_network,
                             image_size)  # initialize the model with the model and network configs
        logger.info(f"Model chosen for inference w.r.t. the current config files: {model}")

    # Initialize the experiment logger
    if config.experiment.exp_logger == 'wandb':
        exp_logger = WandbLogger(project=config.wandb.wandb_project_name, name=experiment_name_timestamped,
                                 save_dir=experiment_folder, log_model=config.wandb.log_model)
    else:
        logger.warning(
            f"Experiment logger {config.experiment.exp_logger} not recognized. The experiment logger is set to Null. Otherwise, choose 'wandb'.")
        exp_logger = None

    all_test_results = main(config, datamodule, model, model_ckpt, exp_logger=exp_logger)

    # End the wandb run
    run.finish()

    # Time taken for the experiment
    log_message = "*** Experiment ended ***\n"
    exp_elapsed_time = time.time() - exp_start_time
    log_message += f"\nTotal experiment time: \n{exp_elapsed_time} seconds ~=\n{exp_elapsed_time / 60} minutes ~=\n{exp_elapsed_time / (60 * 60)} hours"
    logger.info(log_message)

    # Save the figures produced in the /figs folder during the experiment to the experiment folder
    experiment_figs_folder = f"{experiment_folder}/figs"
    os.makedirs(experiment_figs_folder, exist_ok=True)
    copy_folder(f"./figs{wandb_subfolder}",
                experiment_figs_folder)  # copy everything in the /figs folder to the current experiment folder

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
        'best_val_acc': None,
        'best_epoch': None,
        'max_epochs': config.training.max_epochs,
        'train_batch_size': config.data.train_batch_size,
        'val_batch_size': config.data.val_batch_size,
        'test_batch_size': config.data.test_batch_size,
        'task_embedding': config.model.task_embedding,
        'lr': config.model.training_hparams.lr,
        'optimizer': config.model.training_hparams.optimizer,
        'scheduler_type': config.model.training_hparams.scheduler.type,
        'scheduler_interval': config.model.training_hparams.scheduler.interval,
        'scheduler_frequency': config.model.training_hparams.scheduler.frequency,
        'seed': config.base.seed,
    }

    exp_results_dict.update({k: v for k, v in all_test_results['test_results']['test_results_global_avg'].items()})
    exp_results_dict.update({k: v for k, v in all_test_results['test_results']['test_results_per_task_avg'].items()})
    # exp_results_dict.update({k:v for k,v in all_test_results['test_results_per_task'].items()})

    if config.data.use_gen_test_set:
        exp_results_dict.update(
            {k: v for k, v in all_test_results['gen_test_results']['gen_test_results_global_avg'].items()})
        exp_results_dict.update(
            {k: v for k, v in all_test_results['gen_test_results']['gen_test_results_per_task_avg'].items()})
        # exp_results_dict.update({k:v for k,v in all_test_results['gen_test_results_per_task'].items()})

    output_dict_df = pd.DataFrame([exp_results_dict])
    os.makedirs(config.experiment.exp_summary_results_dir, exist_ok=True)
    csv_path = os.path.join(config.experiment.exp_summary_results_dir, 'all_results_summary.csv')
    write_header = not os.path.exists(csv_path)
    output_dict_df.to_csv(csv_path, sep=';', mode='a', index=False, header=write_header)


if __name__ == '__main__':
    logger.info(f"inference.py process ID: {os.getpid()}")

    matplotlib.use('Agg')  # prevent the matplotlib GUI pop-ups from stealing focus

    # Get all the config arguments. This is needed to get the arguments that decide on whether to run sweeps or not and for how many sweeps
    config, _ = get_complete_config()

    if config.wandb.sweep.enabled:

        # Start sweep time
        sweep_start_time = time.time()
        logger.info(f"*** Sweep started ***")

        # Get the sweep yaml file for the data environment specified in the base config
        sweep_yaml_file = f"./configs/sweep_{(config.base.data_env).lower()}.yaml"

        # Load the sweep config yaml file
        with open(sweep_yaml_file, 'r') as f:
            sweep_config = yaml.safe_load(f)

        # Setup WandB sweep
        sweep_id = wandb.sweep(sweep=sweep_config,
                               project=config.wandb.wandb_project_name)

        # Start sweep job
        wandb.agent(sweep_id,
                    function=run_inference_from_main,
                    count=config.wandb.sweep.num_sweeps)

        # End sweep time
        sweep_elapsed_time = time.time() - sweep_start_time
        logger.info(
            f"*** Sweep ended ***\nTotal sweep time: \n{sweep_elapsed_time} seconds ~=\n{sweep_elapsed_time / 60} minutes ~=\n{sweep_elapsed_time / (60 * 60)} hours")

    else:
        run_inference_from_main()


import os
import sys
import argparse
import time
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb

# Personal codebase dependencies
import data
import models
import training
import inference
from utility.utils import generate_timestamped_experiment_name, parse_args_and_configs, log_args_namespace, get_config_specific_args_from_args
from utility.logging import logger

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('medium')


def main():

    logger.info("*** Experiment started ***")
    exp_start_time = time.time()

    # Get the CLI arguments
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()

    # Configs and CLI arguments (frequently changing arguments)
    # NOTE: if an argument is not given through the CLI, the one in the related config file will be used (instead of being defaulted to None)
    parser.add_argument('--seed', type=int, default=None, help='seed for reproducibility')
    parser.add_argument('--max_epochs', type=int, default=None, help='maximum number of epochs to be performed during training of the model')
    parser.add_argument('--use_gen_test_set', action='store_true', help='whether to use the systematic generalization test set')
    parser.add_argument('--test_in_and_out_domain', action='store_true', help='whether to test on both in and out of domain test sets, given that a sys-gen test set is provided')

    # Configs arguments (consistent arguments)
    parser.add_argument("--general_config", default="./configs/general.yaml", help="from where to load the general YAML config", metavar="FILE")
    parser.add_argument("--exp_config", default="./configs/experiment.yaml", help="from where to load the experiment YAML config", metavar="FILE")
    parser.add_argument("--data_config", default="./configs/data.yaml", help="from where to load the YAML config of the chosen data", metavar="FILE")
    parser.add_argument("--model_config", default="./configs/model.yaml", help="from where to load the YAML config of the chosen model", metavar="FILE")
    args = parse_args_and_configs(parser, argv)
    parser.add_argument("--network_config", default=f"./configs/networks/{args.model_backbone}.yaml", help="from where to load the YAML config of the chosen neural network", metavar="FILE")
    args = parse_args_and_configs(parser, argv)
    parser.add_argument("--training_config", default="./configs/training.yaml", help="from where to load the training YAML config", metavar="FILE")
    parser.add_argument("--inference_config", default="./configs/inference.yaml", help="from where to load the inference YAML config", metavar="FILE")
    args = parse_args_and_configs(parser, argv)

    # Log all the arguments in the args Namespace
    log_args_namespace(args)

    # Seed everything for reproducibility
    if args.seed is not None:
        pl.seed_everything(args.seed)

    # Data chosen
    data_module = vars(data)[args.data_module]
    logger.info(f"Data module: {data_module}")
    data_args = get_config_specific_args_from_args(args, args.data_config)
    datamodule = data_module(**data_args)   # initializing the data
    logger.info(f"Data module instantiated (showing the total number of samples per dataloader):\n{datamodule}\n")

    # Model chosen
    model_module = vars(models)[args.model_module]
    logger.info(f"Model module: {model_module}")
    model_args = get_config_specific_args_from_args(args, args.model_config)
    model = model_module(**model_args)   # initializing the model
    logger.trace(f"Model chosen for training: {model} \n")
    logger.info(f"Model hyperparameters for training:\n{model.hparams} \n")

    # Setup experiment folders
    os.makedirs(args.experiments_dir, exist_ok=True)
    experiment_basename = args.dataset_dir.split('/')[-1]
    experiment_name = generate_timestamped_experiment_name(experiment_basename)
    experiment_folder = args.experiments_dir + f"/{experiment_name}"
    os.makedirs(experiment_folder, exist_ok=True)
    
    # Initialize WandB project tracking with config args
    wandb.init(
    project="VisReas-project",
    entity="klim-t",    # this should be a WandB username or team name
    dir=experiment_folder,
    name=experiment_name,
    config=args
    )

    # Initialize the experiment logger
    if args.exp_logger == 'wandb':
        exp_logger = WandbLogger(project='VisReas-project', name=experiment_name, save_dir=experiment_folder, log_model=False)    # NOTE: I set log_model=False because otherwise I get an error of connection with wandb (?)
    else:
        exp_logger = TensorBoardLogger(experiment_folder, default_hp_metric=False)

    # Training
    trainer, best_model, best_model_ckpt, train_results = training.main(args, 
                                                                         experiment_folder, 
                                                                         datamodule, 
                                                                         model, 
                                                                         exp_logger)

    # Testing
    all_test_results = inference.main(args, 
                                  datamodule, 
                                  model=best_model,
                                  model_ckpt_path=None, # we use the best model found during training, so no need to specify a checkpoint path
                                  exp_logger=exp_logger)
    
    # End the wandb run
    wandb.finish()

    # Time taken for the experiment
    log_message = "*** Experiment ended ***\n"
    exp_elapsed_time = time.time() - exp_start_time
    log_message += f"\nTotal experiment time: \n{exp_elapsed_time} seconds ~=\n{exp_elapsed_time/60} minutes ~=\n{exp_elapsed_time/(60*60)} hours"
    logger.info(log_message)

    # Save the results and config arguments that we are the most interested to check quickly when experimenting
    # TODO: only consider the relevant args and results
    exp_results_dict = {
        'experiments_dir': args.experiments_dir,
        'dataset_dir': args.dataset_dir,
        'exp_duration': exp_elapsed_time,
        'exp_logger': args.exp_logger,
        'exp_name': experiment_name,
        'seed': args.seed,
        'data_module': args.data_module,
        'model_module': args.model_module,
        'model_ckpt': args.model_ckpt_path,
        'backbone_ckpt': args.backbone_ckpt_path,
        'freeze_backbone': args.freeze_backbone,
        'best_val_acc': train_results['best_val_acc'],
        'best_epoch': train_results['best_epoch'],
        'max_epochs': args.max_epochs,
        'model_backbone': args.model_backbone,
        'train_batch_size': args.train_batch_size,
        'lr': args.lr,
    }
    
    exp_results_dict.update({k:v for k,v in all_test_results['test_results']['test_results_global_avg'].items()})
    exp_results_dict.update({k:v for k,v in all_test_results['test_results']['test_results_per_task_avg'].items()})
    # exp_results_dict.update({k:v for k,v in test_results['test_results_per_task'].items()})

    if args.use_gen_test_set:
        exp_results_dict.update({k:v for k,v in all_test_results['gen_test_results']['gen_test_results_global_avg'].items()})
        exp_results_dict.update({k:v for k,v in all_test_results['gen_test_results']['gen_test_results_per_task_avg'].items()})
        # exp_results_dict.update({k:v for k,v in gen_test_results['gen_test_results_per_task'].items()})
    
    output_dict_df = pd.DataFrame([exp_results_dict])
    db_folder = os.path.join(args.experiments_dir, args.exp_summary_results_dir)
    os.makedirs(db_folder, exist_ok=True)
    csv_path = os.path.join(db_folder, 'all_results_summary.csv')
    write_header = not os.path.exists(csv_path)
    output_dict_df.to_csv(csv_path, sep=';', mode='a', index=False, header=write_header)


if __name__ == '__main__':
    logger.info(f"experiment.py process ID: {os.getpid()}")

    main()

import os
import sys
import time
import argparse
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, TQDMProgressBar
from typing import Any, Dict, List, Tuple

# Personal codebase dependencies
import data
import models
from utility.utils import parse_args_and_configs, log_args_namespace, get_config_specific_args_from_args, get_latest_ckpt
from utility.logging import logger

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('medium')

class MetricsCallback(Callback):
    """A PyTorch Lightning callback to handle and store metrics during the training process.

    This callback allows to perform actions at key moments of the PyTorch Lightning `Trainer`
    process, such as at the end of a training epoch or after validation. It inherits from the `Callback`
    class provided by PyTorch Lightning.


    Attributes:
        metrics (dict): A dictionary to store metrics collected during training, validation, or testing.
                        The keys are metric names, and the values are lists of metric values over time.
        all_keys (list): A list to keep track of all the metric keys observed during the training process.
        
    """

    def __init__(self, verbose=True):
        super().__init__()

        self.verbose = verbose
        self.metrics = []  # store metrics for each validation epoch
        self.all_keys = []  # keep track of all metric keys observed

    def get_all_metrics(self):
        """This method is called after the training (and testing) processes to retrieve all the collected metrics. 
        It iterates over the stored metrics and organizes them into a dictionary where each key corresponds to a 
        metric name and the value is a list of metric values collected over the epochs.

        Returns:
            dict: all the collected metrics during training for each epoch
        """

        all_metrics = {}
        for k in self.all_keys:
            all_metrics[k] = []

        for m in self.metrics:
            for k in self.all_keys:
                v = m[k] if k in m else np.nan
                all_metrics[k].append(v)

        return all_metrics

    def on_train_epoch_end(self, trainer, pl_model_module):
        """ The name of the method is predefined by PyTorch Lightning. It is called at the end of each training epoch.
        
        Note that what we can access here with trainer is similar to what we can access with self in the VisReasModel class (that inherits from pl.LightningModule).

        Args:
            trainer (pl.Trainer): Instance of the PyTorch Lightning Trainer class.
            pl_model_module (VisReasModel(pl.LightningModule)): The Visual Reasoning model class instance that inherits from pl.LightningModule.
        """

        # Collect the epoch metrics
        epoch_metrics = {}
        for k,v in trainer.callback_metrics.items():
            epoch_metrics[k] = v.item()
            if k not in self.all_keys:
                self.all_keys.append(k)

        self.metrics.append(epoch_metrics)


        if self.verbose >= 1:
            epoch_metrics = trainer.callback_metrics

            # logger.info(f"Considering the metrics: {epoch_metrics.keys()}")
            log_message = ""

            # TODO: when would we need to use .mean() ? So far it is equivalent to not taking the mean.
            # log_message = f"[Epoch {trainer.current_epoch}] Mean metrics: \n"
            # for k, v in epoch_metrics.items():
            #     log_message += f"{k}: {v.mean()} \n"

            log_message += f"[Epoch {trainer.current_epoch}] Current epoch metrics: \n"
            for k, v in epoch_metrics.items():
                log_message += f"{k}: {v} \n"
            
            logger.info(f"Learning rate used at epoch {trainer.current_epoch}: {pl_model_module.lr_schedulers().get_last_lr()}")    # this yields learning_rate_epoch in the logs
            logger.info(log_message)

        if self.verbose >= 2:
            # Log the predicted labels and the true labels for training and validation
            logger.info(f"Epoch train predictions: \n{pl_model_module.train_preds}")
            logger.info(f"Epoch train labels: \n{pl_model_module.train_labels}")
            logger.info(f"Epoch val predictions: \n{pl_model_module.val_preds}")
            logger.info(f"Epoch val labels: \n{pl_model_module.val_labels}")

        # Reset the predictions and labels list for the next epoch
        pl_model_module.train_preds = []
        pl_model_module.train_labels = []
        pl_model_module.val_preds = []
        pl_model_module.val_labels = []

        # logger.info(f"Class of the pl model module: {pl_model_module.__class__}")   # e.g.: CVRModel (which inherits from VisReasModel which inherits from pl.LightningModule)
        # logger.info(f"Attributes of the instance of the pl model module: {pl_model_module.__dict__}")    


def init_callbacks(args, training_folder):

    callbacks = {}

    # Model checkpoint callback
    model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=training_folder, 
                                                    save_top_k=1, 
                                                    mode='max', 
                                                    monitor='metrics/val_acc', # 'metrics/val_loss'
                                                    every_n_epochs=args.ckpt_period, 
                                                    save_last=True)

    callbacks['model_checkpoint'] = model_checkpoint

    # Early stopping callback
    if args.early_stopping != 0:
        early_stopping = pl.callbacks.EarlyStopping(monitor='metrics/val_acc', mode='max', patience=args.es_patience, strict=True, verbose=True)   # we can use stopping_threshold=0.99 to stop when the accuracy metric reaches 0.99
        # early_stopping = pl.callbacks.EarlyStopping(monitor='metrics/val_loss', mode='min', patience=args.es_patience, strict=True, verbose=True)   # we can use stopping_threshold=0.99 to stop when the accuracy metric reaches 0.99
        callbacks['early_stopping'] = early_stopping

    # Progress bar callback
    if args.use_progress_bar:
        progress_bar = TQDMProgressBar(refresh_rate=args.refresh_rate, leave=False)
        callbacks['progress_bar'] = progress_bar

    # Metrics callback
    metrics_callback = MetricsCallback(verbose=args.metrics_callback_verbose)
    callbacks['metrics_callback'] = metrics_callback

    return callbacks

def get_already_trained_model(experiments_dir, model_ckpt_path):
    if model_ckpt_path != '':
        ckpt_path = model_ckpt_path
        logger.info(f'Resuming training from given checkpoint {ckpt_path}')

    else:
        ckpt_path = get_latest_ckpt(experiments_dir)
        if ckpt_path:
            logger.info(f'Resuming training from last found checkpoint {ckpt_path} in {experiments_dir}')
        else:
            logger.info(f"No already trained model checkpoint found in {experiments_dir}. Training from scratch..?")
    
    return ckpt_path

def train(args, model, datamodule, callbacks, exp_logger=None, checkpoint_path=None):

    # Handle callbacks. Note that the callbacks objects are updated by the pl.Trainer() during the training process
    callbacks_list = list(callbacks.values())    # convert the dict of callbacks to a list of callbacks so that it can be passed to the Trainer()

    # Training
    trainer = pl.Trainer(default_root_dir=exp_logger.save_dir if exp_logger else None,
                         num_nodes=1,
                         logger=exp_logger, 
                         callbacks=callbacks_list, 
                         max_epochs=args.max_epochs,
                         devices=args.gpus,
                         accelerator='auto',
                         precision=args.trainer_precision,
                         gradient_clip_val=None,
                         num_sanity_val_steps=0, 
                         enable_progress_bar=True,
                         enable_checkpointing=True,
                         log_every_n_steps=args.log_every_n_steps,
                         fast_dev_run=False
                        )

    trainer.fit(model, datamodule, ckpt_path=checkpoint_path)

    return trainer, callbacks

def get_best_model_from_training(model, callbacks):
    best_model_ckpt_path = callbacks['model_checkpoint'].best_model_path
    
    if best_model_ckpt_path == "":
        best_model = model
    else:
        model_module = model.__class__
        best_model = model_module.load_from_checkpoint(checkpoint_path=best_model_ckpt_path)

    logger.trace(f"Best model loaded: {best_model}")

    return best_model, best_model_ckpt_path

def main(args, training_folder, datamodule, model, exp_logger=None):

    logger.info("*** Training started ***")
    training_start_time = time.time()

    # Resume training with a model checkpoint or load a backbone model checkpoint
    ckpt_path = None
    if args.resume_training:
        ckpt_path = get_already_trained_model(args.experiments_dir, args.model_ckpt_path)
    else:
        if args.backbone_ckpt_path != '':
            model.load_backbone_weights(args.backbone_ckpt_path)

    # Freeze the backbone model if needed
    if args.freeze_backbone:
        model.freeze_backbone_model()

    # Training callbacks
    callbacks = init_callbacks(args, training_folder)

    # Training
    trainer, callbacks = train(args, model, datamodule, callbacks, exp_logger, ckpt_path)

    # Get best model from the training process
    best_model, best_model_ckpt_path = get_best_model_from_training(model, callbacks)

    # Metrics results
    metrics = callbacks['metrics_callback'].get_all_metrics()

    best_val_acc = np.nanmax(metrics['metrics/val_acc'] + [0])
    best_epoch = (np.nanargmax(metrics['metrics/val_acc'] + [0]) + 1) * args.ckpt_period

    log_message = "All epoch training metrics: \n"
    for k, v in metrics.items():
        log_message += f"{k}: {v}" + "\n"
    logger.info(log_message)

    # Access the wandb experiment and save the logs for the model hyperparameters and additional results (than those already logged with log_dict() in the model file)
    if exp_logger:
        exp_logger.log_hyperparams(model.hparams)
        exp_logger.experiment.log({'best_epoch': best_epoch, 'best_val_acc': best_val_acc})

    train_results = {
        'metrics': metrics,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch
    }

    log_message = "*** Training ended ***\n"
    training_elapsed_time = time.time() - training_start_time
    log_message += f"\nTotal training time: \n{training_elapsed_time} seconds ~=\n{training_elapsed_time/60} minutes ~=\n{training_elapsed_time/(60*60)} hours"
    logger.info(log_message)

    return trainer, best_model, best_model_ckpt_path, train_results

if __name__ == '__main__':
    logger.info(f"training.py process ID: {os.getpid()}")

    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()

    # Configs and CLI arguments (frequently changing arguments)
    parser.add_argument('--seed', type=int, default=None, help='seed for reproducibility')
    parser.add_argument('--max_epochs', type=int, default=None, help='maximum number of epochs to be performed during training of the model')
    parser.add_argument('--resume_training', action='store_true', help='whether to resume training from a given checkpoint')

    # Configs arguments (consistent arguments)
    parser.add_argument("--general_config", default="./configs/general.yaml", help="from where to load the general YAML config", metavar="FILE")
    parser.add_argument("--data_config", default="./configs/data.yaml", help="from where to load the YAML config of the chosen data", metavar="FILE")
    parser.add_argument("--model_shared_config", default="./configs/model_shared.yaml", help="from where to load the YAML config of the chosen model", metavar="FILE")
    args = parse_args_and_configs(parser, argv)
    parser.add_argument("--model_config", default=f"./configs/models/{args.model_module}.yaml", help="from where to load the YAML config of the chosen model", metavar="FILE")
    parser.add_argument("--training_config", default="./configs/training.yaml", help="from where to load the YAML config of training-related arguments", metavar="FILE")
    args = parse_args_and_configs(parser, argv)
    parser.add_argument("--network_config", default=f"./configs/networks/{args.model_backbone}.yaml", help="from where to load the YAML config of the chosen neural network", metavar="FILE")
    args = parse_args_and_configs(parser, argv)

    # Log all the arguments in the Namespace
    log_args_namespace(args)

    # Seed everything for reproducibility
    if args.seed is not None:
        pl.seed_everything(args.seed)

    # Create the training folder
    training_folder = "./" + args.data_module.replace("DataModule", "") + "/training"
    os.makedirs(training_folder, exist_ok=True)

    # Data chosen
    data_module = vars(data)[args.data_module]
    logger.info(f"Data module: {data_module}")
    data_args = get_config_specific_args_from_args(args, args.data_config)
    datamodule = data_module(**data_args)   # initializing the data
    logger.info(f"Datamodule (showing the total number of samples per dataloader):\n {datamodule} \n")

    # Model chosen
    model_module = vars(models)[args.model_module]
    logger.info(f"Model module: {model_module}")
    model_args = get_config_specific_args_from_args(args, args.model_config)
    model_shared_args = get_config_specific_args_from_args(args, args.model_shared_config)
    model_args.update(model_shared_args)
    model = model_module(**model_args)   # initializing the model
    logger.trace(f"Model for training: {model} \n")
    logger.info(f"Model hyperparameters for training:\n{model.hparams} \n")

    trainer, best_model, best_model_ckpt_path, train_results = main(args, training_folder, datamodule, model, exp_logger=None)
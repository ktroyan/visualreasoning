import os
import time
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, TQDMProgressBar
from typing import Any, Dict, List, Tuple

# Personal codebase dependencies
import data
import models
from utility.utils import get_complete_config, log_config_dict, get_latest_ckpt
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
            
            log_message += f"\nLearning rate at the end of epoch {trainer.current_epoch}: {pl_model_module.lr_schedulers().get_last_lr()}"    # this yields learning_rate_epoch in the logs
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


def init_callbacks(config, training_folder):

    callbacks = {}

    # Model checkpoint callback
    model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=training_folder, 
                                                    save_top_k=1, 
                                                    mode='max', 
                                                    monitor='metrics/val_acc', # 'metrics/val_loss'
                                                    every_n_epochs=config.training.ckpt_period, 
                                                    save_last=True)

    callbacks['model_checkpoint'] = model_checkpoint

    # Early stopping callback
    if config.training.early_stopping.enabled:
        early_stopping = pl.callbacks.EarlyStopping(monitor='metrics/val_acc', mode='max', patience=config.training.early_stopping.es_patience, strict=True, verbose=True)   # we can use stopping_threshold=0.99 to stop when the accuracy metric reaches 0.99
        # early_stopping = pl.callbacks.EarlyStopping(monitor='metrics/val_loss', mode='min', patience=config.training.es_patience, strict=True, verbose=True)   # we can use stopping_threshold=0.99 to stop when the accuracy metric reaches 0.99
        callbacks['early_stopping'] = early_stopping

    # Progress bar callback
    if config.training.progress_bar.enabled:
        progress_bar = TQDMProgressBar(refresh_rate=config.training.progress_bar.refresh_rate, leave=False)
        callbacks['progress_bar'] = progress_bar

    # Metrics callback
    metrics_callback = MetricsCallback(verbose=config.training.metrics_callback_verbose)
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

def train(config, model, datamodule, callbacks, exp_logger=None, checkpoint_path=None):

    # Handle callbacks. Note that the callbacks objects are updated by the pl.Trainer() during the training process
    callbacks_list = list(callbacks.values())    # convert the dict of callbacks to a list of callbacks so that it can be passed to the Trainer()

    # Training
    trainer = pl.Trainer(default_root_dir=exp_logger.save_dir if exp_logger else None,
                         num_nodes=1,
                         logger=exp_logger, 
                         callbacks=callbacks_list, 
                         max_epochs=config.training.max_epochs,
                         devices=config.base.gpus,
                         accelerator='auto',
                         precision=config.training.trainer_precision,
                         gradient_clip_val=None,
                         num_sanity_val_steps=0, 
                         enable_progress_bar=True,
                         enable_checkpointing=True,
                         log_every_n_steps=config.training.log_every_n_steps,
                         fast_dev_run=False,
                        #  profiler='simple'
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

def main(config, training_folder, datamodule, model, exp_logger=None):

    logger.info("*** Training started ***")
    training_start_time = time.time()

    # Resume training with a model checkpoint or load a backbone model checkpoint
    ckpt_path = None
    if config.training.resume_training:
        ckpt_path = get_already_trained_model(config.experiment.experiments_dir, config.training.model_ckpt_path)
    else:
        if config.training.backbone_ckpt_path != '':
            model.load_backbone_weights(config.training.backbone_ckpt_path)

    # Freeze the backbone model if needed
    if config.training.freeze_backbone:
        model.freeze_backbone_model()

    # Training callbacks
    callbacks = init_callbacks(config, training_folder)

    # Training
    trainer, callbacks = train(config, model, datamodule, callbacks, exp_logger, ckpt_path)

    # Get best model from the training process
    best_model, best_model_ckpt_path = get_best_model_from_training(model, callbacks)

    # Metrics results
    metrics = callbacks['metrics_callback'].get_all_metrics()

    best_val_acc = np.nanmax(metrics['metrics/val_acc'] + [0])
    best_val_epoch = (np.nanargmax(metrics['metrics/val_acc'] + [0]) + 1) * config.training.ckpt_period

    log_message = "All epoch training metrics: \n"
    for k, v in metrics.items():
        log_message += f"{k}: {v}" + "\n"
    logger.info(log_message)

    # Access the wandb experiment and save the logs for the model hyperparameters and additional results (than those already logged with log_dict() in the model file)
    if exp_logger:
        exp_logger.log_hyperparams(model.hparams)
        exp_logger.experiment.log({'best_val_epoch': best_val_epoch, 'best_val_acc': best_val_acc})

    train_results = {
        'metrics': metrics,
        'best_val_acc': best_val_acc,
        'best_val_epoch': best_val_epoch
    }

    log_message = "*** Training ended ***\n"
    training_elapsed_time = time.time() - training_start_time
    log_message += f"\nTotal training time: \n{training_elapsed_time} seconds ~=\n{training_elapsed_time/60} minutes ~=\n{training_elapsed_time/(60*60)} hours"
    logger.info(log_message)

    return trainer, best_model, best_model_ckpt_path, train_results


if __name__ == '__main__':
    logger.info(f"training.py process ID: {os.getpid()}")

    # Get and log all the config arguments
    config = get_complete_config()
    log_config_dict(config)

    # Seed everything for reproducibility
    if config.base.seed is not None:
        pl.seed_everything(config.base.seed)

    # Data chosen
    data_module = vars(data)[config.base.data_module]
    datamodule = data_module(config.data)   # initialize the data with the data config
    logger.info(f"Data module instantiated. Now showing the total number of samples per dataloader:\n{datamodule}\n")

    # Model chosen
    model_module = vars(models)[config.base.model_module]
    model = model_module(config.model, config.backbone_network, config.head_network)   # initialize the model with the model and network configs
    logger.trace(f"Model chosen for training: {model}")

    # Create the training folder
    training_folder = f"./{config.data.data_env}/training"
    os.makedirs(training_folder, exist_ok=True)

    trainer, best_model, best_model_ckpt_path, train_results = main(config, training_folder, datamodule, model, exp_logger=None)
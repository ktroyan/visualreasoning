import os
import time
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, TQDMProgressBar
import numpy as np
import json
from typing import Any, Dict, List, Tuple

# Personal codebase dependencies
import data
import models
from utility.utils import get_complete_config, log_config_dict, get_latest_ckpt
from utility.rearc.utils import plot_metrics_locally as plot_rearc_metrics_locally
from utility.cvr.utils import plot_metrics_locally as plot_cvr_metrics_locally
from utility.custom_logging import logger


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('medium')    # 'high'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = 'true'

# os.environ['TORCH_LOGS'] = "graph_breaks"
# os.environ['TORCH_LOGS'] = "recompiles"
# os.environ['TORCH_LOGS'] = "recompiles,dynamic"


class MetricsCallback(Callback):
    """
    A PyTorch Lightning callback to handle and store metrics during the training process.

    It allows us to perform actions at key moments of the PyTorch Lightning `Trainer`
    process, such as at the end of a training epoch and more. 
    
    It inherits from the `Callback` class provided by PyTorch Lightning.
    """

    def __init__(self, base_config, data_config, save_metrics_folder, verbose=True):
        super().__init__()

        self.base_config = base_config
        self.data_config = data_config
        self.run_metrics_file = save_metrics_folder + "/run_metrics.jsonl"
        self.verbose = verbose
        
        self.metrics = []   # store metrics for each epoch
        self.all_keys = []  # keep track of all metric keys

        # Initialize the lists to store the metrics for all the epochs for local plotting
        self.train_loss_epoch = []
        self.train_acc_epoch = []
        self.train_grid_acc_epoch = []

        self.val_loss_epoch = []
        self.val_acc_epoch = []
        self.val_grid_acc_epoch = []

        if data_config.validate_in_and_out_domain:
            self.gen_val_loss_epoch = []
            self.gen_val_acc_epoch = []
            self.gen_val_grid_acc_epoch = []


    def get_all_epoch_metrics(self) -> Dict[str, List[float]]:
        """
        This method is called after the training (and testing) processes to retrieve all the collected metrics. 
        It iterates over the stored metrics and organizes them into a dictionary where each key corresponds to a 
        metric name and the value is a list of metric values collected over the epochs.

        Returns:
            Dict: all the collected metrics during training for each epoch
        """

        all_metrics = {}
        for k in self.all_keys:
            all_metrics[k] = []

        for m in self.metrics:
            for k in self.all_keys:
                v = m[k] if k in m else np.nan
                all_metrics[k].append(v)

        return all_metrics

    def get_all_local_plotting_metrics(self):
        """
        This method is called after the training process to retrieve all the collected epoch metrics for local plotting. 
        """

        all_local_plotting_metrics = {
            'train_loss_epoch': self.train_loss_epoch,
            'train_acc_epoch': self.train_acc_epoch,
            'val_loss_epoch': self.val_loss_epoch,
            'val_acc_epoch': self.val_acc_epoch
        }

        if self.base_config.data_env in ["REARC", "BEFOREARC"]:
            all_local_plotting_metrics.update({
                'train_grid_acc_epoch': self.train_grid_acc_epoch,
                'val_grid_acc_epoch': self.val_grid_acc_epoch
            })

        if self.data_config.validate_in_and_out_domain:
            all_local_plotting_metrics.update({
                'gen_val_loss_epoch': self.gen_val_loss_epoch,
                'gen_val_acc_epoch': self.gen_val_acc_epoch,
            })

            if self.base_config.data_env in ["REARC", "BEFOREARC"]:
                all_local_plotting_metrics.update({
                    'gen_val_grid_acc_epoch': self.gen_val_grid_acc_epoch
                })

        return all_local_plotting_metrics

    def on_train_start(self, trainer, pl_model_module):
        logger.info("Training started!")

    def tensor_list_to_python_list(self, lst):
        return [x.item() if isinstance(x, torch.Tensor) else x for x in lst]

    def on_train_epoch_end(self, trainer, pl_model_module):
        """ 
        The name of the method is predefined by PyTorch Lightning. It is called at the end of each training epoch.
        NOTE: This method is called before the on_train_epoch_end() method of the pl.LightningModule class.
        
        Note that what we can access here with trainer and pl_model_module is similar to what we can access with self in the VisReasModel class (that inherits from pl.LightningModule).
        Hence, we could also override the on_train_epoch_end() method in the VisReasModel class to access the same information.
        But since we only handle training-related metrics and data here, it is more appropriate to handle it in this callback class.

        Args:
            trainer (pl.Trainer): Instance of the PyTorch Lightning Trainer class.
            pl_model_module (VisReasModel(pl.LightningModule)): The Visual Reasoning model class instance that inherits from pl.LightningModule.
        """

        # TODO: 
        # What are the step metrics given when using .log_dict() from WandB? Because there are only as many of them as there are epochs.
        # It seems that we take the last step performed in the epoch as the step metric, while the epoch metric is the average of the steps (already performed by WandB when using .log_dict() ?)

        # Collect the epoch metrics
        epoch_metrics = {}
        for k,v in trainer.callback_metrics.items():
            epoch_metrics[k] = v.item()
            if k not in self.all_keys:
                self.all_keys.append(k)

        self.metrics.append(epoch_metrics)

        # Log the epoch metrics
        if self.verbose >= 1:
            epoch_metrics = trainer.callback_metrics

            # logger.info(f"Considering the metrics: {epoch_metrics.keys()}")
            log_message = ""

            log_message += f"[Epoch {trainer.current_epoch}] Current epoch metrics: \n"
            for k, v in epoch_metrics.items():
                if "epoch" in k:    # only log the metrics that are for an epoch
                    log_message += f"{k}: {v} \n"
            
            log_message += f"\nLearning rate at the end of epoch {trainer.current_epoch}: {pl_model_module.lr_schedulers().get_last_lr()}"    # this yields learning_rate_epoch in the logs
            logger.info(log_message)

        if self.verbose >= 2:
            log_message += f"[Epoch {trainer.current_epoch}] Current step metrics: \n"
            for k, v in epoch_metrics.items():
                if "step" in k:    # only log the metrics that are for a step of the epoch
                    log_message += f"{k}: {v} \n"
            
            logger.info(log_message)

        if self.verbose >= 3:
            # Log the predicted targets and the true targets for training and validation
            logger.info(f"Epoch train predictions for the first and last batch: \n{pl_model_module.train_preds}")
            logger.info(f"Epoch train targets for the first and last batch: \n{pl_model_module.train_targets}")
            logger.info(f"Epoch val predictions for the first and last batch: \n{pl_model_module.val_preds}")
            logger.info(f"Epoch val targets for the first and last batch: \n{pl_model_module.val_targets}")

            if self.data_config.validate_in_and_out_domain:
                logger.info(f"Epoch OOD val predictions for the first and last batch: \n{pl_model_module.gen_val_preds}")
                logger.info(f"Epoch OOD val targets for the first and last batch: \n{pl_model_module.gen_val_targets}")
        
        # Save epoch and steps metrics for plotting
        if len(pl_model_module.train_loss_step) != 0:
            ## STEPS metrics for the epoch
            # Save all the step metrics for this epoch in a file in case they are needed for later combined plotting
            epoch_step_metrics = {
                'epoch': trainer.current_epoch
            }

            epoch_step_metrics.update({
                'train_loss_step': self.tensor_list_to_python_list(pl_model_module.train_loss_step),
                'train_acc_step': self.tensor_list_to_python_list(pl_model_module.train_acc_step),
                'val_loss_step': self.tensor_list_to_python_list(pl_model_module.val_loss_step),
                'val_acc_step': self.tensor_list_to_python_list(pl_model_module.val_acc_step)
            })

            if self.data_config.validate_in_and_out_domain:
                epoch_step_metrics.update({
                    'gen_val_loss_step': self.tensor_list_to_python_list(pl_model_module.gen_val_loss_step),
                    'gen_val_acc_step': self.tensor_list_to_python_list(pl_model_module.gen_val_acc_step)
                })

            if pl_model_module.base_config.data_env in ["REARC", "BEFOREARC"]:
                epoch_step_metrics.update({
                    'train_grid_acc_step': self.tensor_list_to_python_list(pl_model_module.train_grid_acc_step),
                    'val_grid_acc_step': self.tensor_list_to_python_list(pl_model_module.val_grid_acc_step)
                })

                if self.data_config.validate_in_and_out_domain:
                    epoch_step_metrics.update({
                        'gen_val_grid_acc_step': self.tensor_list_to_python_list(pl_model_module.gen_val_grid_acc_step)
                    })
            
            # Add all the step metrics for this epoch to the .jsonl file storing this run's steps metrics
            with open(self.run_metrics_file, 'a') as f:
                f.write(json.dumps(epoch_step_metrics) + "\n")

            ## EPOCH metrics
            self.train_loss_epoch.append(torch.stack(pl_model_module.train_loss_step).mean())
            self.train_acc_epoch.append(torch.stack(pl_model_module.train_acc_step).mean())
            self.val_loss_epoch.append(torch.stack(pl_model_module.val_loss_step).mean())
            self.val_acc_epoch.append(torch.stack(pl_model_module.val_acc_step).mean())

            if self.data_config.validate_in_and_out_domain and len(pl_model_module.gen_val_loss_step) != 0:
                self.gen_val_loss_epoch.append(torch.stack(pl_model_module.gen_val_loss_step).mean())
                self.gen_val_acc_epoch.append(torch.stack(pl_model_module.gen_val_acc_step).mean())

            # Reset the lists for the next epoch
            pl_model_module.train_loss_step = []
            pl_model_module.train_acc_step = []
            pl_model_module.val_loss_step = []
            pl_model_module.val_acc_step = []

            if self.data_config.validate_in_and_out_domain:
                pl_model_module.gen_val_loss_step = []
                pl_model_module.gen_val_acc_step = []

            if pl_model_module.base_config.data_env in ["REARC", "BEFOREARC"]:
                if len(pl_model_module.train_grid_acc_step) != 0:
                    self.train_grid_acc_epoch.append(torch.stack(pl_model_module.train_grid_acc_step).mean())
                    self.val_grid_acc_epoch.append(torch.stack(pl_model_module.val_grid_acc_step).mean())

                    if self.data_config.validate_in_and_out_domain:
                        self.gen_val_grid_acc_epoch.append(torch.stack(pl_model_module.gen_val_grid_acc_step).mean())

                    # Reset the lists for the next epoch
                    pl_model_module.train_grid_acc_step = []
                    pl_model_module.val_grid_acc_step = []

                    if self.data_config.validate_in_and_out_domain:
                        pl_model_module.gen_val_grid_acc_step = []

        # logger.debug(f"Class of the pl model module: {pl_model_module.__class__}")   # e.g.: CVRModel (which inherits from VisReasModel which inherits from pl.LightningModule)
        # logger.debug(f"Attributes of the instance of the pl model module: {pl_model_module.__dict__}")    


def init_callbacks(config, training_folder):

    callbacks = {}

    # Model checkpoint callback
    if config.training.checkpointing.enabled:
        if "loss" in config.training.checkpointing.monitored_metric:
            mode = 'min'
        elif "acc" in config.training.checkpointing.monitored_metric:
            mode = 'max'
        else:
            raise ValueError(f"Unknown monitored metric for model checkpoint: {config.training.checkpointing.monitored_metric}")
        
        model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=training_folder, 
                                                        # filename="best_model",
                                                        save_top_k=1, 
                                                        mode=mode, 
                                                        monitor=f'metrics/{config.training.checkpointing.monitored_metric}', # 'metrics/val_loss' or 'metrics/val_acc'
                                                        every_n_epochs=config.training.checkpointing.ckpt_period, 
                                                        save_last=True
                                                        )

        callbacks['model_checkpoint'] = model_checkpoint

    # Early stopping callback
    if config.training.early_stopping.enabled:
        if "loss" in config.training.early_stopping.monitored_metric:   # handle the case where we have "val_loss" or "gen_val_loss"
            mode = 'min'
        elif "acc" in config.training.early_stopping.monitored_metric:  # handle the case where we have "val_acc" or "gen_val_acc"
            mode = 'max'
        else:
            raise ValueError(f"Unknown monitored metric for early stopping: {config.training.early_stopping.monitored_metric}")
        
        early_stopping = pl.callbacks.EarlyStopping(monitor=f'metrics/{config.training.early_stopping.monitored_metric}', 
                                                    mode=mode, 
                                                    patience=config.training.early_stopping.es_patience, 
                                                    strict=True, 
                                                    verbose=True
                                                    )   # we can use e.g. stopping_threshold=0.995 to stop when the accuracy metric reaches that value
        
        callbacks['early_stopping'] = early_stopping

    # Progress bar callback
    if config.training.progress_bar.enabled:
        progress_bar = TQDMProgressBar(refresh_rate=config.training.progress_bar.refresh_rate, leave=False)
        callbacks['progress_bar'] = progress_bar

    # Metrics callback
    metrics_callback = MetricsCallback(base_config=config.base,
                                       data_config=config.data, 
                                       save_metrics_folder=training_folder, 
                                       verbose=config.training.metrics_callback_verbose
                                       )
    
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

    # Compile the model for improved performance
    # TODO: See logs for issues
    # if os.name == 'posix':  # posix for Linux, 'nt' for Windows
    #     model = torch.compile(model)

    # Training
    trainer = pl.Trainer(default_root_dir=exp_logger.save_dir if exp_logger else None,
                         num_nodes=1,
                         logger=exp_logger, 
                         callbacks=callbacks_list, 
                         max_epochs=config.training.max_epochs,
                         devices=config.base.n_gpus,
                         accelerator='auto',
                         precision=config.training.trainer_precision,
                         gradient_clip_val=1.0,    # TODO: value?
                         num_sanity_val_steps=0, 
                         enable_progress_bar=config.training.progress_bar.enabled,
                         enable_checkpointing=config.training.checkpointing.enabled,
                         log_every_n_steps=config.training.log_every_n_steps,
                        #  accumulate_grad_batches=False,
                        #  gradient_clip_algorithm='norm',
                        #  detect_anomaly=True,
                        #  fast_dev_run=True,
                        #  barebones=True,
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
        model.freeze_backbone_weights()

    # Training callbacks
    callbacks = init_callbacks(config, training_folder)

    # Training
    trainer, callbacks = train(config, model, datamodule, callbacks, exp_logger, ckpt_path)

    # Get best model from the training process
    best_model, best_model_ckpt_path = get_best_model_from_training(model, callbacks)

    # Metrics results
    metrics = callbacks['metrics_callback'].get_all_epoch_metrics()

    log_message = "All epoch training metrics: \n"
    for k, v in metrics.items():
        if "epoch" in k:    # only log the metrics that are for an epoch
            log_message += f"{k}: {v}" + "\n"
    logger.info(log_message)

    best_val_acc = np.nanmax(metrics['metrics/val_acc'] + [0])
    val_acc_best_epoch = (np.nanargmax(metrics['metrics/val_acc'] + [0]) + 1)
    logger.info(f"Best val accuracy: {best_val_acc} at epoch {val_acc_best_epoch}")

    train_results = {
        'metrics': metrics,
        'best_val_acc': best_val_acc,
        'val_acc_best_epoch': val_acc_best_epoch
    }

    if config.data.validate_in_and_out_domain:
        best_gen_val_acc = np.nanmax(metrics['metrics/gen_val_acc_epoch'] + [0])
        gen_val_acc_best_epoch = (np.nanargmax(metrics['metrics/gen_val_acc_epoch'] + [0]) + 1)
        logger.info(f"Best OOD val accuracy: {best_gen_val_acc} at epoch {gen_val_acc_best_epoch}")
        train_results.update({
            'best_gen_val_acc': best_gen_val_acc,
            'gen_val_acc_best_epoch': gen_val_acc_best_epoch
        })

    if config.base.data_env in ["REARC", "BEFOREARC"]:
        best_val_grid_acc = np.nanmax(metrics['metrics/val_acc_grid_epoch'] + [0])
        val_grid_acc_best_epoch = (np.nanargmax(metrics['metrics/val_acc_grid_epoch'] + [0]) + 1)
        logger.info(f"Best val grid accuracy: {best_val_grid_acc} at epoch {val_grid_acc_best_epoch}")
        train_results.update({
            'best_val_grid_acc': best_val_grid_acc,
            'val_grid_acc_best_epoch': val_grid_acc_best_epoch
        })

    # Plot locally some training and validation metrics
    all_local_plotting_metrics = callbacks['metrics_callback'].get_all_local_plotting_metrics()
    
    if config.base.data_env in ["REARC", "BEFOREARC"]:
        fig_paths = plot_rearc_metrics_locally(training_folder, all_local_plotting_metrics)
    
    elif config.base.data_env == "CVR":
        fig_paths = plot_cvr_metrics_locally(training_folder, all_local_plotting_metrics)
    
    if exp_logger:
        for fig_path in fig_paths:
            exp_logger.log_image(key="figures_learning_curves/"+fig_path.replace("./", ""), images=[fig_path])

        # Access the wandb experiment and save the logs for the model hyperparameters and additional results (than those already logged with log_dict() in the model file)
        exp_logger.log_hyperparams(model.hparams)   # TODO: See if need to rewrite the dict? We should only log relevant hyperparameters that we want to consult as a summary of the training performed
        exp_logger.experiment.log(train_results)

    log_message = "*** Training ended ***\n"
    training_elapsed_time = time.time() - training_start_time
    log_message += f"\nTotal training time: \n{training_elapsed_time} seconds ~=\n{training_elapsed_time/60} minutes ~=\n{training_elapsed_time/(60*60)} hours"
    logger.info(log_message)

    return trainer, best_model, best_model_ckpt_path, train_results


if __name__ == '__main__':
    logger.info(f"training.py process ID: {os.getpid()}")

    # Get and log all the config arguments
    config, _ = get_complete_config()
    log_config_dict(config)

    # Seed everything for reproducibility
    if config.base.seed is not None:
        pl.seed_everything(config.base.seed)

    # Create the training folder
    training_folder = f"./{config.data.data_env}/training"
    os.makedirs(training_folder, exist_ok=True)

    # Data chosen
    data_module = vars(data)[config.base.data_module]
    datamodule = data_module(config.data)   # initialize the data with the data config
    logger.info(f"Data module instantiated. Now showing the total number of samples per dataloader:\n{datamodule}\n")

    # Get the image size from the datamodule. Useful for the model backbone
    image_size = datamodule.image_size
    logger.info(f"Image size considered (with padding): {image_size}")

    # Model chosen
    model_module = vars(models)[config.base.model_module]
    model = model_module(config.base, config.model, config.data, config.backbone_network, config.head_network, image_size, training_folder)   # initialize the model with the model and network configs
    logger.trace(f"Model chosen for training: {model}")

    trainer, best_model, best_model_ckpt_path, train_results = main(config, training_folder, datamodule, model, exp_logger=None)
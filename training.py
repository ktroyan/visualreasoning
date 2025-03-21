import os
import time
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, TQDMProgressBar
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Any, Dict, List, Tuple

# Personal codebase dependencies
import data
import models
from utility.utils import get_complete_config, log_config_dict, get_latest_ckpt
from utility.logging import logger

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('medium')
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = 'true'

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
        self.metrics = []  # store metrics for each epoch
        self.all_keys = []  # keep track of all metric keys observed

        # Initialize the lists to store the metrics for all the epochs for local plotting
        self.train_loss_epoch = []
        self.val_loss_epoch = []
        self.train_acc_epoch = []
        self.val_acc_epoch = []
        self.train_grid_acc_epoch = []
        self.val_grid_acc_epoch = []


    def get_all_epoch_metrics(self):
        """
        This method is called after the training (and testing) processes to retrieve all the collected metrics. 
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

    def get_all_local_plotting_metrics(self):
        """
        This method is called after the training process to retrieve all the collected metrics for local plotting. 

        Returns:
            dict: all the collected metrics (train/val per token losses for all steps/epochs, train/val per token/grid accuracies for all steps/epochs) during training for each step and each epoch
        """

        all_local_plotting_metrics = {
            'train_loss_epoch': self.train_loss_epoch,
            'val_loss_epoch': self.val_loss_epoch,
            'train_acc_epoch': self.train_acc_epoch,
            'val_acc_epoch': self.val_acc_epoch,
            'train_grid_acc_epoch': self.train_grid_acc_epoch,
            'val_grid_acc_epoch': self.val_grid_acc_epoch
        }

        return all_local_plotting_metrics


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

        # TODO: What are the step metrics given when using .log_dict() from WandB? Because there are only as many of them as there are epochs.
        # It seems that we take the last step performed in the epoch as the step metric, while the epoch metric is the average of the steps (already performed by WandB when using .log_dict() ?)

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


            # TODO: When would we need to use .mean() ? So far it is equivalent to not taking the mean.
            # log_message = f"[Epoch {trainer.current_epoch}] Mean metrics: \n"
            # for k, v in epoch_metrics.items():
            #     log_message += f"{k}: {v.mean()} \n"

            log_message += f"[Epoch {trainer.current_epoch}] Current epoch metrics: \n"
            for k, v in epoch_metrics.items():
                log_message += f"{k}: {v} \n"
            
            log_message += f"\nLearning rate at the end of epoch {trainer.current_epoch}: {pl_model_module.lr_schedulers().get_last_lr()}"    # this yields learning_rate_epoch in the logs
            logger.info(log_message)

        if self.verbose >= 2:
            # Log the predicted targets and the true targets for training and validation
            logger.info(f"Epoch train predictions for the first and last batch: \n{pl_model_module.train_preds}")
            logger.info(f"Epoch train targets for the first and last batch: \n{pl_model_module.train_targets}")
            logger.info(f"Epoch val predictions for the first and last batch: \n{pl_model_module.val_preds}")
            logger.info(f"Epoch val targets for the first and last batch: \n{pl_model_module.val_targets}")

        
        # Save to later plot locally
        if len(pl_model_module.train_loss_step) != 0:   # TODO: Quick fix to avoid empty lists due to rpoch metrics not being collected for CVR yet.
            self.train_loss_epoch.append(torch.stack(pl_model_module.train_loss_step).mean())
            self.train_acc_epoch.append(torch.stack(pl_model_module.train_acc_step).mean())
            self.train_grid_acc_epoch.append(torch.stack(pl_model_module.train_grid_acc_step).mean())

            self.val_loss_epoch.append(torch.stack(pl_model_module.val_loss_step).mean())
            self.val_acc_epoch.append(torch.stack(pl_model_module.val_acc_step).mean())
            self.val_grid_acc_epoch.append(torch.stack(pl_model_module.val_grid_acc_step).mean())

        # Reset the lists for the next epoch
        pl_model_module.train_loss_step = []
        pl_model_module.train_acc_step = []
        pl_model_module.train_grid_acc_step = []

        pl_model_module.val_loss_step = []
        pl_model_module.val_acc_step = []
        pl_model_module.val_grid_acc_step = []

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

    # Compile the model for improved performance
    # NOTE: without specifying the backend as 'eager', it seems to fail on my NVIDIA RTX 3070 (Laptop) GPU
    # model = torch.compile(model, backend='eager')

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


def plot_metrics_locally(training_folder, metrics):
    """
    Generate and save plots for training and validation epoch metrics.

    Args:
        training_folder (str): Path to save the plots.
        metrics (dict): Dictionary containing metric lists.
    """

    # Create the /figs folder in the folder for training if it does not exist
    figs_folder_path = os.path.join(training_folder, "figs")
    os.makedirs(figs_folder_path, exist_ok=True)

    # Make sure all elements in the values of the dictionary are on cpu
    metrics = {k: [v.cpu().detach().numpy() for v in values] for k, values in metrics.items()}

    # Set consistent style
    sns.set_theme(style="darkgrid", font_scale=1.2)

    # Plot the metrics and save the figure
    def plot_and_save(x, y1, y2, xlabel, ylabel, title, filename, labels=("Train", "Validation")):
        plt.figure(figsize=(8, 5))
        plt.plot(x, y1, label=labels[0], color="b")
        plt.plot(x, y2, label=labels[1], color="g")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(training_folder, "figs", filename))
        plt.close()

    
    ## Epoch-wise plots
    assert len(metrics['train_acc_epoch']) == len(metrics['val_acc_epoch']) == len(metrics['train_loss_epoch']) == len(metrics['val_loss_epoch']) == len(metrics['train_grid_acc_epoch']) == len(metrics['val_grid_acc_epoch'])
    
    epochs = np.arange(len(metrics['val_acc_epoch'])) + 1

    if len(epochs) == 0:
        logger.warning("The plots cannot be created as there are no metrics saved in the list. The epochs list for the x-axis of the plot is empty.")

    # Plot the training and validation loss per epoch
    plot_and_save(
        x=epochs,
        y1=metrics['train_loss_epoch'],
        y2=metrics['val_loss_epoch'],
        xlabel="Epoch", ylabel="Loss",
        title="Training & Validation Loss (Epoch-wise)",
        filename="loss_epoch.png"
    )

    # Plot the training and validation accuracy per epoch
    plot_and_save(
        x=epochs,
        y1=metrics['train_acc_epoch'],
        y2=metrics['val_acc_epoch'],
        xlabel="Epoch", ylabel="Accuracy",
        title="Training & Validation Accuracy (Epoch-wise)",
        filename="acc_epoch.png"
    )

    # Plot the training and validation grid accuracy per epoch
    plot_and_save(
        x=epochs,
        y1=metrics['train_grid_acc_epoch'],
        y2=metrics['val_grid_acc_epoch'],
        xlabel="Epoch", ylabel="Grid Accuracy",
        title="Training & Validation Grid Accuracy (Epoch-wise)",
        filename="grid_acc_epoch.png"
    )

    logger.info(f"Local plots of relevant training metrics saved in: {figs_folder_path}")



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
    metrics = callbacks['metrics_callback'].get_all_epoch_metrics()

    best_val_acc = np.nanmax(metrics['metrics/val_acc'] + [0])
    best_val_epoch = (np.nanargmax(metrics['metrics/val_acc'] + [0]) + 1) * config.training.ckpt_period

    log_message = "All epoch training metrics: \n"
    for k, v in metrics.items():
        log_message += f"{k}: {v}" + "\n"
    logger.info(log_message)

    # Plot locally some training and validation metrics
    all_local_plotting_metrics = callbacks['metrics_callback'].get_all_local_plotting_metrics()
    plot_metrics_locally(training_folder, all_local_plotting_metrics)

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
    config, _ = get_complete_config()
    log_config_dict(config)

    # Seed everything for reproducibility
    if config.base.seed is not None:
        pl.seed_everything(config.base.seed)

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

    # Create the training folder
    training_folder = f"./{config.data.data_env}/training"
    os.makedirs(training_folder, exist_ok=True)

    trainer, best_model, best_model_ckpt_path, train_results = main(config, training_folder, datamodule, model, exp_logger=None)
import os
import numpy as np
import pandas as pd
import wandb
from PIL import Image
import torch
from torchvision import transforms as tvt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import Dict, List

# Personal codebase dependencies
from utility.logging import logger


def compute_dataset_stats(dataset_path):
    # Load the dataset file
    dataset_file_type = dataset_path.split(".")[-1]
    if dataset_file_type == "json":
        dataset_pd = pd.read_json(dataset_path)

        # Flatten each sample in the 'input' column
        sample_inputs = dataset_pd["input"].apply(lambda image: np.array(image).flatten()).values

    elif dataset_file_type == "csv":
        dataset_pd = pd.read_csv(dataset_path)
        
        # Normalize file paths and load images
        dataset_pd['filepath'] = dataset_pd['filepath'].apply(lambda path: os.path.normpath(os.path.join("CVR", path)))
        
        # Load images and convert to tensors
        samples = [tvt.ToTensor()(Image.open(sample_path)) for sample_path in dataset_pd['filepath']]
        
        # Flatten each sample (3-channel image) and store in the 'input' column
        dataset_pd['input'] = [sample.flatten().numpy() for sample in samples]
        
        # Extract flattened samples
        sample_inputs = dataset_pd['input'].values

    else:
        raise Exception("File format not supported. Please provide a csv or json file.")

    # Check if the 'input' column exists
    if "input" not in dataset_pd.columns:
        raise Exception("The dataset does not contain an 'input' column.")

    logger.info(f"Number of samples: {len(sample_inputs)}")

    ## Global stats (i.e., mean and std computed across all pixels in the dataset)
    # NOTE: this is the one to use for normalization of images of the dataset before creating the dataloader
    
    # Flatten all samples into a single array
    all_pixels = np.concatenate(sample_inputs)

    # Compute the dataset mean and std
    dataset_mean_global = np.mean(all_pixels)
    dataset_std_global = np.std(all_pixels)

    # Log the dataset mean and std
    logger.info(f"Dataset mean (global): {dataset_mean_global}")
    logger.info(f"Dataset std (global): {dataset_std_global}")

    ## Per-sample stats (i.e., mean and std computed per sample and averaged across all samples)
    
    # Compute mean and std per sample
    mean_per_sample = [np.mean(sample) for sample in sample_inputs]
    std_per_sample = [np.std(sample) for sample in sample_inputs]

    # Average the mean and std across all samples
    dataset_mean_per_sample = np.mean(mean_per_sample)
    dataset_std_per_sample = np.mean(std_per_sample)

    # Log the dataset mean and std
    logger.info(f"Dataset mean (average per sample): {dataset_mean_per_sample}")
    logger.info(f"Dataset std (average per sample): {dataset_std_per_sample}")

    return dataset_mean_global, dataset_std_global, dataset_mean_per_sample, dataset_std_per_sample

def plot_metrics_locally(training_folder: str, metrics: Dict) -> List[str]:
    """
    Generate and save plots for training and validation epoch metrics.

    Args:
        training_folder (str): Path to save the plots.
        metrics (dict): Dictionary containing metric lists.
    """

    # Store the paths of the figures created
    fig_paths = []

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
        fig_path = os.path.join(training_folder, "figs", filename)
        plt.savefig(fig_path)
        plt.close()
        return fig_path

    ## Epoch-wise plots
    assert len(metrics['train_acc_epoch']) == len(metrics['val_acc_epoch']) == len(metrics['train_loss_epoch']) == len(metrics['val_loss_epoch'])
    
    epochs = np.arange(len(metrics['val_acc_epoch'])) + 1

    if len(epochs) == 0:
        logger.warning("The plots cannot be created as there are no metrics saved in the list. The epochs list for the x-axis of the plot is empty.")

    # Plot the training and validation loss per epoch
    fig_path = plot_and_save(
        x=epochs,
        y1=metrics['train_loss_epoch'],
        y2=metrics['val_loss_epoch'],
        xlabel="Epoch", ylabel="Loss",
        title="Training & Validation Loss (Epoch-wise)",
        filename="loss_epoch.png"
    )

    fig_paths.append(fig_path)

    # Plot the training and validation accuracy per epoch
    fig_path = plot_and_save(
        x=epochs,
        y1=metrics['train_acc_epoch'],
        y2=metrics['val_acc_epoch'],
        xlabel="Epoch", ylabel="Accuracy",
        title="Training & Validation Accuracy (Epoch-wise)",
        filename="acc_epoch.png"
    )

    fig_paths.append(fig_path)

    logger.info(f"Local plots of relevant training metrics saved in: {figs_folder_path}")

    return fig_paths

def observe_image_predictions(split: str, 
                              inputs: torch.Tensor | list, 
                              preds: torch.Tensor | list, 
                              targets: torch.Tensor | list, 
                              n_samples: int = 4, 
                              batch_index: int = 0,
                              epoch: int = None) -> None:
    """ 
    Observe the inputs, predictions and labels of a subset of a batch.

    TODO: Update the function so that it is more adaptable to the number of samples n_samples
    
    Args:
    - split: str: the split of the data (train, val, test, gen_test)
    - inputs: torch.Tensor or list: the input grid images that were fed to the model
    - preds: torch.Tensor or list: the flattened grid image predictions of the model
    - targets: torch.Tensor or list: the flattened grid image targets
    - batch_index: int: the index in the list of the batch to observe
    - n_samples: int: the number of samples to observe from the batch

    """

    # Get a batch of inputs, predictions and targets
    if isinstance(inputs, list) and isinstance(preds, list) and isinstance(targets, list):
        inputs = inputs[batch_index].cpu()
        preds = preds[batch_index]
        targets = targets[batch_index]
    else:
        batch_index = None

    assert len(inputs) == len(preds) == len(targets), "The number of inputs, predictions and targets must be the same."
    assert len(preds) > 0, "There must be at least one input, prediction and target to observe."

    # Get the number of samples to observe
    n_samples = min(n_samples, len(preds))

    if batch_index is not None:
        logger.debug(f"Observing {n_samples} samples from the batch {batch_index} (of the list of batches given) at {split} time. See /figs folder.")
    else:
        logger.debug(f"Observing {n_samples} samples from a batch at {split} time. See /figs folder.")

    def prepare_display_sample(sample_index, x, y_pred, y, axes_row):
        for image_index in range(4):
            x_numpy = x[image_index, :, :, :].cpu().numpy()
            x_numpy = x_numpy.transpose(1, 2, 0)
            x_numpy = np.clip(x_numpy, 0, 1)

            axes_row[image_index].imshow(x_numpy)
            # axes_row[image_index].axis("off")  # hide axes

        # Add label (located over the first image of each sample)
        axes_row[0].set_title(f"Sample {sample_index} --- Pred: {y_pred} | Label (0-3): {y}", fontsize=12, pad=10)


    # Create the figure
    fig, axes = plt.subplots(n_samples, 4, figsize=(12, 3 * n_samples))
    
    if n_samples == 1:
        axes = [axes]  # make it still iterable if only 1 sample

    for i in range(n_samples):
        x = inputs[i].cpu()
        pred_label = preds[i].cpu().item()
        target_label = targets[i].cpu().item()
        prepare_display_sample(i, x, pred_label, target_label, axes[i])

    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    plt.tight_layout()
    plt.show()


    # Save the figure
    wandb_subfolder = "/" + wandb.run.id if wandb.run is not None else ""
    os.makedirs(f"./figs{wandb_subfolder}", exist_ok=True)   # create the /figs folder if it does not exist
    
    # NOTE: "_of_saved_batches" indicates that the index of the batch here is that of the list given
    # as argument where we saved batches, not that of the batch in the dataloader. 
    # Hence, we can refer to the code in rearc_model.py to see what batches are saved to the list. 
    # Most likely we only save the first and the last batch of the epoch.
    if epoch is not None:
        # Training or Validation
        fig.savefig(f"./figs{wandb_subfolder}/{split}_image_predictions_epoch{epoch}_batch{batch_index}_of_saved_batches.png")
    else:
        # Testing
        fig.savefig(f"./figs{wandb_subfolder}/{split}_image_predictions_batch{batch_index}_of_saved_batches.png")

    plt.close(fig)
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import Dict

# Personal codebase dependencies
from utility.logging import logger


def one_hot_encode(x: torch.Tensor, num_token_categories: int) -> torch.Tensor:
    """
    Performs One-Hot Encoding (OHE) of the values of a 3D tensor (batch dimensions and 2D tensor with possible values/tokens: 0, ..., 9, <pad_token>, ..?)
    """
    # TODO: How to OHE special tokens such as padding token? What about extra tokens such as cls and register tokens? Is my approach correct?

    # Convert to one-hot representation
    x_ohe = torch.nn.functional.one_hot(x.long(), num_classes=num_token_categories)  # [B, H, W, C=num_token_categories]

    # Reshape to get the number of possible categorical values, which is used as channels (C), as the first dimension
    x_ohe = x_ohe.permute(0, 3, 1, 2).float()  # [B, C=num_token_categories, H, W] 

    return x_ohe

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

def observe_image_predictions(split: str, 
                              inputs: torch.Tensor | list, 
                              preds: torch.Tensor | list, 
                              targets: torch.Tensor | list, 
                              image_size: int, 
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
    - image_size: int: the H=W size of the image
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

    # Reshape the predictions and targets to the image size: 1D -> 2D
    # NOTE: inputs does not need reshaping as it is already 2D
    preds = preds.view(-1, image_size, image_size)
    targets = targets.view(-1, image_size, image_size)

    assert len(inputs) == len(preds) == len(targets), "The number of inputs, predictions and targets must be the same."
    assert len(preds) > 0, "There must be at least one input, prediction and target to observe."

    # Get the number of samples to observe
    n_samples = min(n_samples, len(preds))

    if batch_index is not None:
        logger.debug(f"Observing {n_samples} samples from the batch {batch_index} (of the list of batches given) at {split} time. See /figs folder.")
    else:
        logger.debug(f"Observing {n_samples} samples from a batch at {split} time. See /figs folder.")

    # Handle padding tokens. Replace the symbols for pad tokens with the background color
    pad_token = 10.0
    background_token = 0.0  # background (typically black color, as in REARC)

    # Count pad tokens and background tokens BEFORE replacement
    input_pad_count = torch.sum(inputs[0] == pad_token).item()
    pred_pad_count = torch.sum(preds[0] == pad_token).item()
    target_pad_count = torch.sum(targets[0] == pad_token).item()

    input_background_count = torch.sum(inputs[0] == background_token).item()
    pred_background_count = torch.sum(preds[0] == background_token).item()
    target_background_count = torch.sum(targets[0] == background_token).item()

    log_message = ""
    log_message += f"Before pad tokens replacement - Pad Tokens: Input={input_pad_count}, Pred={pred_pad_count}, Target={target_pad_count}\n"
    log_message += f"Before pad tokens replacement - Background Tokens: Input={input_background_count}, Pred={pred_background_count}, Target={target_background_count}\n"
    logger.debug(log_message)

    # Replace pad tokens with background token
    inputs[inputs == pad_token] = background_token
    preds[preds == pad_token] = background_token
    targets[targets == pad_token] = background_token

    # Count pad tokens and background tokens AFTER replacement
    input_pad_count_after = torch.sum(inputs[0] == pad_token).item()
    pred_pad_count_after = torch.sum(preds[0] == pad_token).item()
    target_pad_count_after = torch.sum(targets[0] == pad_token).item()

    input_background_count_after = torch.sum(inputs[0] == background_token).item()
    pred_background_count_after = torch.sum(preds[0] == background_token).item()
    target_background_count_after = torch.sum(targets[0] == background_token).item()

    log_message = ""
    log_message += f"After pad tokens replacement - Pad Tokens: Input={input_pad_count_after}, Pred={pred_pad_count_after}, Target={target_pad_count_after}\n"
    log_message += f"After pad tokens replacement - Background Tokens: Input={input_background_count_after}, Pred={pred_background_count_after}, Target={target_background_count_after}\n"
    logger.debug(log_message)
    
    logger.debug(f"inputs shape: {inputs.shape}, preds shape: {preds.shape}, targets shape: {targets.shape}")
    logger.debug(f"inputs dtype: {inputs.dtype}, preds dtype: {preds.dtype}, targets dtype: {targets.dtype}")
    logger.debug(f"inputs min: {inputs.min()}, preds min: {preds.min()}, targets min: {targets.min()}")
    logger.debug(f"inputs max: {inputs.max()}, preds max: {preds.max()}, targets max: {targets.max()}")

    # Use the same color map as REARC
    cmap = ListedColormap([
        '#000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])

    vmin = 0
    vmax = 9

    # norm = Normalize(vmin=0, vmax=9)    # there are 10 possible symbols (0-9) to predict in the grid image
    # args = {'cmap': cmap, 'norm': norm}

    # Create a figure to plot the samples (input, prediction, target) of the batch
    fig, axs = plt.subplots(3, n_samples, figsize=(n_samples*5, 12), dpi=150)

    for i in range(n_samples):
        input_img = inputs[i, :, :].numpy()
        pred_img = preds[i, :, :].cpu().numpy()
        target_img = targets[i, :, :].cpu().numpy()

        for ax, img, title in zip([axs[0, i], axs[1, i], axs[2, i]], 
                                  [input_img, pred_img, target_img], 
                                  [f"Input {i} of batch {batch_index}", f"Prediction {i} of batch {batch_index}", f"Target {i} of batch {batch_index}"]
                                  ):
            sns.heatmap(img, ax=ax, cbar=False, linewidths=0.01, linecolor='gray', square=True, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    if batch_index is not None:
        if epoch is not None:
            fig.suptitle(f"{split} batch {batch_index} at epoch {epoch}", fontsize=16)
        else:
            fig.suptitle(f"{split} batch {batch_index}", fontsize=16)
    else:
        fig.suptitle(f"{split} batch", fontsize=16)
    
    plt.tight_layout()
    # plt.show()

    # Save the figure
    os.makedirs("./figs", exist_ok=True)   # create the /figs folder if it does not exist
    # NOTE: "_of_saved_batches" indicates that the index of the batch here is that of the list given
    # as argument where we saved batches, not that of the batch in the dataloader. 
    # Hence, we can refer to the code in rearc_model.py to see what batches are saved to the list. 
    # Most likely we only save the first and the last batch of the epoch.
    if epoch is not None:
        # Training or Validation
        fig.savefig(f"./figs/{split}_image_predictions_epoch{epoch}_batch{batch_index}_of_saved_batches.png")
    else:
        # Testing
        fig.savefig(f"./figs/{split}_image_predictions_batch{batch_index}_of_saved_batches.png")

    plt.close(fig)

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


def plot_attention_scores(split, train_inputs, attn_scores, layer_index, num_heads, image_size, num_extra_tokens, seq_len, n_samples, epoch, batch_index):
    """
    Plot attention scores for each attention head.
    It handles extra tokens and reshapes the sequence to a 2D grid for better visualization.

    TODO: 
    For each head on a separate row, plot the input grid and the attention map next to each other.
    Create a figure for each sample selected from the batch.
    Use Seaborn for better visualization --> see my function observe_image_predictions for better grid
    Write the symbols in the grid cells.

    Args:
        split (str): Split of the dataset (train, val, test).
        train_inputs (torch.Tensor): Input images of shape [B, C, H, W].
        attn_scores (List): List of attention scores (of shape [B, num_heads, seq_len, seq_len]).
        layer_index (int): Index of the layer from which the attention scores were obtained.
        num_heads (int): Number of attention heads.
        image_size (int): Size of the image (assuming square grid).
        num_extra_tokens (int): Number of extra tokens added before the data sequence.
        seq_len (int): Total sequence length (extra tokens + data tokens).
        n_samples (int): Number of samples to plot.
        epoch (int): Current epoch number.
        batch_index (int): Index of the batch in the dataset.
    """

    # Get batch_index batch
    attn_scores = attn_scores[batch_index]  # list of length num_layers of tensors [B, num_heads, seq_len, seq_len]

    # Get the layer layer_index from the list of layers
    attn_scores = attn_scores[layer_index]  # [B, num_heads, seq_len, seq_len]

    # Get the first n_samples from the batch
    attn_scores = attn_scores.cpu()
    attn_scores = attn_scores[:n_samples]  # [n_samples, num_heads, seq_len, seq_len]

    assert attn_scores.shape == (n_samples, num_heads, seq_len, seq_len), f"Unexpected shape {attn_scores.shape} for attn_scores"


    for sample_index in range(n_samples):
        sample_attn_scores = attn_scores[sample_index]  # [num_heads, seq_len, seq_len]

        # Create figure with subplots for each attention head
        fig, axes = plt.subplots(nrows=num_heads, figsize=(10, num_heads * 5))

        if num_heads == 1:
            axes = [axes]  # Ensure consistent indexing for single head case

        for head in range(num_heads):
            attn = sample_attn_scores[head]  # [seq_len, seq_len]

            if num_extra_tokens > 0:
                # Split extra tokens and image tokens
                extra_attn = attn[:num_extra_tokens, :]  # [num_extra_tokens, seq_len]
                image_attn = attn[num_extra_tokens:, :]  # [image_tokens, seq_len]

                # Reshape the image attention to a 2D grid
                image_attn = image_attn.reshape(image_size, image_size, seq_len)

                # Average attention across tokens to get a [H, W] heatmap. 
                # That is how we choose to consider the attention given to each patch in the image.
                # If we are interested in the attention given by some specific token, we can use its index to get the attention map for that token.
                image_attn = image_attn.mean(dim=2)  # [H, W]

                # Concatenate extra token attention before the image grid
                extra_attn = extra_attn.mean(dim=1, keepdim=True)  # [num_extra_tokens, 1]
                combined_attn = torch.cat([extra_attn.expand(-1, image_size), image_attn], dim=0)  # [num_extra_tokens + H, W]
            else:
                # No extra tokens, just reshape attention into a grid
                combined_attn = attn.reshape(image_size, image_size, seq_len).mean(dim=2)  # [H, W]

            # Convert to numpy for plotting
            combined_attn = combined_attn.detach().numpy()

            # Plot attention map
            im = axes[head].imshow(combined_attn, cmap='hot', interpolation='nearest')
            axes[head].set_title(f'Attention Head {head}')
            axes[head].set_xlabel("Token Index")
            axes[head].set_ylabel("Tokens (Extra + Grid)" if num_extra_tokens > 0 else "Tokens (Grid Only)")
            plt.colorbar(im, ax=axes[head])

        plt.tight_layout()

        # Save the figure instead of displaying it
        os.makedirs("./figs", exist_ok=True)   # create the /figs folder if it does not exist
        plt.savefig(f"./figs/{split}_attention_plot_layer{layer_index}_sample{sample_index}_epoch{epoch}_batch{batch_index}.png")
        plt.close(fig)

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
    pad_token = 10
    background_token = 0  # background (typically black color, as in REARC)

    # Count pad tokens tokens BEFORE replacement
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

    # Replace border tokens and pad tokens with background token
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
        '#000',     # black (background)
        '#0074D9',  # blue
        '#FF4136',  # red
        '#2ECC40',  # green
        '#FFDC00',  # yellow
        '#AAAAAA',  # gray
        '#F012BE',  # pink
        '#FF851B',  # orange
        '#7FDBFF',  # light blue
        '#870C25',   # burgundy
        '#555555',  # dark gray (border tokens)
    ])

    vmin = 0
    vmax = 9 + 1  # 10 possible symbols (0-9) to predict in the grid image + 1 for the borders' color

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

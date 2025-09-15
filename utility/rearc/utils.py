import os
import torch
import numpy as np
import hashlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import Dict, List, Tuple

# Personal codebase dependencies
from utility.custom_logging import logger


def check_train_test_contamination(train_dataloader, test_dataloader):
    log_message = "Checking for data contamination between train and test sets...\n"

    def hash_sample(sample):
        """ Hash the raw bytes of a PyTorch tensor, later ensuring reliable comparison between samples. """
        
        # Ensure tensor is on CPU
        if sample.is_cuda:
            sample = sample.cpu()
        
        # Ensure contiguous memory layout (for tobytes())
        tensor_contiguous = sample.contiguous()
        
        # Convert to NumPy array and get bytes
        tensor_bytes = tensor_contiguous.numpy().tobytes()
        
        # Hash the bytes
        return hashlib.md5(tensor_bytes).hexdigest()

    def get_sample_hashes(dataloader: torch.utils.data.DataLoader) -> Tuple[set, set]:
        x_hashes = set()
        y_hashes = set()

        x_samples = {}
        y_samples = {}

        for i, tuple_of_batches in enumerate(dataloader):  # for each tuple of batches in the dataloader
            x_batch = tuple_of_batches[0]
            y_batch = tuple_of_batches[1]

            for j in range(x_batch.shape[0]): # for each sample in the batch
                x_sample = x_batch[j]
                y_sample = y_batch[j]

                x_hash = hash_sample(x_sample)
                y_hash = hash_sample(y_sample)
                x_hashes.add(x_hash)
                y_hashes.add(y_hash)

                x_samples[x_hash] = x_sample
                y_samples[y_hash] = y_sample

        return x_hashes, y_hashes, x_samples, y_samples

    # Get hashes for each dataset split
    train_x_hashes, train_y_hashes, train_x_samples, train_y_samples = get_sample_hashes(train_dataloader)
    test_x_hashes, test_y_hashes, test_x_samples, test_y_samples = get_sample_hashes(test_dataloader)

    # Print the number of unique hashes in each dataset split
    log_message += f"Number of unique x samples in train set: {len(train_x_hashes)}\n"
    log_message += f"Number of unique y samples in train set: {len(train_y_hashes)}\n"
    log_message += f"Number of unique x samples in test set: {len(test_x_hashes)}\n"
    log_message += f"Number of unique y samples in test set: {len(test_y_hashes)}\n"

    # Compare the train and test hashes to check for contamination
    x_overlap = train_x_hashes.intersection(test_x_hashes)
    y_overlap = train_y_hashes.intersection(test_y_hashes)

    log_message += f"Number of x overlapping samples: {len(x_overlap)}\n"
    log_message += f"Number of y overlapping samples: {len(y_overlap)}\n"

    if len(x_overlap) == 0:
        log_message += "Success! No input data contamination between train and test sets (it seems)."
        logger.info(log_message)
    
    # TODO: Check for when x-y pair is the same as in some settings it is ok to have x train and x test the same as long as the y is different
    
    else:
        log_message += f"Overlapping x samples: {x_overlap}\n"
        log_message += "Warning! Data contamination detected between train and test sets."

        for i, (x_hash, y_hash) in enumerate(zip(x_overlap, y_overlap)):
            train_x_sample = train_x_samples[x_hash]
            train_y_sample = train_y_samples[y_hash]

            test_x_sample = test_x_samples[x_hash]
            test_y_sample = test_y_samples[y_hash]

            log_message = f"{i}-th sample contaminating train-test...\n"
            log_message += f"Shape of the train x sample: {train_x_sample.shape}, train y sample: {train_y_sample.shape}\n"
            log_message += f"Shape of the test x sample: {test_x_sample.shape}, test y sample: {test_y_sample.shape}\n"
            log_message += f"Now plotting the contaminating samples with hash {x_hash} for x and {y_hash} for y.\n"
            logger.warning(log_message)

            plot_grid_image("./data_contamination", train_x_sample, f"train_x_sample_{x_hash}")
            plot_grid_image("./data_contamination", train_y_sample, f"train_y_sample_{y_hash}")

            plot_grid_image("./data_contamination", test_x_sample, f"test_x_sample_{x_hash}")
            plot_grid_image("./data_contamination", test_y_sample, f"test_y_sample_{y_hash}")


def one_hot_encode(x: torch.Tensor, num_token_categories: int) -> torch.Tensor:
    """
    Performs One-Hot Encoding (OHE) of the values of a 3D tensor (batch dimensions and 2D tensor with possible values/tokens: 0, ..., 9, <pad_token>, ..?)
    
    TODO: 
    How to handle special tokens w.r.t. OHE? What about extra tokens such as cls and register tokens?
    Currently, the choice is to consider all of these tokens as well when doing OHE.
    """

    # Convert to one-hot representation
    x_ohe = torch.nn.functional.one_hot(x.long(), num_classes=num_token_categories)  # [B, H, W, C=num_token_categories]

    # Reshape to get the number of possible categorical values, which is used as channels (C), as the first dimension
    x_ohe = x_ohe.permute(0, 3, 1, 2).float()  # [B, C=num_token_categories, H, W] 

    return x_ohe

def plot_attention_scores(save_folder_path: str,
                          split: str,
                          inputs: List, 
                          targets: List,
                          attn_scores: List,
                          layer_index: int,
                          num_heads: int,
                          image_size: int,
                          num_extra_tokens: int, 
                          seq_len: int,
                          n_samples: int,
                          epoch: int,
                          batch_index: int
                          ) -> List[str]:
    """
    Plot attention scores for each attention head.
    It handles extra tokens and reshapes the sequence to a 2D grid for better visualization of attentions w.r.t. input and target grids.

    Args:
        save_folder_path (str): Path to save the plots.
        split (str): Split of the dataset (train, val, test).
        inputs (List): List of input grid images of shape [B, H, W].
        targets (List): List of target grid images of shape [B, H, W].
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

    fig_paths = []

    # Extract batch
    attn_scores = attn_scores[batch_index][layer_index]  # [B, num_heads, seq_len, seq_len]
    attn_scores = attn_scores[:n_samples].cpu()  # [n_samples, num_heads, seq_len, seq_len]
    
    inputs = inputs[batch_index]    # [B, H, W]
    inputs = inputs[:n_samples].cpu()  # [n_samples, H, W]

    targets = targets[batch_index]  # [B, seq_len]
    targets = targets.view(-1, image_size, image_size)  # [B, H, W]
    targets = targets[:n_samples].cpu()   # [n_samples, H, W]

    assert len(attn_scores) == len(inputs) == len(targets), f"Issue in number of samples for: attn_scores ({len(attn_scores)}), inputs ({len(inputs)}), targets ({len(targets)})"
    
    # Get the number of samples to observe
    n_samples = min(n_samples, len(inputs))

    ## Replace padding tokens with background token
    PAD_TOKEN = 10
    X_ENDGRID_TOKEN = 11 
    Y_ENDGRID_TOKEN = 12
    XY_ENDGRID_TOKEN = 13
    NL_GRID_TOKEN = 14
    background_token = 0

    inputs[inputs == PAD_TOKEN] = background_token
    targets[targets == PAD_TOKEN] = background_token
    min_special_token = min(X_ENDGRID_TOKEN, Y_ENDGRID_TOKEN, XY_ENDGRID_TOKEN, NL_GRID_TOKEN)
    inputs[inputs >= min_special_token] -= 1
    targets[targets >= min_special_token] -= 1

    X_ENDGRID_TOKEN -= 1
    Y_ENDGRID_TOKEN -= 1
    XY_ENDGRID_TOKEN -= 1
    NL_GRID_TOKEN -= 1

    ## Setup colormap
    # The pad tokens have been merged to the background tokens and the other special visual tokens are represented by the same color
    cmap = ListedColormap([
        '#000000',  # black (background)
        '#0074D9',  # blue
        '#FF4136',  # red
        '#2ECC40',  # green
        '#FFDC00',  # yellow
        '#AAAAAA',  # gray (border tokens)
        '#F012BE',  # pink
        '#FF851B',  # orange
        '#7FDBFF',  # light blue
        '#870C25',  # burgundy
        '#555555',  # dark gray
    ])
    vmin = 0
    vmax = 9 + 1

    ## Plot for each sample
    for sample_index in range(n_samples):
        sample_attn_scores = attn_scores[sample_index]  # [num_heads, seq_len, seq_len]
        input_img = inputs[sample_index].numpy()  # [H, W]
        target_img = targets[sample_index].numpy()  # [H, W]

        fig, axes = plt.subplots(nrows=num_heads,
                                 ncols=3,
                                 figsize=(16, 5 * num_heads),
                                 gridspec_kw={'width_ratios': [1, 1.2, 1]},  # make second column (attention map) 1.2 times wider
                                 dpi=150
                                 )

        if num_heads == 1:
            axes = np.expand_dims(axes, axis=0)  # ensure consistent indexing

        for head in range(num_heads):
            attn = sample_attn_scores[head]  # [seq_len, seq_len]

            ## Column 1: Input Grid
            ax_input = axes[head, 0]
            sns.heatmap(input_img, 
                        ax=ax_input, 
                        cbar=False, 
                        square=True,
                        cmap=cmap, 
                        linewidths=0.01, 
                        linecolor='gray',
                        xticklabels=False, 
                        yticklabels=False,
                        vmin=vmin,
                        vmax=vmax
                        )
            ax_input.set_title("Input Grid", fontdict={'fontsize': 12, 'fontweight': 'bold', 'family': 'serif'})

            ## Column 2: Attention Map
            ax_attn = axes[head, 1]

            if num_extra_tokens > 0:
                attn_image_tokens = attn[num_extra_tokens:, num_extra_tokens:]
            else:
                attn_image_tokens = attn

            grid_attn = attn_image_tokens.reshape(image_size, image_size, -1).mean(dim=2).detach().numpy()
            labels = input_img.reshape(image_size, image_size)

            sns.heatmap(grid_attn,
                        ax=ax_attn,
                        cmap="YlOrRd", 
                        square=True,
                        annot=labels, 
                        fmt='d', 
                        annot_kws={"fontsize": 9},
                        linewidths=0.01, 
                        linecolor='gray'
                        )
            
            colorbar = ax_attn.collections[0].colorbar
            colorbar.ax.yaxis.offsetText.set_visible(False) # to hide the scientific notation at the top of the colorbar
            
            ax_attn.set_title(f"Attention Map (Head {head})", fontdict={'fontsize': 12, 'fontweight': 'bold', 'family': 'serif'})
            ax_attn.set_xticks([])
            ax_attn.set_yticks([])

            ## Column 3: Target Grid
            ax_target = axes[head, 2]
            sns.heatmap(target_img, 
                        ax=ax_target, 
                        cbar=False, 
                        square=True,
                        cmap=cmap, 
                        linewidths=0.01, 
                        linecolor='gray',
                        xticklabels=False, 
                        yticklabels=False,
                        vmin=vmin,
                        vmax=vmax
                        )
            ax_target.set_title("Target Grid", fontdict={'fontsize': 12, 'fontweight': 'bold', 'family': 'serif'})

        plt.tight_layout()

        # Create the /figs folder
        figs_folder_path = os.path.join(save_folder_path, "figs")
        os.makedirs(figs_folder_path, exist_ok=True)

        fig_path = f"{figs_folder_path}/{split}_attention_layer{layer_index}_sample{sample_index}_epoch{epoch}_batch{batch_index}.png"
        plt.savefig(fig_path)
        plt.close(fig)
        fig_paths.append(fig_path)
        logger.debug(f"Attention scores for epoch {epoch}, batch {batch_index}, sample {sample_index}, saved in: {fig_path}")

    return fig_paths

def plot_metrics_locally(save_folder_path: str, metrics: Dict) -> List[str]:
    """
    Generate and save plots for training and validation epoch metrics.

    Args:
        save_folder_path (str): Path to save the plots.
        metrics (dict): Dictionary containing metric lists.
    """

    # Store the paths of the figures created
    fig_paths = []

    # Create the /figs folder in the folder for training if it does not exist
    figs_folder_path = os.path.join(save_folder_path, "figs")
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
        fig_path = os.path.join(figs_folder_path, filename)
        plt.savefig(fig_path)
        plt.close()
        return fig_path

    
    ## ID 
    # Epoch-wise plots
    assert len(metrics['train_acc_epoch']) == len(metrics['val_acc_epoch']) == len(metrics['train_loss_epoch']) == len(metrics['val_loss_epoch']) == len(metrics['train_grid_acc_epoch']) == len(metrics['val_grid_acc_epoch'])
    
    epochs = np.arange(len(metrics['val_acc_epoch'])) + 1

    if len(epochs) == 0:
        logger.warning("The plots cannot be created as there are no metrics saved in the list. The epochs list for the x-axis of the plot is empty.")

    # Plot the training and validation loss per epoch
    fig_path = plot_and_save(x=epochs,
                  y1=metrics['train_loss_epoch'],
                  y2=metrics['val_loss_epoch'],
                  xlabel="Epoch", ylabel="Loss",
                  title="Training & Validation Loss (Epoch-wise)",
                  filename="loss_epoch.png"
                  )
    
    fig_paths.append(fig_path)

    # Plot the training and validation accuracy per epoch
    fig_path = plot_and_save(x=epochs,
                  y1=metrics['train_acc_epoch'],
                  y2=metrics['val_acc_epoch'],
                  xlabel="Epoch", 
                  ylabel="Accuracy",
                  title="Training & Validation Accuracy (Epoch-wise)",
                  filename="acc_epoch.png"
                  )
    
    fig_paths.append(fig_path)

    # Plot the training and validation grid accuracy per epoch
    fig_path = plot_and_save(x=epochs,
                  y1=metrics['train_grid_acc_epoch'],
                  y2=metrics['val_grid_acc_epoch'],
                  xlabel="Epoch", 
                  ylabel="Grid Accuracy",
                  title="Training & Validation Grid Accuracy (Epoch-wise)",
                  filename="grid_acc_epoch.png"
                  )
    
    fig_paths.append(fig_path)


    ## OOD
    if 'gen_val_loss_epoch' in metrics:
        # Plot the training loss and OOD validation loss per epoch
        fig_path = plot_and_save(x=epochs,
                    y1=metrics['train_loss_epoch'],
                    y2=metrics['gen_val_loss_epoch'],
                    xlabel="Epoch", ylabel="Loss",
                    title="Training & OOD Validation Loss (Epoch-wise)",
                    filename="ood_loss_epoch.png"
                    )
        
        fig_paths.append(fig_path)

    if 'gen_val_acc_epoch' in metrics:
        # Plot the training and OOD validation accuracy per epoch
        fig_path = plot_and_save(x=epochs,
                    y1=metrics['train_acc_epoch'],
                    y2=metrics['gen_val_acc_epoch'],
                    xlabel="Epoch", 
                    ylabel="Accuracy",
                    title="Training & OOD Validation Accuracy (Epoch-wise)",
                    filename="ood_acc_epoch.png"
                    )
    
        fig_paths.append(fig_path)

    if 'gen_val_grid_acc_epoch' in metrics:
        # Plot the training and OOD validation grid accuracy per epoch
        fig_path = plot_and_save(x=epochs,
                    y1=metrics['train_grid_acc_epoch'],
                    y2=metrics['gen_val_grid_acc_epoch'],
                    xlabel="Epoch", 
                    ylabel="Grid Accuracy",
                    title="Training & OOD Validation Grid Accuracy (Epoch-wise)",
                    filename="ood_grid_acc_epoch.png"
                    )
    
        fig_paths.append(fig_path)

    logger.info(f"Local plots of relevant training metrics saved in: {figs_folder_path}")

    return fig_paths

def plot_image_predictions(save_folder_path: str,
                           split: str,
                           inputs: torch.Tensor | list,
                           preds: torch.Tensor | list,
                           targets: torch.Tensor | list,
                           image_size: int,
                           n_samples: int = 4,
                           batch_index: int = 0,
                           epoch: int = None
                           ) -> List[str]:
    """ 
    Observe the inputs, predictions and labels of a subset of a batch.
    """

    # Store the paths of the figures created
    fig_paths = []

    # Get a batch of inputs, predictions and targets
    if isinstance(inputs, list) and isinstance(preds, list) and isinstance(targets, list):
        inputs = inputs[batch_index].cpu()
        preds = preds[batch_index].cpu()
        targets = targets[batch_index].cpu()
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
        logger.debug(f"Observing {n_samples} samples from the batch {batch_index} (of the list of batches given) at {split} time.")
    else:
        logger.debug(f"Observing {n_samples} samples from a batch at {split} time.")

    # Explicit the symbols chosen for the tokens
    PAD_TOKEN = 10
    X_ENDGRID_TOKEN = 11 
    Y_ENDGRID_TOKEN = 12
    XY_ENDGRID_TOKEN = 13
    NL_GRID_TOKEN = 14
    background_token = 0  # background (typically black color, as in REARC)

    # Log grid tokens info before any token is replaced for plotting
    log_message = "Grid tokens info BEFORE any token replacement:\n"
    log_message += f"inputs dtype: {inputs.dtype}, preds dtype: {preds.dtype}, targets dtype: {targets.dtype}\n"
    log_message += f"inputs shape: {inputs.shape}, preds shape: {preds.shape}, targets shape: {targets.shape}\n"
    log_message += f"inputs min: {inputs.min()}, preds min: {preds.min()}, targets min: {targets.min()}\n"
    log_message += f"inputs max: {inputs.max()}, preds max: {preds.max()}, targets max: {targets.max()}\n"
    logger.debug(log_message)

    replace_pad_tokens = True  # whether to replace pad tokens with background token (0)
    if replace_pad_tokens:
        # Count pad tokens tokens BEFORE replacement
        input_pad_count = torch.sum(inputs[0] == PAD_TOKEN).item()
        pred_pad_count = torch.sum(preds[0] == PAD_TOKEN).item()
        target_pad_count = torch.sum(targets[0] == PAD_TOKEN).item()

        input_background_count = torch.sum(inputs[0] == background_token).item()
        pred_background_count = torch.sum(preds[0] == background_token).item()
        target_background_count = torch.sum(targets[0] == background_token).item()

        log_message = ""
        log_message += f"Before pad tokens replacement - Pad Tokens: Input={input_pad_count}, Pred={pred_pad_count}, Target={target_pad_count}\n"
        log_message += f"Before pad tokens replacement - Background Tokens: Input={input_background_count}, Pred={pred_background_count}, Target={target_background_count}\n"

        # Replace pad tokens with background token
        inputs[inputs == PAD_TOKEN] = background_token
        preds[preds == PAD_TOKEN] = background_token
        targets[targets == PAD_TOKEN] = background_token

        # Count pad tokens and background tokens AFTER replacement
        input_pad_count_after = torch.sum(inputs[0] == PAD_TOKEN).item()
        pred_pad_count_after = torch.sum(preds[0] == PAD_TOKEN).item()
        target_pad_count_after = torch.sum(targets[0] == PAD_TOKEN).item()

        input_background_count_after = torch.sum(inputs[0] == background_token).item()
        pred_background_count_after = torch.sum(preds[0] == background_token).item()
        target_background_count_after = torch.sum(targets[0] == background_token).item()

        log_message += f"After pad tokens replacement - Pad Tokens: Input={input_pad_count_after}, Pred={pred_pad_count_after}, Target={target_pad_count_after}\n"
        log_message += f"After pad tokens replacement - Background Tokens: Input={input_background_count_after}, Pred={pred_background_count_after}, Target={target_background_count_after}\n"
        logger.debug(log_message)

        # Since pad tokens are replaced with background token, we need to adjust the symbols for the other special tokens
        # Shift down by 1 the special tokens (X_ENDGRID_TOKEN, Y_ENDGRID_TOKEN, XY_ENDGRID_TOKEN, NL_GRID_TOKEN)
        min_special_token = min(X_ENDGRID_TOKEN, Y_ENDGRID_TOKEN, XY_ENDGRID_TOKEN, NL_GRID_TOKEN)
        inputs[inputs >= min_special_token] -= 1
        preds[preds >= min_special_token] -= 1
        targets[targets >= min_special_token] -= 1

        X_ENDGRID_TOKEN = X_ENDGRID_TOKEN - 1
        Y_ENDGRID_TOKEN = Y_ENDGRID_TOKEN - 1
        XY_ENDGRID_TOKEN = XY_ENDGRID_TOKEN - 1
        NL_GRID_TOKEN = NL_GRID_TOKEN - 1


    replace_border_tokens = False  # whether to replace border tokens with background token (0)
    if replace_border_tokens:
        # Replace border tokens with background token
        inputs[inputs == X_ENDGRID_TOKEN] = background_token
        preds[preds == X_ENDGRID_TOKEN] = background_token
        targets[targets == X_ENDGRID_TOKEN] = background_token
        inputs[inputs == Y_ENDGRID_TOKEN] = background_token
        preds[preds == Y_ENDGRID_TOKEN] = background_token
        targets[targets == Y_ENDGRID_TOKEN] = background_token
        inputs[inputs == XY_ENDGRID_TOKEN] = background_token
        preds[preds == XY_ENDGRID_TOKEN] = background_token
        targets[targets == XY_ENDGRID_TOKEN] = background_token

    replace_newline_tokens = False  # whether to replace newline tokens with background token (0)
    if replace_newline_tokens:
        # Replace newline tokens with background token
        inputs[inputs == NL_GRID_TOKEN] = background_token
        preds[preds == NL_GRID_TOKEN] = background_token
        targets[targets == NL_GRID_TOKEN] = background_token

    # Log grid tokens info after some tokens may have been replaced for plotting
    log_message = "Grid tokens info AFTER any token replacement:\n"
    log_message += f"inputs dtype: {inputs.dtype}, preds dtype: {preds.dtype}, targets dtype: {targets.dtype}\n"
    log_message += f"inputs shape: {inputs.shape}, preds shape: {preds.shape}, targets shape: {targets.shape}\n"
    log_message += f"inputs min: {inputs.min()}, preds min: {preds.min()}, targets min: {targets.min()}\n"
    log_message += f"inputs max: {inputs.max()}, preds max: {preds.max()}, targets max: {targets.max()}\n"
    logger.debug(log_message)

    # Decide on the colors for the different tokens
    # Choose one:
    no_merge_of_special_tokens = False
    merge_border_tokens = False
    merge_all_special_tokens = True

    if merge_all_special_tokens:
        # Merge all special tokens (border, newline) into one color (gray)
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
            '#870C25',  # burgundy
            '#555555',  # dark gray (all the special visual tokens, so border (3) + newline (1) tokens)
        ])

        vmin = 0
        vmax = 9 + 1
    
    if merge_border_tokens:
        # Replace tokens Y_ENDGRID_TOKEN, XY_ENDGRID_TOKEN with X_ENDGRID_TOKEN
        # Newline token has to take the next lowest index, so NL_GRID_TOKEN
        inputs[inputs == Y_ENDGRID_TOKEN] = X_ENDGRID_TOKEN
        preds[preds == Y_ENDGRID_TOKEN] = X_ENDGRID_TOKEN
        targets[targets == Y_ENDGRID_TOKEN] = X_ENDGRID_TOKEN
        inputs[inputs == XY_ENDGRID_TOKEN] = X_ENDGRID_TOKEN
        preds[preds == XY_ENDGRID_TOKEN] = X_ENDGRID_TOKEN
        targets[targets == XY_ENDGRID_TOKEN] = X_ENDGRID_TOKEN
        inputs[inputs == NL_GRID_TOKEN] = X_ENDGRID_TOKEN + 1
        preds[preds == NL_GRID_TOKEN] = X_ENDGRID_TOKEN + 1
        targets[targets == NL_GRID_TOKEN] = X_ENDGRID_TOKEN + 1

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
            '#870C25',  # burgundy
            '#555555',  # dark gray (border (3) + newline (1) tokens)
            '#9D00FF',  # purple (newline tokens)
        ])

        vmin = 0
        vmax = 9 + 1 + 1    # 10 possible symbols (0-9) to predict in the grid image + 1 for the borders' color + 1 for the newline tokens' colors

    if no_merge_of_special_tokens:
        # No merging of special tokens, each token has its own color
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
            '#870C25',  # burgundy
            '#FF00AA',  # fuchsia (border token X)
            '#9D00FF',  # purple (border token Y)
            '#FF00FF',  # magenta (border token XY)
            '#555555',  # dark gray (newline tokens)
        ])

        vmin = 0
        vmax = 9 + 3 + 1


    # Log how many different symbols are in the grid images
    log_message = "Grid images info:\n"
    log_message += f"inputs dtype: {inputs.dtype}, preds dtype: {preds.dtype}, targets dtype: {targets.dtype}\n"
    log_message += f"inputs shape: {inputs.shape}, preds shape: {preds.shape}, targets shape: {targets.shape}\n"
    log_message += f"inputs min: {inputs.min()}, preds min: {preds.min()}, targets min: {targets.min()}\n"
    log_message += f"inputs max: {inputs.max()}, preds max: {preds.max()}, targets max: {targets.max()}\n"
    log_message += f"inputs unique values: {torch.unique(inputs)}, preds unique values: {torch.unique(preds)}, targets unique values: {torch.unique(targets)}\n"
    logger.debug(log_message)

    # Create a figure to plot the samples (input, prediction, target) of the batch
    fig, axs = plt.subplots(nrows=n_samples,
                            ncols=3, 
                            figsize=(15, n_samples * 4),
                            dpi=150,
                            gridspec_kw={'width_ratios': [1, 1, 1]},
                            squeeze=False,  # to ensure axs is always a 2D array
                            )

    for i in range(n_samples):
        input_img = inputs[i, :, :].numpy()
        pred_img = preds[i, :, :].numpy()
        target_img = targets[i, :, :].numpy()

        for ax, img, title in zip([axs[i, 0], axs[i, 1], axs[i, 2]], 
                                  [input_img, pred_img, target_img], 
                                  [f"Input {i} of batch {batch_index}", f"Prediction {i} of batch {batch_index}", f"Target {i} of batch {batch_index}"]
                                  ):
            sns.heatmap(img, ax=ax, cbar=False, linewidths=0.01, linecolor='gray', square=True, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontdict={'fontsize': 16, 'fontweight': 'bold', 'family': 'serif'})
            ax.set_xticks([])
            ax.set_yticks([])

    if batch_index is not None:
        if epoch is not None:
            title = f"{split} batch {batch_index} at epoch {epoch}"
        else:
            title = f"{split} batch {batch_index}"
    else:
        title = f"{split} batch"

    fig.suptitle(title, family='serif', fontsize=22, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # rect=[left, bottom, right, top]; adjust the layout to make room for the title
    # plt.show()

    # Save the figure
    figs_folder = os.path.join(save_folder_path, "figs")
    os.makedirs(figs_folder, exist_ok=True)   # create the /figs folder if it does not exist

    # NOTE: "_of_saved_batches" indicates that the index of the batch here is that of the list given
    # as argument where we saved batches, not that of the batch in the dataloader. 
    # Hence, we can refer to the code in rearc_model.py to see what batches are saved to the list. 
    # Most likely we only save the first and the last batch of the epoch.
    if epoch is not None:
        # Training or Validation
        fig_path = f"{figs_folder}/{split}_image_predictions_epoch{epoch}_batch{batch_index}_of_saved_batches.png"
        fig.savefig(fig_path)
        fig_paths.append(fig_path)
    else:
        # Testing
        fig_path = f"{figs_folder}/{split}_image_predictions_batch{batch_index}_of_saved_batches.png"
        fig.savefig(fig_path)
        fig_paths.append(fig_path)

    plt.close(fig)

    return fig_paths

def plot_grid_image(figs_folder, grid_image, fig_name="grid_image"):
    """ 
    Plot a grid image from REARC.
    """

    if not isinstance(grid_image, torch.Tensor):
        grid_image = torch.tensor(grid_image)
    
    # Explicit the symbols chosen for the tokens
    PAD_TOKEN = 10
    X_ENDGRID_TOKEN = 11 
    Y_ENDGRID_TOKEN = 12
    XY_ENDGRID_TOKEN = 13
    NL_GRID_TOKEN = 14
    background_token = 0  # background (typically black color, as in REARC)

    replace_pad_tokens = True  # whether to replace pad tokens with background token (0)
    if replace_pad_tokens:
        # Count pad tokens tokens BEFORE replacement
        input_pad_count = torch.sum(grid_image == PAD_TOKEN).item()

        input_background_count = torch.sum(grid_image == background_token).item()

        # Replace pad tokens with background token
        grid_image[grid_image == PAD_TOKEN] = background_token

        # Count pad tokens and background tokens AFTER replacement
        input_pad_count_after = torch.sum(grid_image == PAD_TOKEN).item()

        input_background_count_after = torch.sum(grid_image == background_token).item()

        # Since pad tokens are replaced with background token, we need to adjust the symbols for the other special tokens
        # Shift down by 1 the special tokens (X_ENDGRID_TOKEN, Y_ENDGRID_TOKEN, XY_ENDGRID_TOKEN, NL_GRID_TOKEN)
        min_special_token = min(X_ENDGRID_TOKEN, Y_ENDGRID_TOKEN, XY_ENDGRID_TOKEN, NL_GRID_TOKEN)
        grid_image[grid_image >= min_special_token] -= 1

        X_ENDGRID_TOKEN = X_ENDGRID_TOKEN - 1
        Y_ENDGRID_TOKEN = Y_ENDGRID_TOKEN - 1
        XY_ENDGRID_TOKEN = XY_ENDGRID_TOKEN - 1
        NL_GRID_TOKEN = NL_GRID_TOKEN - 1


    replace_border_tokens = False  # whether to replace border tokens with background token (0)
    if replace_border_tokens:
        # Replace border tokens with background token
        grid_image[grid_image == X_ENDGRID_TOKEN] = background_token
        grid_image[grid_image == Y_ENDGRID_TOKEN] = background_token
        grid_image[grid_image == XY_ENDGRID_TOKEN] = background_token

    replace_newline_tokens = False  # whether to replace newline tokens with background token (0)
    if replace_newline_tokens:
        # Replace newline tokens with background token
        grid_image[grid_image == NL_GRID_TOKEN] = background_token

    # Log grid tokens info after some tokens may have been replaced for plotting
    log_message = "Grid tokens info AFTER any token replacement:\n"
    log_message += f"inputs dtype: {grid_image.dtype}\n"
    log_message += f"grid_image shape: {grid_image.shape}\n"
    log_message += f"grid_image min: {grid_image.min()}\n"
    log_message += f"grid_image max: {grid_image.max()}\n"
    logger.debug(log_message)

    # Decide on the colors for the different tokens
    # Choose one:
    no_merge_of_special_tokens = False
    merge_border_tokens = True
    merge_all_special_tokens = False

    if merge_all_special_tokens:
        # Merge all special tokens (border, newline) into one color (gray)
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
            '#870C25',  # burgundy
            '#555555',  # dark gray (all the special visual tokens, so border (3) + newline (1) tokens)
        ])

        vmin = 0
        vmax = 9 + 1
    
    if merge_border_tokens:
        # Replace tokens Y_ENDGRID_TOKEN, XY_ENDGRID_TOKEN with X_ENDGRID_TOKEN
        # Newline token has to take the next lowest index, so NL_GRID_TOKEN
        grid_image[grid_image == Y_ENDGRID_TOKEN] = X_ENDGRID_TOKEN
        grid_image[grid_image == XY_ENDGRID_TOKEN] = X_ENDGRID_TOKEN
        grid_image[grid_image == NL_GRID_TOKEN] = X_ENDGRID_TOKEN + 1

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
            '#870C25',  # burgundy
            '#555555',  # dark gray (border (3) + newline (1) tokens)
            '#9D00FF',  # purple (newline tokens)
        ])

        vmin = 0
        vmax = 9 + 1 + 1    # 10 possible symbols (0-9) to predict in the grid image + 1 for the borders' color + 1 for the newline tokens' colors

    if no_merge_of_special_tokens:
        # No merging of special tokens, each token has its own color
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
            '#870C25',  # burgundy
            '#FF00AA',  # fuchsia (border token X)
            '#9D00FF',  # purple (border token Y)
            '#FF00FF',  # magenta (border token XY)
            '#555555',  # dark gray (newline tokens)
        ])

        vmin = 0
        vmax = 9 + 3 + 1


    # Log how many different symbols are in the grid images
    log_message = "Grid images info:\n"
    log_message += f"inputs dtype: {grid_image.dtype}\n"
    log_message += f"grid_image shape: {grid_image.shape}\n"
    log_message += f"grid_image min: {grid_image.min()}\n"
    log_message += f"grid_image max: {grid_image.max()}\n"
    log_message += f"grid_image unique values: {torch.unique(grid_image)}\n"
    logger.debug(log_message)

    # Create a figure to plot the samples (input, prediction, target) of the batch
    fig, _ = plt.subplots(nrows=1,
                            ncols=1,
                            figsize=(15, 4),
                            dpi=150,
                            )
    title = "Grid Image"
    sns.heatmap(grid_image, cbar=False, linewidths=0.01, linecolor='gray', square=True, cmap=cmap, vmin=vmin, vmax=vmax)

    fig.suptitle(title, family='serif', fontsize=22, fontweight='bold')
    
    plt.tight_layout()
    # plt.show()

    # Save the figure
    figs_folder = os.path.join(figs_folder, "figs")
    os.makedirs(figs_folder, exist_ok=True)

    fig_path = f"{figs_folder}/{fig_name}.png"
    fig.savefig(fig_path)

    plt.close(fig)

def observe_rearc_input_output_images(save_folder_path, dataloader, split, batch_id=0, n_samples=4):
    """ 
    Observe the input and output images of a batch from the dataloader.
    """
    
    # Get the batch batch_id from the dataloader
    for i, batch in enumerate(dataloader):
        if i == batch_id:
            break
    
    # Get the input and output images
    inputs, outputs = batch[0], batch[1]

    # Number of samples to observe
    n_samples = min(n_samples, len(inputs))

    logger.debug(f"Observing {n_samples} samples from {split} batch {batch_id}.")

    # Handle padding tokens. Replace the symbols for pad tokens with the background color
    pad_token = 10
    inputs[inputs == pad_token] = 0
    outputs[outputs == pad_token] = 0

    # Use the same color map as REARC
    cmap = ListedColormap([
        '#000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
    
    vmin = 0
    vmax = 9

    # Create a figure to plot the samples
    fig, axs = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6), dpi=150)

    for i in range(n_samples):
        input_img = inputs[i].cpu().numpy()
        target_img = outputs[i].cpu().numpy()

        for ax, img, title in zip([axs[0, i], axs[1, i]], 
                                  [input_img, target_img], 
                                  [f"Input {i} of batch {batch_id}", f"Output {i} of batch {batch_id}"]
                                  ):
            sns.heatmap(img, ax=ax, cbar=False, linewidths=0.05, linecolor='gray', square=True, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"{split} batch {batch_id}", fontsize=18)

    plt.tight_layout()
    # plt.show()

    # Save the figure
    fig.savefig(f"{save_folder_path}/{split}_image_input_output_batch{batch_id}.png")

    plt.close(fig)
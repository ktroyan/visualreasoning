import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from typing import Dict

# Personal codebase dependencies
from utility.logging import logger

def observe_image_predictions(preds, labels, n_samples=4):
    """ Observe the predictions and labels of a subset of a batch of images """

    assert len(preds) == len(labels), "The number of predictions and labels must be the same."
    assert len(preds) > 0, "There must be at least one prediction and label to observe."

    # Get the number of samples to observe
    n_samples = min(n_samples, len(preds))

    # Use the same color map as REARC
    cmap = ListedColormap([
        '#000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
    norm = Normalize(vmin=0, vmax=9)    # there are 10 possible symbols
    args = {'cmap': cmap, 'norm': norm}

    # Create a figure to plot the samples
    fig, axs = plt.subplots(2, n_samples, figsize=(n_samples*5, 10))

    for i in range(n_samples):
        # Plot the prediction
        axs[0, i].imshow(preds[i].cpu().numpy(), **args)
        axs[0, i].set_title("Prediction")
        axs[0, i].axis('off')

        # Plot the label
        axs[1, i].imshow(labels[i].cpu().numpy(), **args)
        axs[1, i].set_title("Label")
        axs[1, i].axis('off')

        fig.suptitle("Working?", fontsize=16)

    plt.tight_layout()
    plt.show()
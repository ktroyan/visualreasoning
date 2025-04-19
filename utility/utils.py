import os
import time
import yaml
import glob
import datetime
import torch
from omegaconf import OmegaConf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import shutil
from typing import Any, Dict, List, Tuple

# Personal codebase dependencies
from utility.logging import logger


def get_complete_config(sweep_config=None):
    """
    Load and merge all relevant configuration files using OmegaConf.
    If enabled, the WandB sweep config arguments take priority over the default config arguments.
    The CLI arguments (provided with OmegaConf syntax) take priority (i.e., overwrite any other parameter value).
    Essentially, the merging order in OmegaConf.merge() is important.

    NOTE: 
    The merging order of the configs is important in order to get expected and correct parameter values.
    That is why the CLI config is merged last and the sweep config right before it.
    
    Moreover, we should proceed to the creation of the complete configs in several steps in order to 
    get the correct specific config files to load that may depend on some arguments of other configs.
    
    For example base.model_module decides on what model config to load.
    For example model.backbone decides on what backbone network config to load.
    For example model.head decides on what head network config to load.

    Returns:
        resolved_complete_config_oc (OmegaConf): The complete resolved configuration as an OmegaConf object.
        resolved_complete_config_dict (dict): The complete resolved configuration as a dict.
    """

    def merge_and_resolve_configs(configs: list) -> OmegaConf:
        # Merge configs
        merged_config = OmegaConf.merge(*configs)

        # Explicitly resolve interpolations in the main config so that we get the values needed to load the specific configs below
        OmegaConf.resolve(merged_config)

        return merged_config

    try:
        # Load non-specific configs
        base_config = OmegaConf.load("configs/base.yaml")
        data_config = OmegaConf.load("configs/data.yaml")
        experiment_config = OmegaConf.load("configs/experiment.yaml")
        wandb_config = OmegaConf.load("configs/wandb.yaml")
        training_config = OmegaConf.load("configs/training.yaml")
        inference_config = OmegaConf.load("configs/inference.yaml")
        cli_config = OmegaConf.from_cli()

        # Handle sweep config
        if sweep_config is None:
            sweep_config = {}
        else:
            log_config_dict(sweep_config, "*** Parameter values selected from the sweep config ***")

        # Create resolvers
        def resolve_if_then_else(enabled, set_value):
            return set_value if enabled else None  # return None when disabled
        
        def resolve_if_then_else_sysgen(study_name, set_value):
            if study_name == "sys-gen":
                return set_value
            else:
                return False

        def resolve_data_env_img_size(data_env, img_size):
            if data_env == "CVR":
                return 128
            elif data_env == "REARC":
                return img_size

        # Register the resolvers. Use replace=True to not try to re-register the resolver which would raise an error during WandB sweeps
        OmegaConf.register_new_resolver("resolve_if_then_else", resolve_if_then_else, replace=True)
        OmegaConf.register_new_resolver("resolve_if_then_else_sysgen", resolve_if_then_else_sysgen, replace=True)
        OmegaConf.register_new_resolver("resolve_data_env_img_size", resolve_data_env_img_size, replace=True)

        # Merge all the non-specific configs into a single hierarchical object
        main_config = merge_and_resolve_configs([base_config, data_config, experiment_config, wandb_config, training_config, inference_config, sweep_config, cli_config])

        # Load model-specific config
        model_module = main_config.base.get("model_module", None)
        if not model_module:
            raise ValueError("Error: 'base.model_module' is missing in base.yaml")

        model_config_path = f"configs/models/{model_module}.yaml"
        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f"Error: Model config file '{model_config_path}' not found.")
        model_config = OmegaConf.load(model_config_path)

        # Merge the main config with the specific config into a single hierarchical object
        main_config = merge_and_resolve_configs([main_config, model_config, sweep_config, cli_config])
        
        # Load backbone network-specific config
        backbone_network_name = main_config.model.get("backbone", None)
        if not backbone_network_name:
            raise ValueError("Error: 'model.backbone' is missing in model config.")

        backbone_network_config_path = f"configs/networks/backbones/{backbone_network_name}.yaml"
        if not os.path.exists(backbone_network_config_path):
            raise FileNotFoundError(f"Error: Backbone network config file '{backbone_network_config_path}' not found.")
        backbone_network_config = OmegaConf.load(backbone_network_config_path)

        # Load head network-specific config
        head_network_name = main_config.model.get("head", None)
        if not head_network_name:
            raise ValueError("Error: 'model.head' is missing in model config.")

        head_network_config_path = f"configs/networks/heads/{head_network_name}.yaml"
        if not os.path.exists(head_network_config_path):
            raise FileNotFoundError(f"Error: Head network config file '{head_network_config_path}' not found.")
        head_network_config = OmegaConf.load(head_network_config_path)

        # Merge all configs into a single hierarchical object
        main_config = merge_and_resolve_configs([main_config, backbone_network_config, head_network_config, sweep_config, cli_config])

        # Convert the complete config to a dict
        resolved_complete_config_dict = OmegaConf.to_container(main_config)

        # Convert dict to OmegaConf object so that we can then use dot notation to access the keys
        resolved_complete_config_oc = OmegaConf.create(resolved_complete_config_dict)


    except Exception as e:
        raise RuntimeError(f"Error loading or merging configurations: {e}")
    
    return resolved_complete_config_oc, resolved_complete_config_dict

def log_config_dict(config_dict, log_message=""):
    """
    Log the config dictionary in a structured format.

    Args:
        config_dict (dict): The configuration dictionary to log.
        log_message (str): An optional message to prepend to the log.
    """
    dict_logs_message = OmegaConf.to_yaml(config_dict)
    logger.info(f"{log_message}\n\n{dict_logs_message}")

def get_config_specific_args_from_args_dict(args_dict, specific_config_path):
    # Load the YAML config to get config-specific keys
    with open(specific_config_path, "r") as f:
        specific_config_keys = yaml.safe_load(f).keys()

    # Filter args to include only the arguments of the specific config file given
    specific_config_args = {key: value for key, value in args_dict.items() if key in specific_config_keys}

    return specific_config_args

def save_config(cfg, path):
    with open(path, 'w') as cfg_file:
        yaml.dump(cfg, cfg_file)

def generate_timestamped_experiment_name(exp_basename):
    now = datetime.datetime.now()
    timestamp = now.strftime("%d_%m_%H_%M")
    experiment_name = f"{exp_basename}_{timestamp}"
    return experiment_name

def save_model_metadata_for_ckpt(save_folder, model):
    """
    Save the model class and all the config arguments used to create the model and save it in the current (timestamped) experiment folder.
    This allows us to load the model module/class and config arguments in order to be able to easily load a model from a checkpoint only.

    Args:
        save_folder (str): the folder in which to save the metadata .pth (dict) file
        model (pl.LightningModule): the model to save metadata for
    """

    metadata_for_ckpt = {
    "model_module_name": model.__class__.__name__,
    "hparams": model.hparams    # in fact not necessary to load the hparams as they are already saved in the checkpoint
    }

    torch.save(metadata_for_ckpt, os.path.join(save_folder, "metadata_for_ckpt.pth"))
    logger.info("Model metadata saved for future checkpoint use.")

def find_most_recent_experiment_folder(directory):
    # Search for the the most recent "experiment_" folder in the given directory
    folders = [os.path.join(directory, d) for d in os.listdir(directory) if (os.path.isdir(os.path.join(directory, d)) and "experiment_" in d)]
    if not folders:
        return None

    most_recent_folder = max(folders, key=os.path.getmtime)
    return most_recent_folder

def get_latest_ckpt(directory):
    """ Get the latest checkpoint file in the latest folder following the 

    Args:
        folder str: folder in which to search for the latest checkpoint file

    Returns:
        str: path of the latest checkpoint file
    """

    most_recent_folder = find_most_recent_experiment_folder(directory)

    # Search for the most recent .ckpt file in the found folder
    ckpt_pattern = os.path.join(most_recent_folder, "*.ckpt")
    ckpt_files = glob.glob(ckpt_pattern)
    if not ckpt_files:
        return None

    latest_ckpt = max(ckpt_files, key=os.path.getmtime)
    return latest_ckpt

def get_model_from_ckpt(model_ckpt_path):
    import models   # import the models module here to avoid circular imports

    # Get the model class and hparams arguments from the metadata file
    metadata_path = os.path.join(os.path.dirname(model_ckpt_path), "metadata_for_ckpt.pth")
    metadata_for_ckpt = torch.load(metadata_path, weights_only=False)

    model_module_name = metadata_for_ckpt["model_module_name"]
    # hparams = metadata_for_ckpt["hparams"]    # in fact not necessary to load the hparams as they are already saved in the checkpoint

    # Load the model module
    model_module = vars(models)[model_module_name]
    # model = model_module(**hparams)    # in fact not necessary to load the hparams as they are already saved in the checkpoint

    # Load the model
    model = model_module.load_from_checkpoint(model_ckpt_path)

    return model

def plot_lr_schedule(lr_values: List) -> str:
    plt.figure(figsize=(10, 5))
    plt.plot(lr_values, label="Learning Rate")
    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid()
    fig_path = "./figs/learning_rate_schedule.png"
    plt.savefig(fig_path)
    # plt.show()
    plt.close()
    return fig_path

def plot_absolute_positional_embeddings(pos_embed, num_prefix_tokens=None, viz_as_heatmap=False):
    """ 
    Plot the absolute positional embeddings (APE).

    Args:
        pos_embed (torch.Tensor): Positional embedding tensor of shape [1, seq_len(+num_extra_tokens), embed_dim]
        num_prefix_tokens (int): Number of extra/prefixed tokens (e.g., cls, register tokens)
        viz_as_heatmap (bool): If True, plot a heatmap; otherwise, plot line plots per embedding dimension
    """
    os.makedirs('./figs', exist_ok=True)

    # Remove extra/prefix tokens if needed
    if num_prefix_tokens is not None and num_prefix_tokens > 0:
        embeddings = pos_embed[0, num_prefix_tokens:, :].detach().cpu().numpy()  # [seq_len, embed_dim]
    else:
        embeddings = pos_embed[0, :, :].detach().cpu().numpy()

    plt.figure(figsize=(10, 5))

    if viz_as_heatmap:
        # Each row is a position in the sequence, each column is a dimension in the embedding
        ims = plt.imshow(embeddings, aspect='auto', cmap='viridis')
        plt.colorbar(ims)
        plt.xlabel("Embedding dimension")
        plt.ylabel("Sequence position")
        plt.title("APE")

    else:
        # Plot all embedding dimensions for each position in the sequence
        for i in range(embeddings.shape[1]):  # for each embedding dim
            plt.plot(embeddings[:, i], label=f"Dim {i}", alpha=0.5)  # dim-wise trace across sequence

        plt.xlabel("Sequence position")
        plt.ylabel("Embedding value")
        plt.title("APE Line Plot (dim traces)")
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), ncol=1, fontsize='small', frameon=False)

    # plt.tight_layout()
    plt.savefig('./figs/positional_embeddings.png')
    plt.close()

def timer_decorator(func):
    """"
    Decorator to measure the execution time of a function.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)  # call the function decorated
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"{func.__name__} took {elapsed_time:.4f} seconds to execute.")
        return result
    return wrapper

def delete_folder_content(folder_path):
    shutil.rmtree(folder_path)  
    os.makedirs(folder_path)

def copy_folder(source_folder, destination_folder):
    # Ensure destination folder exists
    os.makedirs(destination_folder, exist_ok=True)
    
    # Copy all contents from source to destination
    for item in os.listdir(source_folder):
        source_path = os.path.join(source_folder, item)
        destination_path = os.path.join(destination_folder, item)

        if os.path.isdir(source_path):
            # Copy subdirectories recursively
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)  
        else:
            # Copy files
            shutil.copy2(source_path, destination_path)


def observe_input_output_images(dataloader, batch_id=0, n_samples=4, split="test"):
    """ 
    Observe the input and output images of a batch from the dataloader.
    
    TODO: It only works for REARC. Update it or create another function for CVR as well if needed.
    """
    
    # Get the batch batch_id from the dataloader
    for i, batch in enumerate(dataloader):
        if i == batch_id:
            break
    
    # Get the input and output images
    inputs, outputs = batch[0], batch[1]

    # Number of samples to observe
    n_samples = min(n_samples, len(inputs))

    logger.debug(f"Observing {n_samples} samples from {split} batch {batch_id}. See /figs folder.")

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
    os.makedirs("./figs", exist_ok=True)   # create the /figs folder if it does not exist
    fig.savefig(f"./figs/{split}_image_input_output_batch{batch_id}.png")

    plt.close(fig)


def process_test_results(config, test_results, test_type="test", exp_logger=None):
    """
    Process the test results and log them.

    TODO:
    Check if code ok and improve handling and display of the results, especially for multi-task experiments.
        
    FIXME:
    For this code to work currently, the batch size should yield a number of elements in each key of results so that it is a multiple of the number of tasks, otherwise the reshape will fail.
    Also see how to handle the case where the results are for multiple test dataloaders
    """

    test_set_path = f"{config.data.dataset_dir}/{test_type}"

    # Check if the folder config.data.dataset_dir contains test_set_path with .csv or .json extension and load it accordingly with pandas
    if os.path.exists(f"{test_set_path}.csv"):
        test_set = pd.read_csv(f"{test_set_path}.csv")
    elif os.path.exists(f"{test_set_path}.json"):
        test_set = pd.read_json(f"{test_set_path}.json")
    else:
        raise FileNotFoundError(f"Test set file not found in the dataset directory: {test_set_path}")

    # Processing of the results and wrap-up of the parameters used
    if config.base.data_env == "REARC":
        tasks_considered = test_set['task'].unique()
    
    elif config.base.data_env == "BEFOREARC":
        # FIXME: Fix issue (e.g., when running for BEFOREARC Compositionality Setting 5 Experiment 1)
        unique_transformations = test_set['transformations'].drop_duplicates().tolist() # TODO: the field 'transformations' contains a list of transformations applied to obtain the output grid form the input grid?
        tasks_considered = ["-".join(transformation_list) for transformation_list in unique_transformations]
    
    elif config.base.data_env == "CVR":
        tasks_considered = test_set['task'].unique()

    logger.debug(f"Post-processing results for tasks: {tasks_considered}")

    global_avg = {metric_key: r.mean() for metric_key, r in test_results.items()}
    per_task = {}
    per_task_avg = {}

    # nb_of_tasks = len(tasks_considered)

    for metric_key, r in test_results.items():
        logger.debug(f"Processing results for metric key: {metric_key}, and result with shape: {r.shape}")
        k_result = r.reshape([len(tasks_considered), -1])
        for i, task in enumerate(tasks_considered):
            if 'task' not in task:
                per_task[f'{metric_key}_task_{task}'] = k_result[i]
                per_task_avg[f'{metric_key}_task_{task}'] = k_result[i].mean()
                
            else:
                per_task[f'{metric_key}_{task}'] = k_result[i]
                per_task_avg[f'{metric_key}_{task}'] = k_result[i].mean()
    
    logger.info(f"Global average results:\n{global_avg}")
    
    log_message = "Per task results for each batch/step:\n"
    for key, value in per_task.items():
        log_message += f"{key}: {value}\n"
    logger.info(log_message)
    
    logger.info(f"Per task average results:\n{per_task_avg}")

    if exp_logger is not None:
        exp_logger.experiment.log({f"{test_type}_results_global_avg/{k}": v for k, v in global_avg.items()})
        exp_logger.experiment.log({f"{test_type}_results_per_task_avg/{k}": v for k, v in per_task_avg.items()})
        # exp_logger.experiment.log({f"{test_type}_results_per_task/{k}": v for k, v in per_task.items()})   # TODO: check how to to log it properly if needed
        exp_logger.save()

    processed_test_results = {
        f"{test_type}_results_global_avg": global_avg,
        f"{test_type}_results_per_task": per_task,
        f"{test_type}_results_per_task_avg": per_task_avg
    }

    return processed_test_results
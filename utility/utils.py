import os
import time
import yaml
import glob
import datetime
import torch
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import shutil

# Personal codebase dependencies
from utility.logging import logger


def get_complete_config():
    """
    Load and merge all relevant configuration files using OmegaConf.

    Returns:
        resolved_complete_config_oc (OmegaConf): The complete resolved configuration as an OmegaConf object.
        resolved_complete_config_dict (dict): The complete resolved configuration as a dict.
    """
    try:
        # Load base configurations
        base_config = OmegaConf.load("configs/base.yaml")
        data_config = OmegaConf.load("configs/data.yaml")
        experiment_config = OmegaConf.load("configs/experiment.yaml")
        wandb_config = OmegaConf.load("configs/wandb.yaml")
        training_config = OmegaConf.load("configs/training.yaml")
        inference_config = OmegaConf.load("configs/inference.yaml")
        cli_config = OmegaConf.from_cli()

        # Merge common configurations
        main_config = OmegaConf.merge(base_config,
                                      data_config,
                                      experiment_config,
                                      wandb_config,
                                      training_config,
                                      inference_config,
                                      cli_config
                                      )

        # Create resolvers
        def resolve_if_then_else(enabled, set_value):
            return set_value if enabled else None  # return None when disabled

        def resolve_data_env_img_size(data_env, img_size):
            if data_env == "CVR":
                return 128
            elif data_env == "REARC":
                return img_size

        # Register the resolvers
        OmegaConf.register_new_resolver("resolve_if_then_else", resolve_if_then_else)
        OmegaConf.register_new_resolver("resolve_data_env_img_size", resolve_data_env_img_size)

        # Explicitly resolve interpolations in the main config so that we get the values needed to load the specific configs below
        OmegaConf.resolve(main_config)

        # Load model-specific config
        model_module = main_config.base.get("model_module", None)
        if not model_module:
            raise ValueError("Error: 'base.model_module' is missing in base.yaml")

        model_config_path = f"configs/models/{model_module}.yaml"
        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f"Error: Model config file '{model_config_path}' not found.")
        model_config = OmegaConf.load(model_config_path)

        # Load backbone network-specific config
        backbone_network_name = model_config.model.get("backbone", None)
        if not backbone_network_name:
            raise ValueError("Error: 'model.backbone' is missing in model config.")

        backbone_network_config_path = f"configs/networks/backbones/{backbone_network_name}.yaml"
        if not os.path.exists(backbone_network_config_path):
            raise FileNotFoundError(f"Error: Backbone network config file '{backbone_network_config_path}' not found.")
        backbone_network_config = OmegaConf.load(backbone_network_config_path)

        # Load head network-specific config
        head_network_name = model_config.model.get("head", None)
        if not head_network_name:
            raise ValueError("Error: 'model.head' is missing in model config.")

        head_network_config_path = f"configs/networks/heads/{head_network_name}.yaml"
        if not os.path.exists(head_network_config_path):
            raise FileNotFoundError(f"Error: Head network config file '{head_network_config_path}' not found.")
        head_network_config = OmegaConf.load(head_network_config_path)

        # Merge all configs into a single hierarchical object
        complete_config = OmegaConf.merge(main_config,
                                          model_config,
                                          backbone_network_config,
                                          head_network_config,
                                          cli_config
                                          )

        # Explicitly resolve value interpolations in the complete config
        OmegaConf.resolve(complete_config)

        # Convert the config to a dict
        resolved_complete_config_dict = OmegaConf.to_container(complete_config)

        # Update the complete config dict with the WandB sweep config 
        # TODO: See how to use the sweep config to update the complete config dict
        #       Probably something like: the sweep config values for this run are resolved, and then we update the complete config dict with these resolved values

        # Convert dict to OmegaConf object so that we can then use dot notation to access the keys
        resolved_complete_config_oc = OmegaConf.create(resolved_complete_config_dict)

    except Exception as e:
        raise RuntimeError(f"Error loading or merging configurations: {e}")
    
    return resolved_complete_config_oc, resolved_complete_config_dict

def log_config_dict(config_dict, log_message=""):
    """
    Logs the config dictionary in a structured format.

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

def plot_lr_schedule(lr_values):
    plt.figure(figsize=(10, 5))
    plt.plot(lr_values, label="Learning Rate")
    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid()
    plt.savefig("./figs/learning_rate_schedule.png")
    # plt.show()
    plt.close()

def plot_absolute_positional_embeddings(pos_embed, num_prefix_tokens=None, viz_as_heatmap=False):
    """ 
    Plot the absolute positional embeddings (APE) used.
    If needed, we can truncate the first num_prefix_tokens tokens from the embeddings plot.
    TODO: Do we need to truncate the embeddings part for the prefix tokens from the plot?
    """
    # Ensure the figs directory exists
    os.makedirs('./figs', exist_ok=True)

    # Truncate the first num_prefix_tokens tokens from the embeddings plot if needed and convert embeddings to numpy
    if num_prefix_tokens is not None:
        embeddings = pos_embed[0, num_prefix_tokens:, :].detach().cpu().numpy()
    else:
        embeddings = pos_embed[0, :, :].detach().cpu().numpy()

    plt.figure(figsize=(10, 5))

    if viz_as_heatmap:
        ims = plt.imshow(embeddings, aspect='auto', label="Absolute Positional Embeddings")
        plt.colorbar(ims)
    else:
        plt.plot(embeddings, label="Absolute Positional Embeddings")
    
    plt.xlabel("Embedding position")
    plt.ylabel("Sequence position")
    plt.title("Positional Embeddings")
    plt.savefig('./figs/positional_embeddings.png')
    # plt.show()
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
        logger.warning(f"{func.__name__} took {elapsed_time:.4f} seconds to execute.")
        return result
    return wrapper

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
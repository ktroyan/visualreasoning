import os
import time
import glob
import datetime
import torch
from omegaconf import OmegaConf
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from typing import Any, Dict, List, Tuple

# Personal codebase dependencies
from utility.logging import logger


def get_complete_config(sweep_config: Dict = None) -> Tuple[OmegaConf, Dict]:
    """
    Load and merge all relevant configuration files using OmegaConf.

    If enabled, the WandB sweep config arguments take priority over the default config arguments.
    The CLI arguments (provided with OmegaConf syntax) take priority (i.e., overwrite any other parameter value).

    Therefore, the merging order of the configs in OmegaConf.merge() is important in order to get expected and correct parameter values.
    That is why the CLI config is merged last and the sweep config right before it.
    
    Moreover, we should proceed with the creation of the complete configs in several steps in order to
    get the correct specific config files to load that may depend on some arguments of other configs.
    
    For example base.model_module decides on what model config to load.
    For example model.backbone decides on what backbone network config to load.
    For example model.head decides on what head network config to load.

    Args:
        sweep_config (dict, optional): The WandB sweep config dictionary. If None, there is no sweep config.

    Returns:
        resolved_complete_config_oc (OmegaConf): The complete resolved configuration as an OmegaConf object.
        resolved_complete_config_dict (dict): The complete resolved configuration as a dict.
    """

    def merge_and_resolve_configs(configs: list) -> OmegaConf:
        # Merge configs
        merged_config = OmegaConf.merge(*configs)

        # Explicitly resolve variable interpolations in the main config so that we get the values needed to load the subsequent specific configs
        OmegaConf.resolve(merged_config)

        return merged_config

    try:
        ## Load non-specific configs
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

        ## Create resolvers.
        # They are used to resolve some parameters' value based on the values of other parameters through variable interpolation.

        # This resolver yields a cleaner config overview of the parameters actually used in the experiment since it sets the value to None when the parameter is not enabled.
        def resolve_if_then_else(enabled, set_value):
            return set_value if enabled else None  # return None when parameter is not enabled

        # This resolver is used to handle specific cases for the sys-gen study
        def resolve_if_then_else_sysgen(study_name, set_value):
            if study_name == "sys-gen":
                return set_value
            else:
                return False

        # This resolver is used to handle specific cases for the compositionality study
        def resolve_if_then_else_compositionality(study_name, set_value):
            if study_name == "compositionality":
                return set_value
            else:
                return False

        # This resolver is used to use an OOD test set only when applicable
        def resolve_use_gen_test_set(study, data_env, bool_value):
            if study in ["sys-gen", "compositionality"] and data_env == "BEFOREARC":
                return bool_value
            elif study in ["sys-gen"] and data_env in ["REARC", "CVR"]:
                return bool_value
            else:
                return False

        # This resolver is used to use an OOD validation set only when applicable
        def resolve_validate_in_and_out_domain(study, data_env, bool_value):
            if study in ["sys-gen", "compositionality"] and data_env == "BEFOREARC":
                return bool_value
            else:
                return False

        # This resolver is used to check if an OOD validation set is used, in which case the monitored metric should be the one of the OOD validation set
        def resolve_if_then_else_validate_in_and_out_domain(validate_in_and_out_domain, ood_monitored_metric, default_monitored_metric):
            if validate_in_and_out_domain:
                return ood_monitored_metric # e.g. gen_val_loss, gen_val_acc
            else:
                return default_monitored_metric # e.g. val_loss, val_acc

        # This resolver is used to set the image size based on the data environment.
        # For CVR, the image size is fixed to 128.
        # For -ARC, it is set to the value provided in the config. If null, the value will be computed as the max. image size within the dataset.
        def resolve_data_env_img_size(data_env, img_size):
            if data_env == "CVR":
                return 128
            elif data_env in ["REARC", "BEFOREARC"]:
                return img_size

        # Register the resolvers. Use replace=True to not try to register twice the resolver which would raise an error during WandB sweeps
        OmegaConf.register_new_resolver("resolve_if_then_else", resolve_if_then_else, replace=True)
        OmegaConf.register_new_resolver("resolve_if_then_else_sysgen", resolve_if_then_else_sysgen, replace=True)
        OmegaConf.register_new_resolver("resolve_if_then_else_compositionality", resolve_if_then_else_compositionality, replace=True)
        OmegaConf.register_new_resolver("resolve_use_gen_test_set", resolve_use_gen_test_set, replace=True)
        OmegaConf.register_new_resolver("resolve_if_then_else_validate_in_and_out_domain", resolve_if_then_else_validate_in_and_out_domain, replace=True)
        OmegaConf.register_new_resolver("resolve_validate_in_and_out_domain", resolve_validate_in_and_out_domain, replace=True)
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
        
        # Load backbone/encoder network-specific config
        backbone_network_name = main_config.model.get("backbone", None)
        if not backbone_network_name:
            raise ValueError("Error: 'model.backbone' is missing in model config.")

        backbone_network_config_path = f"configs/networks/backbones/{backbone_network_name}.yaml"
        if not os.path.exists(backbone_network_config_path):
            raise FileNotFoundError(f"Error: Backbone network config file '{backbone_network_config_path}' not found.")
        backbone_network_config = OmegaConf.load(backbone_network_config_path)

        # Load head/decoder network-specific config
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

        # Convert the config dict to an OmegaConf object so that we can then use dot notation to access the keys
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

def generate_timestamped_experiment_name(exp_basename):
    now = datetime.datetime.now()
    timestamp = now.strftime("%d_%m_%H_%M_%S")
    experiment_name = f"{exp_basename}_{timestamp}"
    return experiment_name

def save_model_metadata_for_ckpt(save_folder_path, model):
    """
    Save the model class and all the config arguments used to create the model and save it in the current (timestamped) experiment folder.
    This allows us to load the model module/class and config arguments in order to be able to easily load a model from a checkpoint only.

    Args:
        save_folder_path (str): the folder in which to save the metadata .pth (dict) file
        model (pl.LightningModule): the model to save metadata for
    """

    metadata_for_ckpt = {
    "model_module_name": model.__class__.__name__,
    "hparams": model.hparams    # in fact, not necessary to load the hparams as they are already saved in the checkpoint
    }

    torch.save(metadata_for_ckpt, os.path.join(save_folder_path, "metadata_for_ckpt.pth"))
    logger.info("Model metadata saved for possible future checkpoint use.")

def find_most_recent_experiment_folder(directory):
    """ Search for the the most recent "experiment_" folder in the given directory. """

    folders = [os.path.join(directory, d) for d in os.listdir(directory) if (os.path.isdir(os.path.join(directory, d)) and "experiment_" in d)]
    if not folders:
        return None

    most_recent_folder = max(folders, key=os.path.getmtime)
    return most_recent_folder

def get_latest_ckpt(directory):
    """ Get the latest checkpoint file within all the experiment folders in the given directory. """

    most_recent_folder = find_most_recent_experiment_folder(directory)

    # Search for the most recent .ckpt file in the found folder
    ckpt_pattern = os.path.join(most_recent_folder, "*.ckpt")
    ckpt_files = glob.glob(ckpt_pattern)
    if not ckpt_files:
        return None

    latest_ckpt = max(ckpt_files, key=os.path.getmtime)
    return latest_ckpt

def get_model_from_ckpt(model_ckpt_path):
    """ Load a (pre-trained) model from a checkpoint file. """

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

def plot_lr_schedule(save_folder_path: str, lr_values: List) -> str:
    """ Plot the learning rate schedule performed during training. """

    # Create a general /figs folder
    figs_path = os.path.join(save_folder_path, "figs")
    os.makedirs(figs_path, exist_ok=True)
    fig_path = os.path.join(figs_path, "learning_rate_schedule.png")

    # Plot the learning rate schedule
    plt.figure(figsize=(10, 5))
    plt.plot(lr_values, label="Learning Rate")
    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid()
    plt.savefig(fig_path)
    # plt.show()
    plt.close()
    return fig_path

def plot_absolute_positional_embeddings(pos_embed, num_prepended_tokens=0, num_appended_tokens=0, viz_as_heatmap=False, sample_index_in_batch=None):
    """ 
    Plot the absolute positional embeddings (APE) for a sample sequence.

    Args:
        pos_embed (torch.Tensor): Positional embedding tensor of shape [1, seq_len(+num_prepended_tokens + num_appended_tokens), embed_dim]
        num_prepended_tokens (int): Number of extra prepended tokens (e.g., cls, register tokens)
        num_appended_tokens (int): Number of extra appended tokens (e.g., task tokens, in-context example)
        viz_as_heatmap (bool): If True, plot a heatmap; otherwise, plot line plots per embedding dimension
        sample_index_in_batch (int): Index of the sample in the batch for which to visualize the dynamic APE
    """

    if sample_index_in_batch is not None:   # visualize the dynamic APE for a sample of the batch
        pos_embed = pos_embed[sample_index_in_batch, :, :].detach().cpu().numpy()    # [pe_length, embed_dim] <-- [B, pe_length, embed_dim]
    else:   # visualize the static APE (same for any sample)
        pos_embed = pos_embed[0, :, :].detach().cpu().numpy()    # [pe_length, embed_dim] <-- [1, pe_length, embed_dim]

    # Remove extra prepended tokens for visualization
    if num_prepended_tokens > 0:
        pos_embed = pos_embed[num_prepended_tokens:, :]

    # Remove extra appended tokens for visualization
    if num_appended_tokens > 0:
        pos_embed = pos_embed[:-num_appended_tokens, :]

    plt.figure(figsize=(10, 5))

    if viz_as_heatmap:
        # Each row is a position in the sequence, each column is a dimension in the embedding
        ims = plt.imshow(pos_embed, aspect='auto', cmap='viridis')
        plt.colorbar(ims)
        plt.xlabel("Embedding dimension")
        plt.ylabel("Sequence position")
        plt.title("APE")

    else:
        # Plot each embedding dimensions for all positions in the sequence
        for i in range(pos_embed.shape[1]):  # for each embedding dim
            plt.plot(pos_embed[:, i], label=f"Dim {i}", alpha=0.5)  # dim-wise trace across sequence

        plt.xlabel("Sequence position")
        plt.ylabel("Embedding value")
        plt.title("APE Line Plot (dim traces)")
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), ncol=1, fontsize='small', frameon=False)

    # plt.tight_layout()
    save_folder_path = "./figs"
    os.makedirs(save_folder_path, exist_ok=True)
    plt.savefig(f'{save_folder_path}/absolute_positional_embeddings.png')
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

def process_test_results(config, test_results, test_type="test", exp_logger=None):
    """
    Process the test results and log them.

    TODO / FIXME:
    Handle and display of the results, especially for multi-task experiments.
    See how to handle the case where the results are for multiple test dataloaders
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

    return None

def get_paper_model_name(config):
    """ Prepare model name for logging """
    model_name = None

    if config.model.backbone == "vit":
        if (
            (not config.model.visual_tokens.enabled) and
            (config.model.ape.enabled) and
            (config.model.ape.ape_type == "learn") and
            (not config.model.rpe.enabled) and
            (not config.model.ope.enabled) and
            (config.model.num_reg_tokens == 0)
        ):
            model_name = "ViT-vanilla"
        
        # elif (
        #     (config.model.visual_tokens.enabled) and 
        #     (config.model.ape.enabled) and 
        #     (config.model.ape.ape_type == "2dsincos") and
        #     (config.model.ape.mixer != "sum") and
        #     (config.model.ope.enabled) and
        #     (config.model.rpe.enabled) and 
        #     ("Alibi" in config.model.rpe.rpe_type)
        # ):
        #     model_name = "ViT-vitarc"
        
        # elif (
        #     (config.model.visual_tokens.enabled) and 
        #     (config.model.ape.enabled) and 
        #     (config.model.ape.ape_type == "2dsincos") and
        #     (config.model.rpe.enabled) and
        #     (config.model.rpe.rpe_type == "rope")
        # ):
        #     model_name = "ViT"

        elif (
            (config.model.visual_tokens.enabled) and 
            (config.model.ape.enabled) and 
            (config.model.ape.ape_type == "2dsincos") and
            (config.model.ape.mixer != "sum") and
            (config.model.ope.enabled) and
            (config.model.rpe.enabled) and
            (config.model.rpe.rpe_type == "rope") and
            (config.model.num_reg_tokens > 0)
        ):
            model_name = "ViT"

    elif (
        (config.model.backbone == "resnet") and
        (not config.model.visual_tokens.enabled)
    ):
        model_name = "ResNet"
    
    elif config.model.backbone == "diffusion_vit":
        model_name = "ViT-diffusion"
    
    elif config.model.backbone == "looped_vit":
        model_name = "ViT-looped"
    
    if model_name is None:
        logger.warning(f"Model backbone was not recognized. Simply using the backbone network name ({config.model.backbone}) from the config for the model name.")
        model_name = config.model.backbone

    return model_name
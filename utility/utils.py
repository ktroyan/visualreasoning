import os
import sys
import yaml
import argparse
from pathlib import Path
import glob
import datetime
import matplotlib.pyplot as plt
from typing import Dict, Any

# Personal codebase dependencies
from utility.logging import logger

def update_args_from_yaml_config(args: argparse.Namespace, config_file: str) -> argparse.Namespace:
    """Update args Namespace from a YAML config file path."""

    if Path(config_file).exists():
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file) or {}
            for key, value in config.items():
                # Only update if the argument is not already set via CLI
                if not hasattr(args, key):
                    setattr(args, key, value)
    else:
        raise FileNotFoundError(f"Config file {config_file} not found.")

    return args

def update_args_from_yaml_configs(args: argparse.Namespace, config_files: list) -> argparse.Namespace:
    """Update args Namespace from a list of YAML config file paths."""

    for config_file in config_files:
        args = update_args_from_yaml_config(args, config_file)

    return args

# NOTE: not used
def parse_config(argv, args, config_path):
    config_vars = {}

    with open(config_path, 'r') as stream:
        config_vars = yaml.safe_load(stream)
    
    if config_vars is not None:
        default_args = argparse.Namespace()
        default_args.__dict__.update(args.__dict__)
        default_args.__dict__.update(config_vars)

        new_keys = {}
        for k, v in args.__dict__.items():
            if '--'+k in argv or '-'+k in argv or (k not in default_args):
                new_keys[k] = v

        default_args.__dict__.update(new_keys)
        args = default_args

    return args

# NOTE: not used
def add_config_args_to_parser(parser, args, configs_to_add):
    """Update args by merging CLI arguments with YAML config arguments.
    
    CLI arguments take precedence over YAML config arguments.
    
    Args:
        parser (argparse.ArgumentParser): The argument parser.
        args (argparse.Namespace): Parsed CLI arguments.
        configs_to_add (list): List of config argument names to load YAML files from.
    
    Returns:
        argparse.Namespace: Updated arguments namespace.
    """
    args_dict = vars(args)  # Convert Namespace to dictionary
    yaml_defaults = {}

    # Load YAML files and store values in yaml_defaults
    for config_arg in configs_to_add:
        config_path = args_dict.get(config_arg)
        if config_path is None:
            continue  # Skip if config file path is not specified

        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f) or {}  # Load YAML file content

                # Allow later configs to override earlier ones
                for key, value in config_data.items():
                    yaml_defaults[key] = value
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Skipping.")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config {config_path}: {e}")

    # Handle store_true arguments
    for action in parser._actions:
        if isinstance(action, argparse._StoreTrueAction):
            arg_name = action.dest
            # If the argument is not set in the CLI, use the YAML value (if present)
            if not getattr(args, arg_name, False):
                yaml_value = yaml_defaults.get(arg_name, False)
                yaml_defaults[arg_name] = bool(yaml_value)

    # Update parser defaults with YAML values
    parser.set_defaults(**yaml_defaults)

    # Reparse to apply updated defaults while keeping CLI arguments
    updated_args = parser.parse_args(sys.argv[1:])
    return updated_args

def log_args_namespace(args):
    # Convert the args Namespace to a dictionary
    args_dict = vars(args)

    # Log all the keys/arguments of the dictionary
    log_message = "*** All arguments contained in the args variable *** \n\n"
    for key, value in args_dict.items():
        log_message += f"{key}: {value}\n"

    logger.info(log_message)

def get_config_specific_args_from_args(args, specific_config_path):
    # Load the YAML config to get config-specific keys
    with open(specific_config_path, "r") as f:
        specific_config_keys = yaml.safe_load(f).keys()

    # Filter args to include only the arguments of the specific config file given
    specific_config_args = {key: value for key, value in vars(args).items() if key in specific_config_keys}

    return specific_config_args

def save_config(cfg, path):
    with open(path, 'w') as cfg_file:
        yaml.dump(cfg, cfg_file)

def generate_timestamped_experiment_name(exp_basename):
    now = datetime.datetime.now()
    timestamp = now.strftime("%d_%m_%H_%M")
    experiment_name = f"{exp_basename}_{timestamp}"
    return experiment_name

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

def plot_lr_schedule(lr_values):
    plt.figure(figsize=(10, 5))
    plt.plot(lr_values, label="Learning Rate")
    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid()
    plt.savefig("learning_rate_schedule.png")
    plt.show()

    # Save the plot at the root
    # TODO: see if useful to save it in the experiment folder
    plt.savefig("learning_rate_schedule.png")

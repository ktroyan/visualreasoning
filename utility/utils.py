import os
import sys
import yaml
import argparse
import glob
import datetime

# Personal codebase dependencies
from utility.logging import logger

def parse_config(args, config_path, argv):
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

def parse_args_and_configs(parser, argv=None):

    # Consider the CLI arguments given
    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)  # args is an instance of a Namespace (i.e., argparse.Namespace)
    
    # Read all the config files from the folder ./configs and its subfolders (e.g., networks/) and add their relative path to a list of config files
    configs = glob.glob('configs/**/*.yaml', recursive=True)
    
    # Add all the arguments from the config files to the parser
    for config_path in configs:
        args = parse_config(args, config_path, argv)    # args is updated iteratively
        
    return args

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

import os
import sys
from datetime import datetime
import yaml
from loguru import logger as loguru_logger


def setup_loguru(loguru_logger):
    # Generate a timestamp for the log file name
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%m-%d-%Hh%M")

    # Get the data environment to have the correct log path
    with open("./configs/base.yaml") as file:
        config_base = yaml.safe_load(file)

    # Make sure the log folder exists
    log_folder = f"{config_base['base']['data_env']}/experiments/logs"
    os.makedirs(log_folder, exist_ok=True)

    # Log file path
    log_filename = (f"{log_folder}/experiment_{formatted_datetime}.log")

    # Logging format
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</> | "
        "<lvl>{level: <8}</> | "
        "<cyan>{file}:{function}:{line}</> | \n"
        "<lvl>{message}</>\n"
    )

    # Remove the default logger with its associated sink
    loguru_logger.remove()
    
    # Add a file sink for logging
    loguru_logger.add(log_filename, level="TRACE", format=fmt)    # levels can be TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR...

    # Add a stdout sink for console logging
    loguru_logger.add(sys.stdout, level="INFO", format=fmt)    # levels can be TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR...

    return loguru_logger

# Configure the logger and make the object accessible to other modules
logger = setup_loguru(loguru_logger)
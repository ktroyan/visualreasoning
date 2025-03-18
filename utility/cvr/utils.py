import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms as tvt
from typing import Dict

# Personal codebase dependencies
from utility.logging import logger

def compute_dataset_stats(dataset_path):
    # Load the dataset file
    dataset_file_type = dataset_path.split(".")[-1]
    if dataset_file_type == "json":
        dataset_pd = pd.read_json(dataset_path)

        # Flatten each sample in the 'input' column
        sample_inputs = dataset_pd["input"].apply(lambda image: np.array(image).flatten()).values

    elif dataset_file_type == "csv":
        dataset_pd = pd.read_csv(dataset_path)
        
        # Normalize file paths and load images
        dataset_pd['filepath'] = dataset_pd['filepath'].apply(lambda path: os.path.normpath(os.path.join("CVR", path)))
        
        # Load images and convert to tensors
        samples = [tvt.ToTensor()(Image.open(sample_path)) for sample_path in dataset_pd['filepath']]
        
        # Flatten each sample (3-channel image) and store in the 'input' column
        dataset_pd['input'] = [sample.flatten().numpy() for sample in samples]
        
        # Extract flattened samples
        sample_inputs = dataset_pd['input'].values

    else:
        raise Exception("File format not supported. Please provide a csv or json file.")

    # Check if the 'input' column exists
    if "input" not in dataset_pd.columns:
        raise Exception("The dataset does not contain an 'input' column.")

    logger.info(f"Number of samples: {len(sample_inputs)}")

    ## Global stats (i.e., mean and std computed across all pixels in the dataset)
    # NOTE: this is the one to use for normalization of images of the dataset before creating the dataloader
    
    # Flatten all samples into a single array
    all_pixels = np.concatenate(sample_inputs)

    # Compute the dataset mean and std
    dataset_mean_global = np.mean(all_pixels)
    dataset_std_global = np.std(all_pixels)

    # Log the dataset mean and std
    logger.info(f"Dataset mean (global): {dataset_mean_global}")
    logger.info(f"Dataset std (global): {dataset_std_global}")

    ## Per-sample stats (i.e., mean and std computed per sample and averaged across all samples)
    
    # Compute mean and std per sample
    mean_per_sample = [np.mean(sample) for sample in sample_inputs]
    std_per_sample = [np.std(sample) for sample in sample_inputs]

    # Average the mean and std across all samples
    dataset_mean_per_sample = np.mean(mean_per_sample)
    dataset_std_per_sample = np.mean(std_per_sample)

    # Log the dataset mean and std
    logger.info(f"Dataset mean (average per sample): {dataset_mean_per_sample}")
    logger.info(f"Dataset std (average per sample): {dataset_std_per_sample}")

    return dataset_mean_global, dataset_std_global, dataset_mean_per_sample, dataset_std_per_sample

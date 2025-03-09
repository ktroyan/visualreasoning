import os
import json
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as tvt
import numpy as np

# Personal codebase dependencies
from data.data_base import DataModuleBase
from utility.logging import logger
from utility.utils import compute_dataset_stats


class CVRVisionDataset(Dataset):
    def __init__(self, csv_dataset_path, image_size=128, transform=None):
        super().__init__()

        # Get the samples paths from the dataset csv metadata file
        self.samples_metadata = pd.read_csv(csv_dataset_path, header=0)
        self.samples_metadata['filepath'] = self.samples_metadata['filepath'].apply(lambda path: os.path.normpath(os.path.join("CVR", path)))   # Update the paths so that it works from the root directory

        # Load the dictionary mapping the task name to the task id
        task_name_to_id_json = "./CVR/final_datasets/CVR_task_name_to_id.json"
        with open(task_name_to_id_json, 'r') as f:
            self.task_name_to_id = json.load(f)

        self.n_samples = len(self.samples_metadata)
        self.image_size = image_size
        self.transform = transform
        self.totensor = tvt.ToTensor()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        sample_metadata = self.samples_metadata.iloc[idx]
        sample_path = sample_metadata['filepath']
        sample_task_name = sample_metadata['task']

        # Convert the task name to a task id
        sample_task_id = self.task_name_to_id[sample_task_name]

        sample = Image.open(sample_path)
        sample = self.totensor(sample)
        img_size = sample.shape[1]
        pad = img_size - self.image_size

        # One sample is 4 RGB (i.e., 3 channels) images of size (img_size, img_size), where img_size = 128
        sample = sample.reshape([3, img_size, 4, img_size]).permute([2, 0, 1, 3])[:, :, pad//2:-pad//2, pad//2:-pad//2]

        # Perform the given transformation on the input image
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, sample_task_id


class CVRVisionDataModule(DataModuleBase):

    def __init__(self, data_config, image_size=128, **kwargs):

        super().__init__(data_config.num_workers, 
                         data_config.train_batch_size, 
                         data_config.val_batch_size, 
                         data_config.test_batch_size, 
                         data_config.test_in_and_out_domain)    # pass the relevant arguments of the child class's __init__() method to the parent class __init__() method to make sure that the parent class is initialized properly

        _unmatched_args = kwargs

        train_set_path = data_config.dataset_dir + '/train.csv'
        val_set_path = data_config.dataset_dir + '/val.csv'
        test_set_path = data_config.dataset_dir + '/test.csv'

        if data_config.transform.enabled:
            dataset_mean_global, dataset_std_global, _, _ = compute_dataset_stats(train_set_path)
            transform = self._transforms(dataset_mean_global, dataset_std_global)
        else:
            transform = None

        # Create the torch Dataset objects that will then be used to create the dataloaders
        self.train_set = CVRVisionDataset(train_set_path, image_size=image_size, transform=transform)
        self.val_set = CVRVisionDataset(val_set_path, image_size=image_size, transform=transform)
        self.test_set = CVRVisionDataset(test_set_path, image_size=image_size, transform=transform)

        if data_config.use_gen_test_set:
            gen_test_set_path = data_config.dataset_dir + '/test_gen.csv'
            self.gen_test_set = CVRVisionDataset(gen_test_set_path, image_size=image_size, transform=transform)

    # TODO: How did they choose mean of 0.9 and std of 0.1 ?
    # TODO: Check if the mean and std are computed and used correctly
    # NOTE:
    # mean: The average pixel value across the dataset
    # std: The standard deviation of pixel values across the dataset.
    def _transforms(self, mean=0.9, std=0.1):
        transforms = tvt.Compose([
            tvt.Normalize((mean, mean, mean), (std, std, std)),
        ])
        return transforms
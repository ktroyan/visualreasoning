import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms as tvt
import torch.nn.functional as F

# Personal codebase dependencies
from data.data_base import DataModuleBase
from utility.logging import logger
from utility.utils import compute_dataset_stats


class REARCVisionDataset(Dataset):
    def __init__(self, json_dataset_path, image_size=30, transform=None):
        super().__init__()

        # Get the samples paths from the dataset csv metadata file
        self.data_samples = pd.read_json(json_dataset_path)

        # Load the dictionary mapping the task name to the task id
        task_name_to_id_json = "./REARC/final_datasets/REARC_task_name_to_id.json"
        with open(task_name_to_id_json, 'r') as f:
            self.task_name_to_id = json.load(f)

        self.n_samples = len(self.data_samples)
        self.image_grid_size = image_size
        self.transform = transform

    def pad_tensor(self, x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        x_tensor_padded = F.pad(
            x_tensor, 
            (0, self.image_grid_size - x.shape[1],  # pad width dim (left, right)
             0, self.image_grid_size - x.shape[0]), # pad height dim (top, bottom)
            value=0  # padding value
        )
        return x_tensor_padded

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        sample = self.data_samples.iloc[idx]
        x = sample['input']
        y = sample['output']
        sample_task_name = sample['task']

        # Convert the task name to a task id
        sample_task_id = self.task_name_to_id[sample_task_name]

        # Transform the input grid image x (which is a list of list of integers) to a 2D-tensor
        x = torch.tensor(x, dtype=torch.float32)
        
        # Transform the output grid image y (which is a list of list of integers) to a 2D-tensor
        y = torch.tensor(y, dtype=torch.float32)

        # Pad the input tensor to the desired fixed size
        x = self.pad_tensor(x)

        # Pad the output tensor to the desired fixed size
        y = self.pad_tensor(y)

        # Add a channel (C) dimension to the 2D-tensor. This is necessary if we use a model made for images as input
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)  # TODO: do we have to add a channel dimension to the output tensor? Or undo the channel dimension to the input when computing the loss and acc?

        # Perform the given transformation on the input image
        if self.transform is not None:
            x = self.transform(x)

        return x, y, sample_task_id


class REARCVisionDataModule(DataModuleBase):

    def __init__(self, data_config, image_size=30, **kwargs):

        super().__init__(data_config.num_workers,
                         data_config.train_batch_size, 
                         data_config.val_batch_size, 
                         data_config.test_batch_size, 
                         data_config.test_in_and_out_domain)    # pass the relevant arguments of the child class's __init__() method to the parent class __init__() method to make sure that the parent class is initialized properly

        _unmatched_args = kwargs

        train_set_path = data_config.dataset_dir + '/train.json'
        val_set_path = data_config.dataset_dir + '/val.json'
        test_set_path = data_config.dataset_dir + '/test.json'


        if data_config.transform.enabled:
            dataset_mean_global, dataset_std_global, _, _ = compute_dataset_stats(train_set_path)
            transform = self._transforms(dataset_mean_global, dataset_std_global)
        else:
            transform = None

        # Create the torch Dataset objects that will then be used to create the dataloaders
        self.train_set = REARCVisionDataset(train_set_path, image_size=image_size, transform=transform)
        self.val_set = REARCVisionDataset(val_set_path, image_size=image_size, transform=transform)
        self.test_set = REARCVisionDataset(test_set_path, image_size=image_size, transform=transform)

        if data_config.use_gen_test_set:
            gen_test_set_path = data_config.dataset_dir + '/test_gen.json'
            self.gen_test_set = REARCVisionDataset(gen_test_set_path, image_size=image_size, transform=transform)


    # TODO: Check if the mean and std are computed and used correctly
    # NOTE
    # mean: The average pixel value across the dataset
    # std: The standard deviation of pixel values across the dataset.
    def _transforms(self, mean=0.9, std=0.1):
        transforms = tvt.Compose([
            tvt.Normalize((mean,), (std,)),
        ])
        return transforms
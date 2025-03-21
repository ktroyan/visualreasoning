import json
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

# Personal codebase dependencies
from data.data_base import DataModuleBase
from utility.logging import logger


class REARCDataset(Dataset):
    def __init__(self, json_dataset_path, image_size, transform=None):
        super().__init__()

        # Get the samples paths from the dataset csv metadata file
        self.data_samples_df = pd.read_json(json_dataset_path)

        # Load the dictionary mapping the task name to the task id
        task_name_to_id_json = "./REARC/final_datasets/REARC_task_name_to_id.json"
        with open(task_name_to_id_json, 'r') as f:
            self.task_name_to_id = json.load(f)

        self.n_samples = len(self.data_samples_df)
        self.transform = transform

        # Define the maximum image size to which all images will be padded
        self.max_img_size = image_size  # if we want to pad to a fixed size given as argument


    def pad_tensor(self, x: torch.Tensor, pad_value: int) -> torch.Tensor:
        """ 
        Pad the 2D input tensor x to the desired fixed size with the given pad value 

        TODO: What value should the padding take for convenience? Some str symbol such as "<pad_token>" or simply some int such as 11?
        """
        x_padded = F.pad(
            x, 
            (0, self.max_img_size - x.shape[1],  # pad width dim (left, right)
             0, self.max_img_size - x.shape[0]), # pad height dim (top, bottom)
            value=pad_value  # padding value
        )
        return x_padded

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        sample = self.data_samples_df.iloc[idx]
        x = sample['input']
        y = sample['output']
        sample_task_name = sample['task']

        # Convert the task name to a task id
        sample_task_id = self.task_name_to_id[sample_task_name]

        # Transform the input grid image x (which is a list of list of integers) to a 2D-tensor
        x = torch.tensor(x, dtype=torch.float32)
        
        # Transform the output grid image y (which is a list of list of integers) to a 2D-tensor
        y = torch.tensor(y, dtype=torch.float32)

        # Get the true (i.e., non-padded) size of the output tensor
        y_true_size = y.shape
        y_true_size = torch.tensor(y_true_size, dtype=torch.float32)

        # Pad the input tensor to the desired fixed size
        x = self.pad_tensor(x, pad_value=10)

        # Pad the output tensor to the desired fixed size; it is needed for Dataloader collation and teacher forcing AR decoding
        y = self.pad_tensor(y, pad_value=10)

        # Perform the given transformation on the input image
        if self.transform is not None:
            x = self.transform(x)

        return x, y, sample_task_id, y_true_size    # Tensor, Tensor, Tensor, list


def get_max_grid_size(df, columns=["input", "output"]):
    
    def get_grid_size(grid):
        if not grid:  # handle empty grid
            return (0, 0)
        height = len(grid)  # number of rows
        width = len(grid[0]) if height > 0 else 0  # width of the first row; NOTE: all rows of a same sample input/output are assumed to have the same width since otherwise it is not a rectangular grid image
        return height, width

    # Apply function to get (height, width) for each grid
    img_sizes = df[columns].map(get_grid_size)  # apply to each element in the DataFrame

    # Find max height and max width across all specified columns
    max_height = img_sizes.apply(lambda row: max(h for h, _ in row), axis=1).max()
    max_width = img_sizes.apply(lambda row: max(w for _, w in row), axis=1).max()

    max_size = max(max_height, max_width)

    return max_size

def get_max_img_size_across_dataset_splits(data_splits_paths):
    overall_max_img_size = 0
    for split_path in data_splits_paths:
        # Get the samples paths from the dataset csv metadata file
        data_samples_df = pd.read_json(split_path)

        # Define the maximum image size to which all images will be padded
        max_img_size = get_max_grid_size(data_samples_df, columns=["input", "output"])    # find the maximum image size in the dataset; useful to minimize padding
        if max_img_size > overall_max_img_size:
            overall_max_img_size = max_img_size
    
    return overall_max_img_size

class REARCDataModule(DataModuleBase):

    def __init__(self, data_config, **kwargs):

        super().__init__(data_config.num_workers,
                         data_config.shuffle_train_dl,
                         data_config.train_batch_size, 
                         data_config.val_batch_size, 
                         data_config.test_batch_size, 
                         data_config.test_in_and_out_domain)    # pass the relevant arguments of the child class's __init__() method to the parent class __init__() method to make sure that the parent class is initialized properly

        _unmatched_args = kwargs

        self.image_size = data_config.image_size # if None, the max grid image size will be inferred from the dataset

        train_set_path = data_config.dataset_dir + '/train.json'
        val_set_path = data_config.dataset_dir + '/val.json'
        test_set_path = data_config.dataset_dir + '/test.json'
        data_splits_paths = [train_set_path, val_set_path, test_set_path]

        if data_config.use_gen_test_set:
            gen_test_set_path = data_config.dataset_dir + '/test_gen.json'
            data_splits_paths.append(gen_test_set_path)

        # Max. image size (to which to pad all images)
        if self.image_size is None:
            self.image_size = get_max_img_size_across_dataset_splits(data_splits_paths)
            if self.image_size == 0:
                raise ValueError("The max grid image size across the dataset splits is 0. Please check the dataset.")
            logger.info(f"Using the max grid image size of {self.image_size} that was inferred from the dataset splits.")
        else:
            logger.info(f"Using set max grid image size of {self.image_size} for padding.")


        # Data transformation
        if data_config.transform.enabled:
            transform = self._transforms()
        else:
            transform = None

        # Create the torch Dataset objects that will then be used to create the dataloaders
        self.train_set = REARCDataset(train_set_path, self.image_size, transform=transform)
        self.val_set = REARCDataset(val_set_path, self.image_size, transform=transform)
        self.test_set = REARCDataset(test_set_path, self.image_size, transform=transform)

        if data_config.use_gen_test_set:
            self.gen_test_set = REARCDataset(gen_test_set_path, self.image_size, transform=transform)

    def _transforms(self):
        return None
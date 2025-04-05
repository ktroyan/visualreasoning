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
        self.max_img_size = image_size + 1 + 1 # +1 to account for the border tokens (right, bottom) and +1 for the newline token at the end of each row and make the grid square

        self.PAD_TOKEN = 10
        self.X_ENDGRID_TOKEN = 11
        self.Y_ENDGRID_TOKEN = 12
        self.XY_ENDGRID_TOKEN = 13
        self.NL_GRID_TOKEN = 14
        self.NUM_SPECIAL_TOKENS = 5 # PAD, X_END, Y_END, XY_END, NL_GRID_TOKEN
        
        log_message = ""
        log_message += f"Input grid size (with special tokens): {self.max_img_size}x{self.max_img_size}"
        log_message += f"\nSpecial data token IDs: PAD={self.PAD_TOKEN}, X_END={self.X_ENDGRID_TOKEN}, Y_END={self.Y_ENDGRID_TOKEN}, XY_END={self.XY_ENDGRID_TOKEN}, NL_GRID_TOKEN={self.NL_GRID_TOKEN}"
        logger.info(log_message)


    def pad_with_2d_visual_tokens(self, grid: torch.Tensor) -> torch.Tensor:
        """ 
        Pad the grid (to a fixed size) with visual tokens.
        Essentially:
        - add border tokens around (right, bottom) the true grid
        - add pad tokens after the border tokens and on all row positions except the last (i.e., except for the last column)
        - add a special newline pad token at the last position of each row
        """
        
        # Add special border tokens to the original grid
        h, w = grid.shape
        bordered_grid = torch.full((h + 1, w + 1), self.PAD_TOKEN, dtype=torch.long)    # [h+1, w+1] to account for the border tokens
        bordered_grid[0:h, 0:w] = grid  # place original grid inside

        bordered_grid[h, 0:w] = self.Y_ENDGRID_TOKEN    # bottom border
        bordered_grid[0:h, w] = self.X_ENDGRID_TOKEN    # right border
        bordered_grid[h, w] = self.XY_ENDGRID_TOKEN     # bottom-right corner

        # Add special pad tokens to the bordered grid
        current_h, current_w = bordered_grid.shape  # [h+1, w+1]
        
        pad_right = self.max_img_size - current_w - 1   # -1 to account for the NL_GRID_TOKEN padded later
        pad_bottom = self.max_img_size - current_h
        
        if pad_right < 0 or pad_bottom < 0:
             raise ValueError(f"Grid with borders ({current_h}x{current_w}) exceeds target image_size ({self.max_img_size}x{self.max_img_size}). Increase image_size.")

        grid_padded = F.pad(
            bordered_grid, 
            (0, pad_right,  # pad width dim (left, right)
             0, pad_bottom), # pad height dim (top, bottom)
            mode='constant', 
            value=self.PAD_TOKEN    # padding value
        )   # [self.max_img_size, self.max_img_size-1]

        # Add a special newline pad token at the last position of each row
        grid_padded = F.pad(
            grid_padded, 
            (0, 1,  # pad width dim (left, right)
             0, 0), # pad height dim (top, bottom)
            mode='constant', 
            value=self.NL_GRID_TOKEN    # padding value
        )   # [self.max_img_size, self.max_img_size]

        assert grid_padded.shape == (self.max_img_size, self.max_img_size), f"Grid shape after padding {grid_padded.shape} does not match expected shape ({self.max_img_size}, {self.max_img_size})."
        
        return grid_padded

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        # Get the sample data for the current index idx
        sample = self.data_samples_df.iloc[idx]
        x = sample['input']
        y = sample['output']
        sample_task_name = sample['task']

        # Convert the task name to a task id tensor
        sample_task_id = torch.tensor(self.task_name_to_id[sample_task_name], dtype=torch.long)

        # Transform the input grid image x (which is a list of list of integers) to a 2D-tensor
        x = torch.tensor(x, dtype=torch.long)

        # Perform the given transformation on the input image
        # TODO: Should the potential transform be applied before the border and pad tokens added or after?
        if self.transform is not None:
            x = self.transform(x)

        # Transform the output grid image y (which is a list of list of integers) to a 2D-tensor
        y = torch.tensor(y, dtype=torch.long)

        # Get the true (i.e., non-padded) size of the output tensor
        y_true_size = y.shape
        y_true_size = torch.tensor(y_true_size, dtype=torch.long)

        # Add special visual tokens and pad the input tensor to the desired fixed size
        x = self.pad_with_2d_visual_tokens(x)

        # Add special visual tokens and pad the output tensor to the desired fixed size
        y = self.pad_with_2d_visual_tokens(y)

        assert x.shape == (self.max_img_size, self.max_img_size), f"Input grid shape {x.shape} does not match expected shape ({self.max_img_size}, {self.max_img_size})."
        assert y.shape == (self.max_img_size, self.max_img_size), f"Output grid shape {y.shape} does not match expected shape ({self.max_img_size}, {self.max_img_size})."

        return x, y, sample_task_id, y_true_size


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
                         data_config.test_in_and_out_domain
                         )

        _unmatched_args = kwargs

        self.image_size = data_config.image_size # if None, the max grid image size will be inferred from the dataset

        train_set_path = data_config.dataset_dir + '/train.json'
        val_set_path = data_config.dataset_dir + '/val.json'
        test_set_path = data_config.dataset_dir + '/test.json'
        data_splits_paths = [train_set_path, val_set_path, test_set_path]

        if data_config.use_gen_test_set:
            gen_test_set_path = data_config.dataset_dir + '/test_gen.json'
            data_splits_paths.append(gen_test_set_path)

        # Max. image size (without considering the special visual tokens)
        if self.image_size is None:
            self.image_size = get_max_img_size_across_dataset_splits(data_splits_paths)
            if self.image_size == 0:
                raise ValueError("The max grid image size across the dataset splits is 0. Please check the dataset.")
            logger.info(f"Using the max grid image size of {self.image_size} that was inferred from the dataset splits.")
        else:
            logger.info(f"Using the set max grid image size of {self.image_size} for padding.")

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
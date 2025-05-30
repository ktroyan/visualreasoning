import json
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np

# Personal codebase dependencies
from .data_base import DataModuleBase
from utility.custom_logging import logger
from .external.vitarc.obj_idx_utils import generate_input_type_ids_multi

def create_object_grid(input_grid, max_img_size, special_tokens_dic):
    """
    Create a grid of object ids for the given input grid.
    The resulting grid is a 2D tensor of size (max_img_size, max_img_size) where each cell contains the object id for that cell.
    
    TODO: What should be our approach? Moreover, discuss their rectangle bounding boxes approach vs. my thought of segmenting instead.
          The former may confuse the model I think.
    
    What they seem to do in ViTARC is: (see gen_dataset.py from line 185)
    - replace all the values in the grid by their custom tokens
    - create the object ids grid for the original input grid (without any padding or special tokens)
    - flatten the object ids grid
    - pad the object ids grid with zeros to reach the tgt size (which is their maximum size)
    I am not sure why they do it for both the input grid and target grid, but I guess that what matters is the input grid.

    Regardless, I think that what we should do is:
    - create the object ids grid for the original input grid (without any padding or special tokens)
    - [convert to a tensor]
    - pad the object ids grid with PAD tokens to reach the max_img_size
    - flatten the grid
    """

    object_ids_grid = generate_input_type_ids_multi(np.array(input_grid), visualize=False)  # NOTE: may get warning due to several workers/processes running for data
    object_ids_grid = torch.tensor(object_ids_grid, dtype=torch.long)

    # Pad with PAD tokens
    object_ids_grid = F.pad(
        object_ids_grid, 
        (0, max_img_size - object_ids_grid.shape[1],    # pad right
         0, max_img_size - object_ids_grid.shape[0]),   # pad bottom
        mode='constant', 
        value=special_tokens_dic['PAD']    # padding value
    )   # [max_img_size, max_img_size]

    # Flatten the object ids grid to a sequence
    object_ids_grid = object_ids_grid.flatten()

    return object_ids_grid

class REARCDataset(Dataset):
    def __init__(self, json_dataset_path, image_size, use_visual_tokens, use_grid_object_ids, transform=None):
        super().__init__()

        # Get whether visual tokens should be used or not
        self.use_visual_tokens = use_visual_tokens

        # Get whether grid object ids should be used or not
        self.use_grid_object_ids = use_grid_object_ids

        # Get the samples paths from the json dataset split
        self.data_samples_df = pd.read_json(json_dataset_path)

        # Load the dictionary mapping the task name to the task id
        task_name_to_id_json = "./REARC/final_datasets/REARC_task_name_to_id.json"
        with open(task_name_to_id_json, 'r') as f:
            self.task_name_to_id = json.load(f)

        self.n_samples = len(self.data_samples_df)
        self.transform = transform

        # Store special tokens
        self.special_tokens = {}

        # Define padding token (always)
        self.PAD_TOKEN = 10 # token id
        self.special_tokens['PAD'] = self.PAD_TOKEN

        # Define visual tokens (only if enabled)
        if use_visual_tokens:
            self.X_ENDGRID_TOKEN = 11   # token id
            self.Y_ENDGRID_TOKEN = 12   # token id
            self.XY_ENDGRID_TOKEN = 13  # token id
            self.NL_GRID_TOKEN = 14     # token id
            self.NUM_SPECIAL_TOKENS = 5 # PAD, X_END, Y_END, XY_END, NL_GRID_TOKEN

            self.special_tokens['X_ENDGRID'] = self.X_ENDGRID_TOKEN
            self.special_tokens['Y_ENDGRID'] = self.Y_ENDGRID_TOKEN
            self.special_tokens['XY_ENDGRID'] = self.XY_ENDGRID_TOKEN
            self.special_tokens['NL_GRID'] = self.NL_GRID_TOKEN
            
            # Define the maximum image size to which all images have to be padded
            self.max_img_size = image_size + 1 + 1 # +1 to account for the border tokens (right, bottom) and +1 for the newline token at the end of each row and make the grid square

            log_message = f"Input grid size (with special tokens): {self.max_img_size}x{self.max_img_size}"
            log_message += f"\nSpecial data token IDs: PAD={self.PAD_TOKEN}, X_END={self.X_ENDGRID_TOKEN}, Y_END={self.Y_ENDGRID_TOKEN}, XY_END={self.XY_ENDGRID_TOKEN}, NL_GRID_TOKEN={self.NL_GRID_TOKEN}"

        else:
            self.NUM_SPECIAL_TOKENS = 1 # only the PAD token

            # Define the maximum image size to which all images have to be padded
            self.max_img_size = image_size
            
            log_message = f"Input grid size (with special tokens): {self.max_img_size}x{self.max_img_size}"
            log_message += f"\nSpecial data token IDs: PAD={self.PAD_TOKEN}"
        
        logger.info(log_message)

    def is_grid_valid(self, grid: torch.Tensor) -> bool:
        """ TODO: Implement this function to check if a grid is valid w.r.t. the tokens defined, sizes, etc. """
        raise NotImplementedError("The function is_grid_valid is not implemented.")

    def add_noise(self, grid: torch.Tensor, noise_ratio: float = 0.5) -> torch.Tensor:
        """
        Add noise to the grid by randomizing `noise_ratio` of the values in a contiguous block (randomly placed).
        The noise ratio is thus the expected maximum accuracy that a model can reach with such data.
        """
        h, w = grid.shape
        num_elements = int(h * w * noise_ratio)

        # Clone the original grid (to avoid in-place modification with possible side-effects)
        grid = grid.clone()

        # Randomly choose a start index for the contiguous block
        max_start = h * w - num_elements
        start_idx = torch.randint(0, max_start + 1, (1,)).item()
        indices = torch.arange(start_idx, start_idx + num_elements)

        # Number of different tokens that can appear in the grid
        num_token_types = 10 + self.NUM_SPECIAL_TOKENS

        # Add the noise
        for idx in indices:
            row = idx // w
            col = idx % w
            grid[row, col] = torch.randint(0, num_token_types, (1,)).item()

        return grid


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
            (0, pad_right,      # pad width dim (left, right)
             0, pad_bottom),    # pad height dim (top, bottom)
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
    
    def pad_2d(self, grid: torch.Tensor) -> torch.Tensor:
        """ 
        Pad the grid (to a fixed size) with pad tokens.
        Essentially:
        - add pad tokens to the right and bottom of the grid to reach the max grid size.
        """
        
        current_h, current_w = grid.shape
        
        pad_right = self.max_img_size - current_w
        pad_bottom = self.max_img_size - current_h
        
        if pad_right < 0 or pad_bottom < 0:
             raise ValueError(f"Grid ({current_h}x{current_w}) exceeds target image_size ({self.max_img_size}x{self.max_img_size}). Increase image_size.")
        
        grid_padded = F.pad(
            grid, 
            (0, pad_right,      # pad width dim (left, right)
             0, pad_bottom),    # pad height dim (top, bottom)
            mode='constant', 
            value=self.PAD_TOKEN    # padding value
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
        # TODO: Should the potential transform be applied before the border and pad tokens added or after? Usually before?
        if self.transform is not None:
            x = self.transform(x)

        # Transform the output grid image y (which is a list of list of integers) to a 2D-tensor
        y = torch.tensor(y, dtype=torch.long)

        # Get the true (i.e., non-padded) size of the output tensor
        y_true_size = y.shape
        y_true_size = torch.tensor(y_true_size, dtype=torch.long)


        # TODO: See the TODO below.
        # if self.use_grid_object_ids:
        #     x_grid_object_ids = create_object_grid(x, self.max_img_size, self.special_tokens)
        # else:
        #     # We cannot return None, so we create a grid of -1 values (as such value never appears) as it will not be used
        #     x_grid_object_ids = torch.full((self.max_img_size * self.max_img_size,), -1, dtype=torch.long)  # [max_img_size * max_img_size]


        if self.use_visual_tokens:
            # Use special visual tokens and pad the input tensor to the desired fixed size
            x = self.pad_with_2d_visual_tokens(x)

            # Use special visual tokens and pad the output tensor to the desired fixed size
            y = self.pad_with_2d_visual_tokens(y)
        
        else:
            # Pad the input tensor to the desired fixed size
            x = self.pad_2d(x)

            # Pad the output tensor to the desired fixed size
            y = self.pad_2d(y)

        assert x.shape == (self.max_img_size, self.max_img_size), f"Input grid shape {x.shape} does not match expected shape ({self.max_img_size}, {self.max_img_size})."
        assert y.shape == (self.max_img_size, self.max_img_size), f"Output grid shape {y.shape} does not match expected shape ({self.max_img_size}, {self.max_img_size})."


        # Create a grid containing object ids for the input grid x
        # TODO: Should they be created with the original input grid or the padded one (possibly with special visual tokens too) ?
        #       I guess the latter, but not sure when checking VITARC code as they seem to use the original input grid.
        #       But then I am not sure if it's a good choice as the model will see border tokens and newline tokens and in the
        #       object ids grid there would be zeros (according to ViTARC code), so background (?) at those locations.
        #       Hence, I am thinking that it is better to perform it on the fully padded (with special tokens too) grid?
        #       Otherwise mark them as PAD tokens instead of background ?
        if self.use_grid_object_ids:
            x_grid_object_ids = create_object_grid(x, self.max_img_size, self.special_tokens)
        else:
            # We cannot return None, so we create a grid of -1 values (as such value never appears) as it will not be used
            x_grid_object_ids = torch.full((self.max_img_size * self.max_img_size,), -1, dtype=torch.long)  # [max_img_size * max_img_size]


        # Add noise to the target grid y
        # y = self.add_noise(y, noise_ratio=0.5)  # TODO: Remove after we have checked that the code is correct

        return x, y, sample_task_id, y_true_size, x_grid_object_ids, self.special_tokens


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

    def __init__(self, data_config, model_config, **kwargs):

        super().__init__(data_config.num_workers,
                         data_config.shuffle_train_dl,
                         data_config.train_batch_size, 
                         data_config.val_batch_size, 
                         data_config.test_batch_size,
                         data_config.use_gen_test_set,
                         data_config.validate_in_and_out_domain
                         )

        _unmatched_args = kwargs

        self.image_size = data_config.image_size # if None, the max grid image size will be inferred from the dataset

        train_set_path = data_config.dataset_dir + '/train.json'
        val_set_path = data_config.dataset_dir + '/val.json'
        test_set_path = data_config.dataset_dir + '/test.json'
        data_splits_paths = [train_set_path, val_set_path, test_set_path]

        if data_config.use_gen_test_set:
            gen_test_set_path = data_config.dataset_dir + '/gen_test.json'
            data_splits_paths.append(gen_test_set_path)
            
            if data_config.validate_in_and_out_domain:
                gen_val_set_path = data_config.dataset_dir + '/gen_test.json'   # NOTE: We make the choice to use gen_test_set to monitor the OOD validation performance during training because the official final results are reported on a new OOD test set
                data_splits_paths.append(gen_val_set_path)

        # Max. image size (without considering any sort of special tokens such as padding or other visual tokens) across all the dataset splits
        if self.image_size is None:
            self.image_size = get_max_img_size_across_dataset_splits(data_splits_paths)
            if self.image_size == 0:
                raise ValueError("The max grid image size across the dataset splits is 0. Please check the dataset.")
            logger.info(f"Using a max grid image size of {self.image_size} that was inferred from the dataset splits.")
        else:
            logger.info(f"Using a set (through config) max grid image size of {self.image_size}.")

        # Get whether visual tokens should be used or not
        if model_config.visual_tokens.enabled:
            use_visual_tokens = True
            logger.info(f"Visual Tokens enabled. Using special visual tokens for the input and output grids.")
        else:
            use_visual_tokens = False

        if model_config.ope.enabled:
            use_grid_object_ids = True
            logger.info(f"OPE enabled. We will create grid object ids for the input grid.")
        else:
            use_grid_object_ids = False

        # Data transformation
        if data_config.transform.enabled:
            transform = self._transforms()
        else:
            transform = None

        # Create the torch Dataset objects that will then be used to create the dataloaders
        self.train_set = REARCDataset(train_set_path, self.image_size, use_visual_tokens, use_grid_object_ids, transform=transform)
        self.val_set = REARCDataset(val_set_path, self.image_size, use_visual_tokens, use_grid_object_ids, transform=transform)
        self.test_set = REARCDataset(test_set_path, self.image_size, use_visual_tokens, use_grid_object_ids, transform=transform)

        if data_config.use_gen_test_set:
            self.gen_test_set = REARCDataset(gen_test_set_path, self.image_size, use_visual_tokens, use_grid_object_ids, transform=transform)

            if data_config.validate_in_and_out_domain:
                self.gen_val_set = REARCDataset(gen_val_set_path, self.image_size, use_visual_tokens, use_grid_object_ids, transform=transform)

    def _transforms(self):
        return None
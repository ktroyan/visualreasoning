import json
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset


# Personal codebase dependencies
from .data_base import DataModuleBase
from utility.logging import logger
from .external.vitarc.obj_idx_utils import generate_input_type_ids_multi

def create_object_grid(input_grid, max_img_size, special_tokens_dic):
    """
    Create a grid of object ids for the given input grid.
    The resulting grid is a 2D tensor of size (max_img_size, max_img_size) where each cell contains the object id for that cell.
    
    TODO: What should be our approach?
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

class BEFOREARCDataset(Dataset):
    def __init__(self, dataset_split_df, image_size, use_visual_tokens, use_grid_object_ids, task_embedding_approach, max_transformation_depth, transform=None):
        super().__init__()

        # Get whether visual tokens should be used or not
        self.use_visual_tokens = use_visual_tokens

        # Get whether grid object ids should be used or not
        self.use_grid_object_ids = use_grid_object_ids

        # Get the task embedding approach to use (useful for Compositionality)
        self.task_embedding_approach = task_embedding_approach

        # Get the samples paths from the parquet file to a pandas dataframe
        self.dataset_split_df = dataset_split_df
        
        self.n_samples = len(dataset_split_df)
        self.transform = transform

        # Store special grid tokens
        self.special_grid_tokens = {}

        # Define padding token (always)
        self.PAD_TOKEN = 10 # token id
        self.special_grid_tokens['PAD'] = self.PAD_TOKEN

        # Define visual tokens (only if enabled)
        if use_visual_tokens:
            self.X_ENDGRID_TOKEN = 11   # token id
            self.Y_ENDGRID_TOKEN = 12   # token id
            self.XY_ENDGRID_TOKEN = 13  # token id
            self.NL_GRID_TOKEN = 14     # token id
            self.NUM_SPECIAL_GRID_TOKENS = 5 # PAD, X_END, Y_END, XY_END, NL_GRID_TOKEN

            self.special_grid_tokens['X_ENDGRID'] = self.X_ENDGRID_TOKEN
            self.special_grid_tokens['Y_ENDGRID'] = self.Y_ENDGRID_TOKEN
            self.special_grid_tokens['XY_ENDGRID'] = self.XY_ENDGRID_TOKEN
            self.special_grid_tokens['NL_GRID'] = self.NL_GRID_TOKEN
            
            # Define the maximum image size to which all images have to be padded
            self.max_img_size = image_size + 1 + 1 # +1 to account for the border tokens (right, bottom) and +1 for the newline token at the end of each row and make the grid square

            log_message = f"Input grid size (with special tokens): {self.max_img_size}x{self.max_img_size}"
            log_message += f"\nSpecial data token IDs: PAD={self.PAD_TOKEN}, X_END={self.X_ENDGRID_TOKEN}, Y_END={self.Y_ENDGRID_TOKEN}, XY_END={self.XY_ENDGRID_TOKEN}, NL_GRID_TOKEN={self.NL_GRID_TOKEN}"

        else:
            self.NUM_SPECIAL_GRID_TOKENS = 1 # only the PAD token

            # Define the maximum image size to which all images have to be padded
            self.max_img_size = image_size
            
            log_message = f"Input grid size (with special tokens): {self.max_img_size}x{self.max_img_size}"
            log_message += f"\nSpecial data token IDs: PAD={self.PAD_TOKEN}"
        
        logger.info(log_message)

        # Get the maximum token id value in the dict of special tokens self.special_tokens
        max_token_id = max(self.special_grid_tokens.values())

        ## Task embedding
        if task_embedding_approach == 'task_tokens':
            self.max_transformation_depth = max_transformation_depth

            logger.info(f"Using task embedding (task tokens) of length: {self.max_transformation_depth}")            

            # Define all the possible elementary transformations
            # TODO: For now we only define those as they are the ones that appear in the experiments
            self.elementary_transformations_to_token_ids = {'identity': max_token_id + 1,    # sort of transformation padding; used to handle variable composite transformation true depth
                                                            'translate_up': max_token_id + 2,
                                                            'rot90': max_token_id + 3,
                                                            'mirror_horizontal': max_token_id + 4,
                                                            'pad_right': max_token_id + 5,
                                                            'fill_holes_different_color': max_token_id + 6,
                                                            'change_shape_color': max_token_id + 7,
                                                            'pad_top': max_token_id + 8,
                                                            'crop_bottom_side': max_token_id + 9,
                                                            'extend_contours_different_color': max_token_id + 10,
                                                            'translate_down': max_token_id + 11,
                                                            'extend_contours_same_color:': max_token_id + 12,
                                                            'pad_left': max_token_id + 13,
                                                            'mirror_vertical': max_token_id + 14,
                                                            'crop_top_side': max_token_id + 15,
                                                            }

            logger.info(f"The token IDs for the transformations are:\n{self.elementary_transformations_to_token_ids}")
        
        else:
            self.max_transformation_depth = 0
            self.elementary_transformations_to_token_ids = {}

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
        num_token_types = 10 + self.NUM_SPECIAL_GRID_TOKENS

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

    def convert_to_tensor(self, grid):
        """ Convert array of numpy objects (lists) into proper 2D array and then to torch (long) tensor. """
        if isinstance(grid, np.ndarray) and grid.dtype == np.object_:
            grid = np.array(list(grid))
        return torch.tensor(grid, dtype=torch.long)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        # Get the sample data for the current index idx
        sample = self.dataset_split_df.iloc[idx]

        # Transform the input grid image x (which is a list of list of integers) to a 2D-tensor
        x = self.convert_to_tensor(sample['input'])

        # Transform the output grid image y (which is a list of list of integers) to a 2D-tensor
        y = self.convert_to_tensor(sample['output'])

        ## Task embedding
        if self.task_embedding_approach == 'task_tokens':

            # The task embedding created for a sample will have to be appended to the sample sequence in the encoder model, once the input grid is flattened to a sequence (so right after PatchEmbed).
            # We create the sequence of tokens for the task embedding by using the elementary transformations obtained from the list in the field "transformations" of the sample. 
            # We maintain the same order and pad with the identity transformation token.
            # A transformation is of the form: ["elementary_transformation_1", "elementary_transformation_2", ...]. It can also just be ["elementary_transformation_1"].

            elem_transformations_sequence = sample['transformation_suite']  # sequence of elementary transformations
        
            # Create the sequence of tokens based on elem_transformations_sequence
            transformation_tokens_sequence = [self.elementary_transformations_to_token_ids[transformation] for transformation in elem_transformations_sequence]

            # Pad the sequence with the identity transformation token
            transformation_tokens_sequence += [self.elementary_transformations_to_token_ids['identity']] * (self.max_transformation_depth - len(transformation_tokens_sequence))

            # Convert the sequence of tokens to a tensor
            task_tokens_seq = torch.tensor(transformation_tokens_sequence, dtype=torch.long)

        elif self.task_embedding_approach == 'example_in_context':
            # Get the input-output pair of grids as an example of the transformation to perform
            example_input = self.convert_to_tensor(sample['demo_input'])
            example_output = self.convert_to_tensor(sample['demo_output'])

        ## Input image data-transform
        if self.transform is not None:
            x = self.transform(x)

        # Get the true (i.e., non-padded) size of the output tensor
        y_true_size = y.shape
        y_true_size = torch.tensor(y_true_size, dtype=torch.long)

        ## Visual tokens and padding
        if self.use_visual_tokens:
            # Use special visual tokens and pad the input tensor to the desired fixed size
            x = self.pad_with_2d_visual_tokens(x)

            # Use special visual tokens and pad the output tensor to the desired fixed size
            y = self.pad_with_2d_visual_tokens(y)

            # Use VTs and pad as well the example input and output tensors if applicable
            if self.task_embedding_approach == 'example_in_context':
                example_input = self.pad_with_2d_visual_tokens(example_input)
                example_output = self.pad_with_2d_visual_tokens(example_output)
        
        else:
            # Pad the input tensor to the desired fixed size
            x = self.pad_2d(x)

            # Pad the output tensor to the desired fixed size
            y = self.pad_2d(y)

            # Pad as well the example input and output tensors if applicable
            if self.task_embedding_approach == 'example_in_context':
                example_input = self.pad_2d(example_input)
                example_output = self.pad_2d(example_output)


        assert x.shape == (self.max_img_size, self.max_img_size), f"Input grid shape {x.shape} does not match expected shape ({self.max_img_size}, {self.max_img_size})."
        assert y.shape == (self.max_img_size, self.max_img_size), f"Output grid shape {y.shape} does not match expected shape ({self.max_img_size}, {self.max_img_size})."

        ## Task embedding (cont'd). Handle the cases where the task embedding (of the given approach) is not used
        if self.task_embedding_approach == 'example_in_context':
            example_in_context = [example_input, example_output]
        else:
            placeholder_grid = torch.full((self.max_img_size, self.max_img_size), -1, dtype=torch.long)  # [max_img_size, max_img_size]
            # Placeholder variable
            example_in_context = [placeholder_grid, placeholder_grid]  # [example_input, example_output] are not used in this case

        if not self.task_embedding_approach == 'task_tokens':
            # Placeholder variable
            task_tokens_seq = torch.full((self.max_transformation_depth,), -1, dtype=torch.long)  # [max_transformation_depth]

        ## Grid object IDs for OPE 
        # Create a grid containing object ids for the input grid x
        # TODO: Should they be created with the original input grid or the padded one (possibly with special visual tokens too) ?
        #       I guess the latter, but not sure when checking VITARC code as they seem to use the original input grid.
        #       But then I am not sure if it's a good choice as the model will see border tokens and newline tokens and in the
        #       object ids grid there would be zeros (according to ViTARC code), so background (?) at those locations.
        #       Hence, I am thinking that it is better to perform it on the fully padded (with special tokens too) grid?
        #       Otherwise mark them as PAD tokens instead of background ?
        if self.use_grid_object_ids:
            x_grid_object_ids = create_object_grid(x, self.max_img_size, self.special_grid_tokens)
        else:
            # We cannot return None, so we create a grid of -1 values (as such value never appears) as it will not be used
            x_grid_object_ids = torch.full((self.max_img_size * self.max_img_size,), -1, dtype=torch.long)  # [max_img_size * max_img_size]

        ## Noisy target (for model performance sanity checks w.r.t. the data)
        # Add noise to the target grid y
        # y = self.add_noise(y, noise_ratio=0.5)  # TODO: Remove after we have checked that the data are created as intended and there is no leakage
        
        return x, y, task_tokens_seq, example_in_context, y_true_size, x_grid_object_ids, self.special_grid_tokens


def get_max_grid_size(df, columns=["input", "output"]):
    
    def get_grid_size(grid):
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

def get_max_img_size_across_dataset_splits(dataset_splits):
    overall_max_img_size = 0

    for split in dataset_splits:

        # Get the maximum image size (to which all images will be padded) for the current dataset split
        max_img_size = get_max_grid_size(split, columns=["input", "output"])    # find the maximum image size in the dataset; useful to minimize padding
        if max_img_size > overall_max_img_size:
            overall_max_img_size = max_img_size
    
    return overall_max_img_size

def get_max_transformation_depth_across_dataset_splits(dataset_splits):
    overall_max_transformation_depth = 0

    for split in dataset_splits:

        # Get the maximum transformation depth (to which all the task embeddings will be padded with the identity transformation token) for the current dataset split
        max_transformation_depth = split['transformation_suite'].apply(len).max()    # find the maximum transformation depth in the current dataset split

        # Update the overall maximum transformation depth
        if max_transformation_depth > overall_max_transformation_depth:
            overall_max_transformation_depth = max_transformation_depth
    
    return overall_max_transformation_depth

class BEFOREARCDataModule(DataModuleBase):

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

        # Load the dataset from HuggingFace
        # Get experiment study from path
        study = data_config.dataset_dir.split('/')[-3]
        if "sys-gen" in study:  # to match the local naming convention of the studies
            study = study.replace('sys-gen', 'generalization')

        # Get experiment setting from path
        setting = data_config.dataset_dir.split('/')[-2]

        # Get experiment name from path
        exp_name = data_config.dataset_dir.split('/')[-1]

        # Dataset path (using HuggingFace datasets)
        dataset_path = f"{study}/{setting}/{exp_name}"
        
        train_set_parquet = load_dataset("taratataw/before-arc", data_files={"data": f"{dataset_path}/train.parquet"})
        val_set_parquet = load_dataset("taratataw/before-arc", data_files={"data": f"{dataset_path}/val.parquet"})
        test_set_parquet = load_dataset("taratataw/before-arc", data_files={"data": f"{dataset_path}/test.parquet"})

        # Convert parquet to pandas dataframe
        train_set_df = train_set_parquet['data'].to_pandas()
        val_set_df = val_set_parquet['data'].to_pandas()
        test_set_df = test_set_parquet['data'].to_pandas()

        dataset_splits = [train_set_df, val_set_df, test_set_df]

        if data_config.use_gen_test_set:
            gen_test_set_parquet = load_dataset("taratataw/before-arc", data_files={"data": f"{dataset_path}/test_ood.parquet"})
            gen_test_set_df = gen_test_set_parquet['data'].to_pandas()
            dataset_splits.append(gen_test_set_df)

            if data_config.validate_in_and_out_domain:
                gen_val_set_parquet = load_dataset("taratataw/before-arc", data_files={"data": f"{dataset_path}/val_ood.parquet"})
                gen_val_set_df = gen_val_set_parquet['data'].to_pandas()
                dataset_splits.append(gen_val_set_df)

        # Max. image size (without considering any sort of special tokens such as padding or other visual tokens)
        if self.image_size is None:
            self.image_size = get_max_img_size_across_dataset_splits(dataset_splits)
            if self.image_size == 0:
                raise ValueError("The max grid image size across the dataset splits is 0. Please check the dataset.")
            logger.info(f"Using a max grid image size of {self.image_size} that was inferred from the dataset splits.")
        else:
            logger.info(f"Using a set (through config) max grid image size of {self.image_size}.")

        if model_config.task_embedding.enabled:
            if model_config.task_embedding.approach == 'task_tokens':
                task_embedding_approach = 'task_tokens'
                # Max. transformation depth
                # Check the maximum transformation depth (i.e., how many elementary transformations at most compose the transformations part of the dataset splits)
                # max_transformation_depth = get_max_transformation_depth_across_dataset_splits(dataset_splits)
                max_transformation_depth = 4    # TODO: Fix it to 4 for now as a fix to handling extra appended tokens more easily (e.g., in Alibi RPE)
            
            elif model_config.task_embedding.approach == 'example_in_context':
                task_embedding_approach = 'example_in_context'
                max_transformation_depth = 0

            else:
                task_embedding_approach = None
                max_transformation_depth = 0
        
        else:
            task_embedding_approach = None
            max_transformation_depth = 0


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
        self.train_set = BEFOREARCDataset(train_set_df, self.image_size, use_visual_tokens, use_grid_object_ids, task_embedding_approach, max_transformation_depth, transform=transform)
        self.val_set = BEFOREARCDataset(val_set_df, self.image_size, use_visual_tokens, use_grid_object_ids, task_embedding_approach, max_transformation_depth, transform=transform)
        self.test_set = BEFOREARCDataset(test_set_df, self.image_size, use_visual_tokens, use_grid_object_ids, task_embedding_approach, max_transformation_depth, transform=transform)

        if data_config.use_gen_test_set:
            self.gen_test_set = BEFOREARCDataset(gen_test_set_df, self.image_size, use_visual_tokens, use_grid_object_ids, task_embedding_approach, max_transformation_depth, transform=transform)

            if data_config.validate_in_and_out_domain:
                self.gen_val_set = BEFOREARCDataset(gen_val_set_df, self.image_size, use_visual_tokens, use_grid_object_ids, task_embedding_approach, max_transformation_depth, transform=transform)

    def _transforms(self):
        return None
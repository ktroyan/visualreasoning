import os
import json
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as tvt
import torch.nn.functional as F

# Personal codebase dependencies
from .data_base import DataModuleBase
from utility.cvr.utils import compute_dataset_stats
from utility.custom_logging import logger


class CVRDataset(Dataset):
    def __init__(self, csv_dataset_path, image_size, transform=None):
        super().__init__()

        self.nb_images_within_sample = 4

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

    def mix_input_and_create_artificial_labels(self, x):
        """
        Randomly permute the images within the input sample and create the associated ground-truth (artificial) label. 
        That is, randomly permute the images so that the odd image is not always the last one (which we don't want the model to learn) and keep track of its new index within the sample.
        """
        # Create a random permutation of nb_images_within_sample elements
        perms = torch.randperm(self.nb_images_within_sample)

        # Apply the random permutation to the images in the input sample
        x = x[perms]

        # The odd image is at the end of the input sample, so we mark it with a 1 (while the others are 0)
        image_indices_marked = torch.zeros(self.nb_images_within_sample, dtype=torch.int)
        image_indices_marked[-1] = 1    # image_indices_marked = [0, 0, ..., 0, 1]
        
        # Apply the random permutation to the marked image indices and get the new index of the odd image
        y = image_indices_marked[perms].argmax()    # argmax() returns the new index of the odd image since it is 1 and all the others are 0

        return x, y


    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        sample_metadata = self.samples_metadata.iloc[idx]
        sample_path = sample_metadata['filepath']
        sample_task_name = sample_metadata['task']

        # Convert the task name to a task id. This is later used to create a task embedding using nn.Embedding()
        sample_task_id = self.task_name_to_id[sample_task_name]

        # Load image and convert it to a tensor
        sample = Image.open(sample_path)
        sample = self.totensor(sample)
        img_size = sample.shape[1]  # image is RGB and assumed square

        # NOTE: As per CVR code, we are not padding, but rather cropping the image.
        # One sample is 4 RGB (i.e., 3 channels) images of size (img_size, img_size), where img_size = 128
        pad = img_size - self.image_size
        sample = sample.reshape([3, img_size, 4, img_size]).permute([2,0,1,3])[:, :, pad//2:-pad//2, pad//2:-pad//2]

        # Perform the given transformation on the input image
        if self.transform is not None:
            sample = self.transform(sample)

        x, y = self.mix_input_and_create_artificial_labels(sample)

        return x, y, sample_task_id


class CVRDataModule(DataModuleBase):

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

        self.image_size = data_config.image_size

        train_set_path = data_config.dataset_dir + '/train.csv'
        val_set_path = data_config.dataset_dir + '/val.csv'
        test_set_path = data_config.dataset_dir + '/test.csv'

        if data_config.transform.enabled:
            # Compute the mean and std for normalization
            # dataset_mean_global, dataset_std_global, _, _ = compute_dataset_stats(train_set_path)

            # Use the mean and std normalization values from the CVR codebase
            # transform = self._transforms(dataset_mean_global, dataset_std_global)

            transform = self._transforms()
        else:
            transform = None

        # Create the torch Dataset objects that will then be used to create the dataloaders
        self.train_set = CVRDataset(train_set_path, image_size=self.image_size, transform=transform)
        self.val_set = CVRDataset(val_set_path, image_size=self.image_size, transform=transform)
        self.test_set = CVRDataset(test_set_path, image_size=self.image_size, transform=transform)

        if data_config.use_gen_test_set:
            gen_test_set_path = data_config.dataset_dir + '/gen_test.csv'
            self.gen_test_set = CVRDataset(gen_test_set_path, image_size=self.image_size, transform=transform)
            
            if data_config.validate_in_and_out_domain:
                gen_val_set_path = data_config.dataset_dir + '/gen_test.csv'   # NOTE: We make the choice to use gen_test_set to monitor the OOD validation performance during training because the official final results are reported on a new OOD test set
                self.gen_val_set = CVRDataset(gen_val_set_path, image_size=self.image_size, transform=transform)

    def _transforms(self, mean=0.9, std=0.1):
        # TODO: How did they choose mean of 0.9 and std of 0.1 ?
        # TODO: Check if the mean and std are computed and used correctly
        # NOTE:
        # mean: The average pixel value across the dataset
        # std: The standard deviation of pixel values across the dataset.
        transforms = tvt.Compose([
            tvt.Normalize((mean, mean, mean), (std, std, std)),
        ])
        return transforms
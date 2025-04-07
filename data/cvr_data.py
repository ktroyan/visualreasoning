import os
import json
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as tvt
import torch.nn.functional as F

# Personal codebase dependencies
from .data_base import DataModuleBase
from utility.cvr.utils import compute_dataset_stats
from utility.logging import logger


class CVRDataset(Dataset):
    def __init__(self, csv_dataset_path, image_size, transform=None):
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
        img_size = sample.shape[1]  # image is square

        ## My approach, thinking we were to handle padding and not cropping
        # pad = self.image_size - img_size
        # Ensure pad is non-negative
        # if pad < 0:
        #     logger.warning(f"Image size {img_size} is larger than target {self.image_size}. Setting number of pad tokens to use to 0.")
        #     pad = 0

        # One sample is 4 RGB (i.e., 3 channels) images of size (img_size, img_size), where img_size = 128
        # sample = F.pad(sample, (pad // 2, pad - pad // 2, pad // 2, pad - pad // 2))
        # sample = sample.reshape(3, self.image_size, 4, self.image_size).permute(2, 0, 1, 3)

        ## CVR original approach
        # TODO: They are not padding, but rather cropping the image. Why is that even though the image size set for generation was 128? See why.
        # One sample is 4 RGB (i.e., 3 channels) images of size (img_size, img_size), where img_size = 128
        pad = img_size - self.image_size
        sample = sample.reshape([3, img_size, 4, img_size]).permute([2,0,1,3])[:, :, pad//2:-pad//2, pad//2:-pad//2]


        # Perform the given transformation on the input image
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, sample_task_id


class CVRDataModule(DataModuleBase):

    def __init__(self, data_config, **kwargs):

        super().__init__(data_config.num_workers,
                         data_config.shuffle_train_dl,
                         data_config.train_batch_size, 
                         data_config.val_batch_size, 
                         data_config.test_batch_size, 
                         data_config.test_in_and_out_domain)    # pass the relevant arguments of the child class's __init__() method to the parent class __init__() method to make sure that the parent class is initialized properly

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
            gen_test_set_path = data_config.dataset_dir + '/test_gen.csv'
            self.gen_test_set = CVRDataset(gen_test_set_path, image_size=self.image_size, transform=transform)

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
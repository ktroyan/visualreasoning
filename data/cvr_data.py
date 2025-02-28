import os
import json
import pandas as pd
from PIL import Image
from torchvision import transforms as tvt
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

# Personal codebase dependencies
from utility.logging import logger


class DataModuleBase(LightningDataModule):

    def __init__(self, num_workers, train_batch_size, val_batch_size, test_batch_size, test_in_and_out_domain=False):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.test_in_and_out_domain = test_in_and_out_domain
        
        self.train_transform = None 
        self.test_transform = None

        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.gen_test_set = None

        self.num_workers = num_workers

    def train_dataloader(self):

        logger.info("Preparing training dataloader.")

        train_loader = DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            # drop_last=True,
            drop_last=False,
        )

        return train_loader

    def val_dataloader(self):
            
        logger.info("Preparing validation dataloader.")
        
        val_loader = DataLoader(
            self.val_set,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

        return val_loader

    # Here we handle the test AND gen_test dataloaders
    def test_dataloader(self):

        if self.test_in_and_out_domain:
            logger.info("Preparing test dataloader and sys-gen dataloader for testing.")

            test_dataloader = DataLoader(
                self.test_set,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True,
            )

            if self.gen_test_set is not None:
                gen_test_dataloader = DataLoader(
                    self.gen_test_set,
                    batch_size=self.test_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    persistent_workers=True,
                    pin_memory=True,
                )
            else:
                raise ValueError("The torch Dataset gen_test_set is None, so it cannot be used for testing.")

            return [test_dataloader, gen_test_dataloader]
        
        else:
            if self.gen_test_set is not None:
                logger.info("Preparing sys-gen test dataloader for testing.")

                gen_test_dataloader = DataLoader(
                    self.gen_test_set,
                    batch_size=self.test_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    persistent_workers=True,
                    pin_memory=True,
                )

                return gen_test_dataloader

            else:
                logger.info("Preparing test dataloader for testing.")

                test_dataloader = DataLoader(
                    self.test_set,
                    batch_size=self.test_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    persistent_workers=True,
                    pin_memory=True,
                )

            return test_dataloader

    @property
    def nb_train_data(self):
        assert self.train_set is not None, (f"Need to load train data before calling {self.nb_train_data.__name__}")
        return len(self.train_set)

    @property
    def nb_val_data(self):
        assert self.val_set is not None, (f"Need to load val data before calling {self.nb_val_data.__name__}")
        return len(self.val_set)

    @property
    def nb_test_data(self):
        assert self.test_set is not None, (f"Need to load test data before calling {self.nb_test_data.__name__}")
        return len(self.test_set)   # NOTE: we assume that the test set and the gen_test set have the same number of samples

    @property
    def nb_gen_test_data(self):
        assert self.gen_test_set is not None, (f"Need to load systematic generalization test data before calling {self.nb_gen_test_data.__name__}")
        return len(self.gen_test_set)

class CVRDataset(Dataset):
    def __init__(self, csv_dataset_path, image_size=128, transform=None):
        super().__init__()

        # Get the samples paths from the dataset csv metadata file
        self.samples_metadata = pd.read_csv(csv_dataset_path, header=0)
        self.samples_metadata['filepath'] = self.samples_metadata['filepath'].apply(lambda path: os.path.normpath(os.path.join("CVR", path)))

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

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, sample_task_id


class CVRDataModule(DataModuleBase):

    def __init__(
        self,
        dataset_dir,
        train_transform,
        test_transform,
        num_workers,
        train_batch_size,
        val_batch_size,
        test_batch_size,
        image_size=128,
        use_gen_test_set=False,
        test_in_and_out_domain=False,
        **kwargs,
    ):

        super().__init__(num_workers, train_batch_size, val_batch_size, test_batch_size, test_in_and_out_domain)    # pass the relevant arguments of the child class's __init__() method to the parent class __init__() method to make sure that the parent class is initialized properly

        _unmatched_args = kwargs

        transform = self._default_transforms()

        train_set_path = dataset_dir + '/train.csv'
        val_set_path = dataset_dir + '/val.csv'
        test_set_path = dataset_dir + '/test.csv'

        # Create the torch Dataset objects that will then be used to create the dataloaders
        self.train_set = CVRDataset(train_set_path, image_size=image_size, transform=transform)
        self.val_set = CVRDataset(val_set_path, image_size=image_size, transform=transform)
        self.test_set = CVRDataset(test_set_path, image_size=image_size, transform=transform)

        if use_gen_test_set:
            gen_test_set_path = dataset_dir + '/test_gen.csv'
            self.gen_test_set = CVRDataset(gen_test_set_path, image_size=image_size, transform=transform)

    def _default_transforms(self):
        transforms = tvt.Compose([
            tvt.Normalize((0.9, 0.9, 0.9), (0.1, 0.1, 0.1)),
        ])
        return transforms

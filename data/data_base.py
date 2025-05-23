from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

# Personal codebase dependencies
from utility.custom_logging import logger

class DataModuleBase(LightningDataModule):

    def __init__(self, num_workers, shuffle_train_dl, train_batch_size, val_batch_size, test_batch_size, use_gen_test_set=False, validate_in_and_out_domain=False):
        super().__init__()

        self.shuffle_train_dl = shuffle_train_dl
    
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

        self.use_gen_test_set = use_gen_test_set
        self.validate_in_and_out_domain = validate_in_and_out_domain

        # The torch dataset splits are created in the data environment's respective data module (e.g., BEFOREARCDataModule) 
        self.train_set = None
        self.val_set = None
        self.gen_val_set = None # NOTE: We make the choice to use gen_test_set to monitor the OOD validation performance during training because the official final results are reported on a new OOD test set
        self.test_set = None
        self.gen_test_set = None

        self.num_workers = num_workers

    def train_dataloader(self):

        logger.info("Preparing training dataloader.")

        train_loader = DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle_train_dl,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            # drop_last=True,
            drop_last=False,
        )

        logger.info("Done with training dataloader.")

        return train_loader

    def val_dataloader(self):
        """ We handle the in-domain validation set as well as the OOD validation set if applicable. """

        if not self.validate_in_and_out_domain:
            logger.info("Preparing validation dataloader.")
            
            val_loader = DataLoader(
                self.val_set,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True,
            )

            logger.info("Done with validation dataloader.")
        
        else:
            
            logger.info("Preparing in-domain validation dataloader and OOD validation dataloader.")

            val_loader = DataLoader(
                self.val_set,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True,
            )

            gen_val_loader = DataLoader(
                self.gen_val_set,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True,
            )

            logger.info("Done with in-domain validation dataloader and OOD validation dataloader.")

            return [val_loader, gen_val_loader]

        return val_loader

    def test_dataloader(self):
        """ We handle the in-domain test set as well as the OOD test set if applicable. """

        if not self.use_gen_test_set:
            logger.info("Preparing test dataloader.")

            test_dataloader = DataLoader(
                self.test_set,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True,
            )

            logger.info("Done with test dataloader.")

        else:
            
            logger.info("Preparing in-domain test dataloader and OOD test dataloader for testing.")

            test_dataloader = DataLoader(
                self.test_set,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True,
            )

            gen_test_dataloader = DataLoader(
                self.gen_test_set,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True,
            )

            logger.info("Done with in-domain test dataloader and OOD test dataloader for testing.")
            
            return [test_dataloader, gen_test_dataloader]
        
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
    def nb_gen_val_data(self):
        assert self.gen_val_set is not None, (f"Need to load systematic generalization val data before calling {self.nb_gen_val_data.__name__}")
        return len(self.gen_val_set)

    @property
    def nb_test_data(self):
        assert self.test_set is not None, (f"Need to load test data before calling {self.nb_test_data.__name__}")
        return len(self.test_set)

    @property
    def nb_gen_test_data(self):
        assert self.gen_test_set is not None, (f"Need to load systematic generalization test data before calling {self.nb_gen_test_data.__name__}")
        return len(self.gen_test_set)

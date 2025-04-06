from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

# Personal codebase dependencies
from utility.logging import logger

class DataModuleBase(LightningDataModule):

    def __init__(self, num_workers, shuffle_train_dl, train_batch_size, val_batch_size, test_batch_size, test_in_and_out_domain=False):
        super().__init__()

        self.shuffle_train_dl = shuffle_train_dl
    
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.test_in_and_out_domain = test_in_and_out_domain

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

            logger.info("Done with test dataloader and sys-gen test dataloader.")

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

                logger.info("Done with sys-gen test dataloader.")

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

                logger.info("Done with test dataloader.")

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

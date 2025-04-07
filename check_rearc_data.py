from omegaconf import OmegaConf

# Person codebase dependencies
from data.rearc_data import REARCDataModule
from utility.rearc.utils import check_train_test_contamination

if __name__ == "__main__":

    # Define the path to the dataset to check
    dataset_to_check = './REARC/final_datasets/sample-efficiency/exp_setting_1/experiment_4'
    # dataset_to_check = './REARC/check_data_experiment_1'

    # Manually create the data config as per the /configs/data.yaml config file
    data_config = {'num_workers': 4,
                    'dataset_dir': dataset_to_check,
                    'image_size': None,
                    'use_gen_test_set': False,
                    'transform': {'enabled': False},
                    'shuffle_train_dl': True,
                    'train_batch_size': 10,
                    'val_batch_size': 10,
                    'test_batch_size': 10,
                    'test_in_and_out_domain': False
                    }

    # Convert the data config dictionary to an OmegaConf object
    data_config = OmegaConf.create(data_config)

    # Create the data module
    datamodule = REARCDataModule(data_config)   # initialize the data with the data config
    print(f"Data module instantiated. Now showing the total number of samples per dataloader:\n{datamodule}\n")

    # Get the image size from the datamodule
    image_size = datamodule.image_size
    print(f"Max. image size considered (with padding): {image_size}")

    # Create train and test dataloader
    train_dataloader = datamodule.train_dataloader()    # dataloader with which the models are trained
    test_dataloader = datamodule.test_dataloader()      # dataloader with which the models are evaluated for benchmarking

    check_train_test_contamination(train_dataloader, test_dataloader)

    exit(0)
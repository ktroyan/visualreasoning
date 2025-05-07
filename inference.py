import os
import pprint
import time
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
import pytorch_lightning as pl


# Personal codebase dependencies
import data
import models
from utility.utils import get_complete_config, log_config_dict, get_model_from_ckpt, get_paper_model_name, process_test_results
from utility.rearc.utils import observe_rearc_input_output_images, check_train_test_contamination as check_rearc_train_test_contamination
from utility.logging import logger


torch.set_float32_matmul_precision('medium')    # 'high'
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def write_inference_results_logs(config, inference_folder, all_test_results, paper_model_name):
    # Format filename
    test_results_file_name = (
        f"{config.base.data_env.lower()}_"
        f"{config.experiment.study.replace('-', '')}_"
        f"{config.experiment.setting.replace('exp_setting_', 'es')}_"
        f"{config.experiment.name.replace('experiment_', 'exp')}_"
        f"{paper_model_name}_"
        "test_results.log"
    )

    test_results_file_path = os.path.join(inference_folder, test_results_file_name)

    # Write results with metadata
    with open(test_results_file_path, 'w') as f:
        # Log metadata
        f.write("*** Data ***\n")
        f.write(f"Data Environment: {config.base.data_env}\n")
        f.write(f"Study: {config.experiment.study}\n")
        f.write(f"Experiment Setting: {config.experiment.setting}\n")
        f.write(f"Experiment Name: {config.experiment.name}\n\n")

        # Log model name
        f.write(f"*** Model ***\n")
        full_model_name = f"{paper_model_name} + {config.model.head}"   # model name including network head
        f.write(f"Model: {full_model_name}\n\n")

        # Log test results
        f.write("*** Test Results ***\n")
        for result_set_name, result_dict in all_test_results.items():
            f.write(f"{result_set_name}:\n")
            for metric_name, metric_values in result_dict.items():
                values = np.array(metric_values)
                mean_value = values.mean()
                f.write(f"  {metric_name}:\n")
                f.write(f"    steps: {values.tolist()}\n")
                f.write(f"    epoch: {mean_value:.6f}\n")
            f.write("\n")

        # Log full config at the end of the log file
        f.write("\n\n*** Config ***\n")
        f.write(OmegaConf.to_yaml(config))
        f.write("\n")

    logger.info(f"Test results saved to: {test_results_file_path}")

def main(config, inference_folder, datamodule, model=None, model_ckpt_path=None, exp_logger=None):

    logger.info("*** Inference started ***")
    inference_start_time = time.time()

    # Trainer (for inference/testing)
    trainer = pl.Trainer(num_nodes=1,   # number of gpu nodes for distributed training
                         logger=exp_logger,
                         devices=config.base.n_gpus,
                         accelerator='auto',
                         enable_progress_bar=True,
                        )

    # Model
    if model is not None:
        log_message = f"A specific (most likely trained) model instance from class {model.__class__} was given for inference.\n\n"
        # log_message += f"Model: {model}\n"
        log_message += "No model checkpoint was given directly."
        logger.info(log_message)
    
    elif model_ckpt_path is not None:
        model = get_model_from_ckpt(model_ckpt_path)
        log_message = f"Model loaded from checkpoint for inference: {model_ckpt_path}\n\n"
        # log_message += f"Model: {model}\n"
        logger.info(log_message)

    else:
        raise ValueError("No model instance or model checkpoint path was given for inference.")


    logger.info(f"All hyperparameters of the model module used for inference:\n{model.hparams} \n")

    # Sanity check for data contamination between train and test dataloaders
    if config.inference.check_data_contamination and config.base.data_env in ["REARC", "BEFOREARC"]:
        train_dataloader = datamodule.train_dataloader()
        
        if isinstance(datamodule.test_dataloader(), list):
            test_dataloader = datamodule.test_dataloader()[0]       # get in-domain test dataloader
            check_rearc_train_test_contamination(train_dataloader, test_dataloader)
            gen_test_dataloader = datamodule.test_dataloader()[1]   # get OOD test dataloader
            check_rearc_train_test_contamination(train_dataloader, gen_test_dataloader)

        else:
            test_dataloader = datamodule.test_dataloader()
            check_rearc_train_test_contamination(train_dataloader, test_dataloader)


    # Testing
    trainer.test(model=model, datamodule=datamodule, verbose=True)  # NOTE: if more than one val/test dataloader was created in the datamodule, all the val/test dataloaders will be used for validation/testing
   

    # Additional logging and plotting if needed
    if config.inference.inference_verbose == 1 and config.base.data_env in ["REARC", "BEFOREARC"]:
        logger.debug(f"Test predictions: {model.test_preds}")
        logger.debug(f"Test targets: {model.test_targets}")

        test_dataloader = datamodule.test_dataloader()[0] if isinstance(datamodule.test_dataloader(), list) else datamodule.test_dataloader()   # get in-domain test dataloader
        observe_rearc_input_output_images(save_folder_path=inference_folder, dataloader=test_dataloader, split="test", batch_id=0, n_samples=4)

        if config.data.use_gen_test_set:
            logger.debug(f"OOD test predictions: {model.gen_test_preds}")
            logger.debug(f"OOD test targets: {model.gen_test_targets}")

            gen_test_dataloader = datamodule.test_dataloader()[1]   # get OOD test dataloader
            observe_rearc_input_output_images(save_folder_path=inference_folder, dataloader=gen_test_dataloader, split="gen_test", batch_id=0, n_samples=4)

            if config.data.validate_in_and_out_domain:
                logger.debug(f"OOD val predictions: {model.gen_val_preds}")
                logger.debug(f"OOD val targets: {model.gen_val_targets}")

                gen_val_dataloader = datamodule.val_dataloader()[1] # get OOD val dataloader
                observe_rearc_input_output_images(save_folder_path=inference_folder, dataloader=gen_val_dataloader, split="gen_val", batch_id=0, n_samples=4)        

    ## Test results
    test_results = model.test_results
    all_test_results = {'test_results': test_results}

    if config.data.use_gen_test_set:
        gen_test_results = model.gen_test_results
        all_test_results.update({'gen_test_results': gen_test_results})

    paper_model_name = get_paper_model_name(config)

    # Save test results relevant to the paper in a log file
    write_inference_results_logs(config, inference_folder, all_test_results, paper_model_name)

    # End of inference
    log_message = "*** Inference ended ***\n"
    inference_elapsed_time = time.time() - inference_start_time
    log_message += f"\nTotal inference time: \n{inference_elapsed_time} seconds ~=\n{inference_elapsed_time/60} minutes ~=\n{inference_elapsed_time/(60*60)} hours"
    logger.info(log_message)

    return all_test_results


if __name__ == '__main__':
    logger.info(f"inference.py process ID: {os.getpid()}")

    # Get and log all the config arguments
    config, _ = get_complete_config()
    log_config_dict(config)

    # Seed everything for reproducibility
    if config.base.seed is not None:
        pl.seed_everything(config.base.seed)

    # Create the inference folder
    inference_folder = f"./{config.data.data_env}/inference"
    os.makedirs(inference_folder, exist_ok=True)

    # Data chosen
    data_module = vars(data)[config.base.data_module]
    datamodule = data_module(config.data)   # initialize the data with the data config
    logger.info(f"Data module instantiated. Now showing the total number of samples per dataloader:\n{datamodule}\n")

    # Get the image size from the datamodule. Useful for the model backbone
    image_size = datamodule.image_size
    logger.info(f"Image size considered (with padding): {image_size}")

    # Model chosen
    model_ckpt = config.inference.inference_model_ckpt
    if model_ckpt is not None and model_ckpt != "":
        logger.info(f"Model checkpoint path chosen for inference: {model_ckpt}")
        model = None
    else:
        model_module = vars(models)[config.base.model_module]
        model = model_module(config.base, config.model, config.data, config.backbone_network, config.head_network, image_size, inference_folder)   # initialize the model with the model and network configs
        logger.info(f"Model chosen for inference w.r.t. the current config files: {model}")

    test_results = main(config, inference_folder, datamodule, model, model_ckpt, exp_logger=None)

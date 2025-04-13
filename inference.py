import os
import time
import pandas as pd
import torch
import pytorch_lightning as pl


# Personal codebase dependencies
import data
import models
from utility.utils import get_complete_config, log_config_dict, get_model_from_ckpt, observe_input_output_images, process_test_results
from utility.rearc.utils import check_train_test_contamination as check_rearc_train_test_contamination
from utility.logging import logger


torch.set_float32_matmul_precision('medium')    # 'high'
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def main(config, datamodule, model=None, model_ckpt_path=None, exp_logger=None):

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


    logger.info(f"All hyperparameters of the model used for inference:\n{model.hparams} \n")

    # Sanity check for data contamination between train and test dataloaders
    if config.inference.check_data_contamination:
        train_dataloader = datamodule.train_dataloader()
        test_dataloader = datamodule.test_dataloader()[0] if isinstance(datamodule.test_dataloader(), list) else datamodule.test_dataloader()
        check_rearc_train_test_contamination(train_dataloader, test_dataloader)


    # Testing
    trainer.test(model=model, datamodule=datamodule, verbose=True)  # NOTE: if more than one test dataloader was created in the datamodule, all the test dataloaders will be used for testing
   

    # Additional logging and plotting if needed
    if config.inference.inference_verbose == 1 and config.base.data_env in ["REARC", "BEFOREARC"]:
        # TODO: Implement for other data environments

        logger.debug(f"Test predictions: {model.test_preds}")
        logger.debug(f"Test targets: {model.test_targets}")

        test_dataloader = datamodule.test_dataloader()[0] if isinstance(datamodule.test_dataloader(), list) else datamodule.test_dataloader()
        observe_input_output_images(dataloader=test_dataloader, batch_id=0, n_samples=4, split="test")

        if config.data.use_gen_test_set:
            logger.debug(f"Sys-gen test predictions: {model.gen_test_preds}")
            logger.debug(f"Sys-gen test targets: {model.gen_test_targets}")

            gen_test_dataloader = datamodule.test_dataloader()[1]
            observe_input_output_images(dataloader=gen_test_dataloader, batch_id=0, n_samples=4, split="gen_test")            

    # Process the test results for better logging
    test_results = model.test_results
    processed_test_results = process_test_results(config, test_results, test_type="test", exp_logger=None)

    if config.data.use_gen_test_set:
        gen_test_results = model.gen_test_results
        processed_gen_test_results = process_test_results(config, gen_test_results, test_type="gen_test", exp_logger=None)
    else:
        processed_gen_test_results = {}

    all_test_results = {'test_results': processed_test_results, 'gen_test_results': processed_gen_test_results}

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
        model = model_module(config.base, config.model, config.backbone_network, config.head_network, image_size)   # initialize the model with the model and network configs
        logger.info(f"Model chosen for inference w.r.t. the current config files: {model}")
    

    test_results = main(config, datamodule, model, model_ckpt, exp_logger=None)

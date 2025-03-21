import os
import time
import pandas as pd
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns


# Personal codebase dependencies
import data
import models
from utility.utils import get_complete_config, log_config_dict, get_model_from_ckpt
from utility.logging import logger

torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


# FIXME: for this code to work currently, the batch size should yield a number of elements in each key of results so that it is a multiple of the number of tasks, otherwise the reshape will fail. Fix it.
# Also see how to handle the case where the results are for multiple test dataloaders
def process_test_results(config, test_results, test_type="test", exp_logger=None):

    # Just handle some naming consistency issue with the generated data splits that are named "test_gen" instead of "gen_test"
    if test_type == "gen_test":
        test_set_path = f"{config.data.dataset_dir}/{'_'.join(reversed(test_type.split('_')))}"
    else:
        test_set_path = f"{config.data.dataset_dir}/{test_type}"

    # Check if the folder config.data.dataset_dir contains test_set_path with .csv or .json extension and load it accordingly with pandas
    if os.path.exists(f"{test_set_path}.csv"):
        test_set = pd.read_csv(f"{test_set_path}.csv")
    elif os.path.exists(f"{test_set_path}.json"):
        test_set = pd.read_json(f"{test_set_path}.json")
    else:
        raise FileNotFoundError(f"Test set file not found in the dataset directory: {test_set_path}")

    # Processing of the results and wrap-up of the parameters used
    tasks_considered = test_set['task'].unique()

    logger.info(f"Post-processing results for tasks: {tasks_considered}")

    global_avg = {metric_key: r.mean() for metric_key, r in test_results.items()}
    per_task = {}
    per_task_avg = {}

    # nb_of_tasks = len(tasks_considered)

    for metric_key, r in test_results.items():
        logger.info(f"Processing results for metric key: {metric_key}, and result with shape: {r.shape}")
        k_result = r.reshape([len(tasks_considered), -1])
        for i, task in enumerate(tasks_considered):
            if 'task' not in task:
                per_task[f'{metric_key}_task_{task}'] = k_result[i]
                per_task_avg[f'{metric_key}_task_{task}'] = k_result[i].mean()
                
            else:
                per_task[f'{metric_key}_{task}'] = k_result[i]
                per_task_avg[f'{metric_key}_{task}'] = k_result[i].mean()
    
    logger.info(f"Global average results:\n{global_avg}")
    
    log_message = "Per task results for each batch/step:\n"
    for key, value in per_task.items():
        log_message += f"{key}: {value}\n"
    logger.info(log_message)
    
    logger.info(f"Per task average results:\n{per_task_avg}")

    if exp_logger is not None:
        exp_logger.experiment.log({f"{test_type}_results_global_avg/{k}": v for k, v in global_avg.items()})
        exp_logger.experiment.log({f"{test_type}_results_per_task_avg/{k}": v for k, v in per_task_avg.items()})
        # exp_logger.experiment.log({f"{test_type}_results_per_task/{k}": v for k, v in per_task.items()})   # TODO: check how to to log it properly if needed
        exp_logger.save()

    processed_test_results = {
        f"{test_type}_results_global_avg": global_avg,
        f"{test_type}_results_per_task": per_task,
        f"{test_type}_results_per_task_avg": per_task_avg
    }

    return processed_test_results


def observe_input_output_images(dataloader, batch_id=0, n_samples=4, split="test"):
    
    # Get the batch batch_id from the dataloader
    for i, batch in enumerate(dataloader):
        if i == batch_id:
            break
    
    # Get the input and output images
    inputs, outputs = batch[0], batch[1]

    # Number of samples to observe
    n_samples = min(n_samples, len(inputs))

    logger.debug(f"Observing {n_samples} samples from {split} batch {batch_id}. See /figs folder.")

    # Handle padding tokens. Replace the symbols for pad tokens with the background color
    # TODO: How to handle the pad token properly? For example if its decided value is not 10.0 anymore as it was changed in the other parts of the code?
    pad_token = 10
    inputs[inputs == pad_token] = 0
    outputs[outputs == pad_token] = 0

    # Use the same color map as REARC
    cmap = ListedColormap([
        '#000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
    
    vmin = 0
    vmax = 9

    # Create a figure to plot the samples
    fig, axs = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6), dpi=150)

    for i in range(n_samples):
        input_img = inputs[i].cpu().numpy()
        target_img = outputs[i].cpu().numpy()

        for ax, img, title in zip([axs[0, i], axs[1, i]], 
                                  [input_img, target_img], 
                                  [f"Input {i} of batch {batch_id}", f"Output {i} of batch {batch_id}"]
                                  ):
            sns.heatmap(img, ax=ax, cbar=False, linewidths=0.05, linecolor='gray', square=True, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"{split} batch {batch_id}", fontsize=18)

    plt.tight_layout()
    # plt.show()

    # Save the figure
    os.makedirs("./figs", exist_ok=True)   # create the /figs folder if it does not exist
    fig.savefig(f"./figs/{split}_image_input_output_batch{batch_id}.png")

    plt.close(fig)


def main(config, datamodule, model=None, model_ckpt_path=None, exp_logger=None):

    logger.info("*** Inference started ***")
    inference_start_time = time.time()

    trainer = pl.Trainer(num_nodes=1,   # number of gpu nodes for distributed training
                         logger=exp_logger,
                         devices=config.base.gpus,
                         accelerator='auto',
                         enable_progress_bar=True,
                        )

    # Model
    if model is not None:
        log_message = f"A specific model instance from class {model.__class__} was given for inference.\n\n"
        # log_message += f"Model: {model}\n"
        log_message += "No model checkpoint is used."
        logger.info(log_message)
    
    elif model_ckpt_path is not None:
        model = get_model_from_ckpt(model_ckpt_path)
        log_message = f"Model loaded from checkpoint for inference: {model_ckpt_path}\n\n"
        # log_message += f"Model: {model}\n"
        logger.info(log_message)

    else:
        raise ValueError("No model instance or model checkpoint path was given for inference.")


    logger.info(f"All hyperparameters of the model used for inference:\n{model.hparams} \n")

    # Testing
    trainer.test(model=model, datamodule=datamodule, verbose=True)  # NOTE: if more than one test dataloader was created in the datamodule, all the test dataloaders will be used for testing
   
    if config.inference.inference_verbose == 1:
        logger.debug(f"Test predictions: {model.test_preds}")
        logger.debug(f"Test targets: {model.test_targets}")

        # TODO: Make it work or skip it for CVR
        test_dataloader = datamodule.test_dataloader()[0] if isinstance(datamodule.test_dataloader(), list) else datamodule.test_dataloader()
        observe_input_output_images(dataloader=test_dataloader, batch_id=0, n_samples=4, split="test")

        if config.data.use_gen_test_set:
            logger.debug(f"Sys-gen test predictions: {model.gen_test_preds}")
            logger.debug(f"Sys-gen test targets: {model.gen_test_targets}")

            # TODO: Make it work or skip it for CVR
            gen_test_dataloader = datamodule.test_dataloader()[1]
            observe_input_output_images(dataloader=gen_test_dataloader, batch_id=0, n_samples=4, split="gen_test")            

    test_results = model.test_results
    processed_test_results = process_test_results(config, test_results, test_type="test", exp_logger=None)

    if config.data.use_gen_test_set:
        gen_test_results = model.gen_test_results
        processed_gen_test_results = process_test_results(config, gen_test_results, test_type="gen_test", exp_logger=None)
    else:
        processed_gen_test_results = {}

    all_test_results = {'test_results': processed_test_results, 'gen_test_results': processed_gen_test_results}

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
    logger.info(f"Max. image size considered (with padding): {image_size}")

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

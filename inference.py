import os
import sys
import time
import argparse
import pandas as pd
import torch
import pytorch_lightning as pl

# Personal codebase dependencies
import data
import models
from utility.utils import parse_args_and_configs, log_args_namespace, get_config_specific_args_from_args
from utility.logging import logger

torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# FIXME: for this code to work currently, the batch size should yield a number of elements in each key of results so that it is a multiple of the number of tasks, otherwise the reshape will fail. Fix it.
# Also see how to handle the case where the results are for multiple test dataloaders
def process_test_results(args, test_results, test_type="test", exp_logger=None):

    # Just handle some naming consistency issue with the generated data splits that are named "test_gen" instead of "gen_test"
    if test_type == "gen_test":
        test_set = pd.read_csv(f"{args.dataset_dir}/{'_'.join(reversed(test_type.split('_')))}.csv")
    
    else:
        test_set = pd.read_csv(f"{args.dataset_dir}/{test_type}.csv")
    
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
    
    log_message = "Per task results:\n"
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

def main(args, datamodule, model, model_ckpt_path=None, exp_logger=None):

    logger.info("*** Inference started ***")
    inference_start_time = time.time()

    trainer = pl.Trainer(num_nodes=1,
                         logger=exp_logger, 
                         devices=args.gpus,
                         accelerator='auto',
                         enable_progress_bar=True,
                         log_every_n_steps=args.log_every_n_steps,
                        )

    # Model
    if model_ckpt_path is not None:
        model_module = model.__class__
        model = model_module.load_from_checkpoint(model_ckpt_path)
        log_message = f"Model loaded from checkpoint for inference: {model_ckpt_path}\n\n"
        # log_message += f"Model: {model}\n"
        logger.info(log_message)

    else:
        log_message = f"Model instance from class {model.__class__} was given for inference.\n\n"
        # log_message += f"Model: {model}\n"
        logger.info(log_message)

    logger.info(f"Model hyperparameters of the model used for inference:\n{model.hparams} \n")

    # Testing
    trainer.test(model=model, datamodule=datamodule, verbose=True)  # NOTE: if more than one test dataloader was created in the datamodule, all the test dataloaders will be used for testing
   
    if args.inference_verbose == 1:
        logger.info(f"Test predictions: {model.test_preds}")
        logger.info(f"Test labels: {model.test_labels}")

        if args.use_gen_test_set:
            logger.info(f"Sys-gen test predictions: {model.gen_test_preds}")
            logger.info(f"Sys-gen test labels: {model.gen_test_labels}")

    test_results = model.test_results
    processed_test_results = process_test_results(args, test_results, test_type="test", exp_logger=None)

    if args.use_gen_test_set:
        gen_test_results = model.gen_test_results
        processed_gen_test_results = process_test_results(args, gen_test_results, test_type="gen_test", exp_logger=None)
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

    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()

    # Frequently changing CLI arguments
    parser.add_argument('--seed', type=int, default=None, help='seed for reproducibility')
    
    # Consistent CLI arguments
    parser.add_argument("--general_config", default="./configs/general.yaml", help="from where to load the general YAML config", metavar="FILE")
    parser.add_argument("--data_config", default="./configs/data.yaml", help="from where to load the YAML config of the chosen data", metavar="FILE")
    parser.add_argument("--model_config", default="./configs/model.yaml", help="from where to load the YAML config of the chosen model", metavar="FILE")
    args = parse_args_and_configs(parser, argv)
    parser.add_argument("--network_config", default=f"./configs/networks/{args.model_backbone}.yaml", help="from where to load the YAML config of the chosen neural network", metavar="FILE")
    parser.add_argument("--inference_config", default="./configs/inference.yaml", help="from where to load the inference YAML config", metavar="FILE")
    args = parse_args_and_configs(parser, argv)

    # Log all the arguments in the Namespace
    log_args_namespace(args)

    # Seed everything for reproducibility
    if args.seed is not None:
        pl.seed_everything(args.seed)

    # Data chosen
    data_module = vars(data)[args.data_module]
    logger.info(f"Data module: {data_module}")
    data_args = get_config_specific_args_from_args(args, args.data_config)
    datamodule = data_module(**data_args)   # initializing the data
    logger.info(f"Datamodule (showing the total number of samples per dataloader):\n {datamodule} \n")

    # Model chosen
    model_module = vars(models)[args.model_module]
    logger.info(f"Model module: {model_module}")
    model_args = get_config_specific_args_from_args(args, args.model_config)
    model = model_module(**model_args)   # initializing the model

    # NOTE: the model initialized should match the model checkpoint used for inference

    test_results = main(args, datamodule, model, args.inference_model_ckpt_path, exp_logger=None)

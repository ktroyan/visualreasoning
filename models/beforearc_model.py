import os
import torch
from torch import nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl

# Personal codebase dependencies
from networks.backbones.resnet import get_resnet
from networks.backbones.transformer import get_transformer_encoder
from networks.backbones.vit import get_vit
from networks.backbones.looped_vit import get_looped_vit
from networks.heads.mlp import get_mlp_head
from networks.heads.transformer import get_transformer_decoder
from networks.heads.xtransformer import get_xtransformer_decoder
from networks.heads.mytransformer import get_mytransformer_decoder
from utility.utils import plot_lr_schedule
from utility.rearc.utils import plot_image_predictions, plot_attention_scores   # TODO: See if need to adapt
from utility.logging import logger

# os.environ['TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS'] = '1'


class VisReasModel(pl.LightningModule):
    """
    Model module class that handles the training, validation and testing logic of the model.
    """
    
    def __init__(self, base_config, model_config, data_config, backbone_network_config, image_size, save_folder):
        super().__init__()

        self.base_config = base_config
        self.model_config = model_config
        self.data_config = data_config
        self.backbone_network_config = backbone_network_config
        self.image_size = image_size
        self.save_folder = save_folder

        # Train, val, test, OOD test (if applicable) and OOD val (if applicable) inputs, predictions and targets for local observation
        self.train_inputs = []
        self.train_preds = []
        self.train_targets = []

        self.val_inputs = []
        self.val_preds = []
        self.val_targets = []

        if data_config.validate_in_and_out_domain:
            self.gen_val_inputs = []
            self.gen_val_preds = []
            self.gen_val_targets = []

        self.test_inputs = []
        self.test_preds = []
        self.test_targets = []

        if data_config.use_gen_test_set:
            self.gen_test_inputs = []
            self.gen_test_preds = []
            self.gen_test_targets = []

        # Metrics for local plotting
        self.train_loss_step = []
        self.train_acc_step = []
        self.train_grid_acc_step = []

        self.val_loss_step = []
        self.val_acc_step = []
        self.val_grid_acc_step = []

        if data_config.validate_in_and_out_domain:
            self.gen_val_loss_step = []
            self.gen_val_acc_step = []
            self.gen_val_grid_acc_step = []

        # Test and OOD test (if applicable) results for logging
        self.test_step_results = [] # NOTE: this is needed to store the results of the test step for each batch (i.e., at each step), and display the final results at the end of the epoch
        if data_config.use_gen_test_set:
            self.gen_test_step_results = [] # NOTE: this is needed to store the results of the generalization test step for each batch (i.e., at each step), and display the final results at the end of the epoch

        # Learning rate values for plotting LR schedule
        self.lr_values = []

        # Store the attention scores
        if model_config.attention_map.enabled:
            self.train_attention_scores = []
            self.val_attention_scores = []

            if data_config.validate_in_and_out_domain:
                self.gen_val_attention_scores = []


    def load_backbone_weights(self, checkpoint_path):
        self.model_backbone.load_state_dict(torch.load(checkpoint_path, weights_only=False)['model'], strict=False)
        logger.info(f"Loaded ckpt weights for backbone at ckpt path: {checkpoint_path}")

    def freeze_backbone_weights(self):
        for param in self.model_backbone.parameters():
            param.requires_grad = False

    def create_true_size_mask(self, B: int, H: int, W: int, y_true_size: torch.Tensor) -> torch.Tensor:
        """ 
        Create a multiplicative mask to ignore all the symbols outside of the true size of the grid image.
        The non-symbol (e.g., padding) tokens would thus be ignored when computing the metrics (e.g., loss, accuracy). 
        
        Vectorized and avoid graph breaks by staying in tensor ops.
        """

        # Create a mask of shape [B, H, W] initialized to False (0)
        mask = torch.zeros((B, H, W), dtype=torch.bool, device=self.device)  # [B, H, W] ; initialize all as 0/False (padded)

        # Get row and column indices for the true size grid
        row_idx = torch.arange(H, device=self.device).view(1, H, 1)  # [1, H, 1]
        col_idx = torch.arange(W, device=self.device).view(1, 1, W)  # [1, 1, W]

        # Broadcast true sizes to be compatible with the row and column indices of the full grid
        true_h = y_true_size[:, 0].view(B, 1, 1)  # [B, 1, 1] <-- [B, 1] (height of the grid)
        true_w = y_true_size[:, 1].view(B, 1, 1)  # [B, 1, 1] <-- [B, 1] (width of the grid)

        # Get mask with True where (row < true_h) and (col < true_w)
        mask = (row_idx < true_h) & (col_idx < true_w)  # [B, H, W] <-- [B, H, 1] boolean mask for the height and [B, 1, W] boolean mask for the width

        # Flatten 2D mask
        mask = mask.view(B, -1) # [B, seq_len=H*W] <-- [B, H, W] (where the rectangle [0:true_h, 0:true_w] is True)

        return mask

    def shared_step(self, batch):
        """
        The same code is shared between training, validation and test steps. 
        In theory, we would not need to compute and return the loss for testing, hence we would only call shared_step() in the test_step() method. 
        However, we still do it, so the purpose of this shared_step() is lesser.
        """

        x, y, task_tokens_seq, example_in_context, y_true_size, x_grid_object_ids, special_grid_tokens_dict = batch   # [B, H, W], [B, H, W], [B], [B, 2], [B, seq_len], Dict

        B, H, W = x.shape

        # Flatten 2D tensor (grid image) y
        y = y.view(B, -1)  # [B, seq_len=H*W] <-- [B, H, W]

        # Handle input grid object ids for OPE
        if not self.model_config.ope.enabled:
            x_grid_object_ids = None

        # Handle task embedding
        if not self.model_config.task_embedding.enabled:
            task_tokens_seq = None
            example_in_context = None
        
        elif self.model_config.task_embedding.approach == 'example_in_context':
            task_tokens_seq = None
        
        elif self.model_config.task_embedding.approach == 'task_tokens':
            example_in_context = None

        
        # Forward pass through the whole model
        y_hat = self(x, y, task_tokens_seq=task_tokens_seq, example_in_context=example_in_context, x_grid_object_ids=x_grid_object_ids)  # computed logits

        # Permute the dimensions of y_hat to be [B, num_classes, seq_len] instead of [B, seq_len, num_classes] to match PyTorch's cross_entropy function format
        y_hat = y_hat.permute(0, 2, 1)  # [B, num_classes, seq_len] <-- [B, seq_len, num_classes]

        # Create the multiplicative mask based on the true sizes of y to only compute the metrics w.r.t. the actual tokens to predict in the target
        true_size_mask = self.create_true_size_mask(B, H, W, y_true_size)

        return x, y_hat, y, true_size_mask, special_grid_tokens_dict

    def step(self, batch, batch_idx):

        x, y_hat, y, mask, special_grid_tokens_dict = self.shared_step(batch)    # [B, num_classes, seq_len], [B, seq_len], [B, seq_len]

        B, H, W = x.shape
        B, seq_len = y.shape

        # probabilities = F.softmax(y_hat, dim=1)  # compute the probabilities (normalized logits) of the model for each sample of the batch

        # TODO: 
        # See exactly how we want to compute the loss and metrics. What types of tokens we want to consider.
        # Also, convert the non-data tokens to background tokens (i.e., 0) when computing the loss and metrics, no?
        # Also, consider weighting the tokens (e.g., there are many more bakground tokens) differently when computing the loss for grid with padding.

        # Loss per symbol (with all sorts of padding considered): compute the loss per token/symbol
        per_sample_loss = F.cross_entropy(y_hat, y.long(), reduction='none').float()  # [B, seq_len]
        loss_symbol_with_pad = (per_sample_loss.mean()).unsqueeze(0)

        # Loss per symbol (without padding): compute the loss per token/symbol and then apply the mask to ignore the padding tokens
        per_sample_loss = F.cross_entropy(y_hat, y.long(), reduction='none').float()  # [B, seq_len]
        loss_symbol_no_pad = ((per_sample_loss * mask).sum() / mask.sum()).unsqueeze(0)  # only consider non-padding elements

        # Compute predictions
        preds = torch.argmax(y_hat, dim=1)  # [B, seq_len]; predictions for each token/symbol of the model for each sample of the batch

        # Accuracy per symbol (with padding) (i.e., the accuracy of the model in predicting the correct symbol for each pixel of the grid considering the whole max. padded grid, thus also the padding tokens)
        acc_symbol_with_pad = (preds == y).float().mean().unsqueeze(0)

        # Accuracy per symbol (without padding) (i.e., the accuracy of the model in predicting the correct symbol for each pixel of the grid considering only the target grid, that is, without considering the padding tokens)
        acc_symbol_no_pad = (((preds == y) * mask).sum().float() / mask.sum()).unsqueeze(0)  # only consider non-padding elements

        # Grid accuracy (only count as correct if the entire padded grid is correct)
        acc_grid_with_pad = torch.all(preds == y, dim=1).float().mean().unsqueeze(0)

        # Grid accuracy (only count as correct if entire non-padding grid is correct)
        acc_grid_no_pad = torch.all((preds == y) | ~mask, dim=1).float().mean().unsqueeze(0)   # | ~mask ensures automatically count as correct the padding tokens

        logs = {'loss': loss_symbol_with_pad, 
                'loss_no_pad': loss_symbol_no_pad, 
                'acc': acc_symbol_with_pad,
                'acc_no_pad': acc_symbol_no_pad, 
                'acc_grid': acc_grid_with_pad, 
                'acc_grid_no_pad': acc_grid_no_pad
                }
        
        loss = loss_symbol_with_pad

        return loss, logs, preds

    def training_step(self, batch, batch_idx):
        """
        This method is called for each batch during the training phase.
        This is a default PyTorch Lightning method that we override to define the training logic.
        """
        x, y, task_tokens_seq, example_in_context, y_true_size, x_grid_object_ids, special_grid_tokens_dict = batch

        B, H, W = x.shape

        loss, logs, preds = self.step(batch, batch_idx)

        # Logging
        self.log_dict({f"metrics/train_{k}": v for k,v in logs.items()}, 
                      prog_bar=True, 
                      logger=True, 
                      on_step=True, 
                      on_epoch=True, 
                      add_dataloader_idx=False
                      )
        
        # For the first and last training batch of the epoch
        if (batch_idx == 0) or (batch_idx == self.trainer.num_training_batches - 1):
            # Save batch (of inputs, preds, targets) for current epoch for plotting
            self.train_inputs.append(x)
            self.train_preds.append(preds)
            self.train_targets.append(y)

            if self.model_config.attention_map.enabled:
                # Store attention scores
                if hasattr(self.encoder, 'get_attention_scores'):
                    attn_scores = self.encoder.get_attention_scores()
                    if attn_scores is not None:
                        self.train_attention_scores.append(attn_scores)
                    else:
                        logger.warning(f"Attention scores were None for train epoch {self.current_epoch} and batch {batch_idx}.")

        # Store to plot locally
        self.train_loss_step.append(logs['loss'])
        self.train_acc_step.append(logs['acc'])
        self.train_grid_acc_step.append(logs['acc_grid'])

        # Log the current learning rate
        self.log_dict({"learning_rate": self.lr_schedulers().get_last_lr()[-1]}, 
                    prog_bar=True, 
                    logger=True, 
                    on_step=True, 
                    on_epoch=True, 
                    )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        This method is called for each batch during the validation phase.
        This is a default PyTorch Lightning method that we override to define the validation logic.

        NOTE: Currently val_loss is the monitored metric during training
        """

        x, y, task_tokens_seq, example_in_context, y_true_size, x_grid_object_ids, special_grid_tokens_dict = batch

        B, H, W = x.shape

        loss, logs, preds = self.step(batch, batch_idx)

        if dataloader_idx == 0:
            # Logging
            self.log_dict({f"metrics/val_{k}": v for k, v in logs.items()}, 
                          prog_bar=True, 
                          logger=True, 
                          on_step=True, 
                          on_epoch=True,
                          add_dataloader_idx=False
                          )    # NOTE: this is monitored for best checkpoint and early stopping

            # Save to plot locally
            self.val_loss_step.append(logs['loss'])
            self.val_acc_step.append(logs['acc'])
            self.val_grid_acc_step.append(logs['acc_grid'])

            # For the first and last validation batch of the epoch
            if (batch_idx == 0) or (batch_idx == self.trainer.num_val_batches[0] - 1):
                
                # Store batch (of inputs, preds, targets) for current epoch for plotting
                self.val_inputs.append(x)
                self.val_preds.append(preds)
                self.val_targets.append(y)

                if self.model_config.attention_map.enabled:
                    # Store attention scores
                    if hasattr(self.encoder, 'get_attention_scores'):
                        attn_scores = self.encoder.get_attention_scores()
                        if attn_scores is not None:
                            self.val_attention_scores.append(attn_scores)
                        else:
                            logger.warning(f"Attention scores were None for val epoch {self.current_epoch} and batch {batch_idx}.")
            
        elif dataloader_idx == 1:
            # Logging
            self.log_dict({f"metrics/gen_val_{k}": v for k, v in logs.items()}, 
                          prog_bar=True, 
                          logger=True, 
                          on_step=True, 
                          on_epoch=True,
                          add_dataloader_idx=False
                          )

            # Save to plot locally
            self.gen_val_loss_step.append(logs['loss'])
            self.gen_val_acc_step.append(logs['acc'])
            self.gen_val_grid_acc_step.append(logs['acc_grid'])

            # For the first and last validation batch of the epoch
            if (batch_idx == 0) or (batch_idx == self.trainer.num_val_batches[1] - 1):
                
                # Store batch (of inputs, preds, targets) for current epoch for plotting
                self.gen_val_inputs.append(x)
                self.gen_val_preds.append(preds)
                self.gen_val_targets.append(y)

                if self.model_config.attention_map.enabled:
                    # Store attention scores
                    if hasattr(self.encoder, 'get_attention_scores'):
                        attn_scores = self.encoder.get_attention_scores()
                        if attn_scores is not None:
                            self.gen_val_attention_scores.append(attn_scores)
                        else:
                            logger.warning(f"Attention scores were None for gen val epoch {self.current_epoch} and batch {batch_idx}.")

        return loss

    def on_train_epoch_end(self):
        """
        This method is called at the end of each training epoch.
        This is a default PyTorch Lightning method that we override to define the logic at the end of each training epoch.
        NOTE: It is called after the on_train_epoch_end() method of the Callback class.
        """

        figs_to_log = []

        # Plot attention maps if attention maps are enabled and exist
        if self.model_config.attention_map.enabled and hasattr(self.encoder, 'get_attention_scores'):
            # Plot attention maps of some training and validation samples of the first and last batch seen during the epoch
            
            for batch_index, split in zip([0, -1], ["train", "val"]):
                # Plot attention maps of some training and validation samples of the first and last batch seen during the epoch
                fig_paths = plot_attention_scores(self.save_folder,
                                                  split, 
                                                  self.train_inputs,self.train_targets, 
                                                  self.train_attention_scores,
                                                  self.model_config.attention_map.layer, 
                                                  self.backbone_network_config.num_heads, 
                                                  self.image_size, 
                                                  self.encoder.num_extra_tokens, 
                                                  self.encoder.seq_len, 
                                                  n_samples=self.model_config.attention_map.n_samples, 
                                                  epoch=self.current_epoch, 
                                                  batch_index=batch_index
                                                  )
                
                figs_to_log.append(fig_paths)

            # Log the figures to wandb
            for fig_paths in figs_to_log:
                for fig_path in fig_paths:
                    self.logger.log_image(key="figures_attention_maps/"+fig_path.replace("./", ""),
                                          images=[fig_path]
                                          )

        # Plot model predictions
        if self.model_config.observe_preds.enabled:
            # Plot a few training and validation samples (inputs, predictions, targets) of the first and last batch seen during the epoch
            
            for batch_index, split in zip([0, -1], ["train", "val"]):
                # Plot a few training and validation samples of the first and last batch seen during the epoch
                fig_paths = plot_image_predictions(self.save_folder,
                                                   split,
                                                   self.train_inputs, 
                                                   self.train_preds,
                                                   self.train_targets,
                                                   self.image_size,
                                                   n_samples=self.model_config.observe_preds.n_samples,
                                                   batch_index=batch_index,
                                                   epoch=self.current_epoch
                                                   )
                
                figs_to_log.append(fig_paths)

            # Log the figures to wandb
            for fig_paths in figs_to_log:
                for fig_path in fig_paths:
                    try:
                        self.logger.log_image(key="figures_image_predictions/"+fig_path.replace("./", ""),
                                              images=[fig_path]
                                              )
                    except Exception as e:
                        log_message = f"Error logging image predictions to wandb: {e}"
                        log_message += f"Issue for figure path: {fig_path}"
                        log_message += f"This image logging was skipped."
                        logger.warning(log_message)

        # Reset the lists for the next epoch
        self.train_inputs = []
        self.train_preds = []
        self.train_targets = []

        self.val_inputs = []
        self.val_preds = []
        self.val_targets = []

        if self.data_config.validate_in_and_out_domain:
            self.gen_val_inputs = []
            self.gen_val_preds = []
            self.gen_val_targets = []

        self.train_attention_scores = []
        self.val_attention_scores = []

        if self.data_config.validate_in_and_out_domain:
            self.gen_val_attention_scores = []

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
        This method is called for each batch during the testing phase.
        This is a default PyTorch Lightning method that we override to define the testing logic.
        """

        x, y, task_tokens_seq, example_in_context, y_true_size, x_grid_object_ids, special_grid_tokens_dict = batch

        x, y_hat, y, mask, special_grid_tokens_dict = self.shared_step(batch)

        B, H, W = x.shape
        B, seq_len = y.shape

        # TODO: See exactly how we want to compute the loss and metrics. What types of tokens we want to consider.
        # Also, convert the non-data tokens to background tokens (i.e., 0) when computing the loss and metrics, no?

        # Loss per symbol (with all sort of padding considered): compute the loss per token/symbol
        per_sample_loss = F.cross_entropy(y_hat, y.long(), reduction='none').float()  # [B, seq_len]
        loss_symbol_with_pad = per_sample_loss.mean().unsqueeze(0)

        # Loss per symbol (without padding considered): compute the loss per token/symbol and then apply the mask to ignore the padding tokens
        per_sample_loss = F.cross_entropy(y_hat, y.long(), reduction='none').float()  # [B, seq_len]
        loss_symbol_no_pad = ((per_sample_loss * mask).sum() / mask.sum()).unsqueeze(0)  # only consider non-padding elements

        # Compute predictions
        preds = torch.argmax(y_hat, dim=1)  # [B, seq_len]; predictions for each token/symbol of the model for each sample of the batch

        # Accuracy per symbol (with padding) (i.e., the accuracy of the model in predicting the correct symbol for each pixel of the grid considering the whole max. padded grid, thus also the padding tokens)
        acc_symbol_with_pad = (preds == y).float().mean().unsqueeze(0)

        # Accuracy per symbol (without padding) (i.e., the accuracy of the model in predicting the correct symbol for each pixel of the grid considering only the target grid, that is, without considering the padding tokens)
        acc_symbol_no_pad = (((preds == y) * mask).sum().float() / mask.sum()).unsqueeze(0)  # only consider non-padding elements

        # Grid accuracy with pad (only count as correct if the entire padded grid is correct)
        acc_grid_with_pad = torch.all(preds == y, dim=1).float().mean().unsqueeze(0)

        # Grid accuracy without pad (only count as correct if entire non-padding grid is correct)
        acc_grid_no_pad = torch.all((preds == y) | ~mask, dim=1).float().mean().unsqueeze(0)   # | ~mask ensures automatically count as correct the padding tokens

        logs = {'loss': loss_symbol_with_pad,
                'loss_no_pad': loss_symbol_no_pad,
                'acc': acc_symbol_with_pad, 
                'acc_no_pad': acc_symbol_no_pad, 
                'acc_grid': acc_grid_with_pad, 
                'acc_grid_no_pad': acc_grid_no_pad
                }

        if dataloader_idx == 0:
            # Logging
            results = {f"test_{k}": v for k, v in logs.items()}
            self.log_dict(results, logger=True, on_step=True, prog_bar=True, add_dataloader_idx=False)
            self.test_step_results.append(results)

            self.test_inputs.append(x)
            self.test_preds.append(preds)
            self.test_targets.append(y)

        elif dataloader_idx == 1:
            # Logging
            results = {f"gen_test_{k}": v for k, v in logs.items()}
            self.log_dict(results, logger=True, on_step=True, prog_bar=True, add_dataloader_idx=False)
            self.gen_test_step_results.append(results)

            self.gen_test_inputs.append(x)
            self.gen_test_preds.append(preds)
            self.gen_test_targets.append(y)
        
        return results
    
    def on_test_epoch_end(self):
        """
        This method is called at the end of the testing phase.
        This is a default PyTorch Lightning method that we override to define the logic at the end of the testing phase.
        """

        figs_to_log = []

        if len(self.test_step_results) != 0:
            test_step_results = self.test_step_results

            test_keys = list(test_step_results[0].keys())  # we take the first element (i.e., for the first epoch) of the list since all elements have the same keys

            # Results should contain a key for each metric (e.g., loss, acc) and the corresponding values for the single epoch seen during testing
            test_results = {k: torch.cat([x[k] for x in test_step_results]).cpu().numpy() for k in test_keys}

            log_message = f"[Test epoch {self.current_epoch}] Metrics per batch: \n"
            for k, v in test_results.items():
                log_message += f"{k}: {v} \n"

            logger.info(log_message)
            
            self.test_results = test_results

            # Plot a few test samples (inputs, predictions, targets) of the first and last batch of testing (single epoch)
            if self.model_config.observe_preds.enabled:
                fig_paths = plot_image_predictions(self.save_folder, "test", self.test_inputs, self.test_preds, self.test_targets, self.image_size, n_samples=self.model_config.observe_preds.n_samples, batch_index=0)
                figs_to_log.append(fig_paths)
                fig_paths = plot_image_predictions(self.save_folder, "test", self.test_inputs, self.test_preds, self.test_targets, self.image_size, n_samples=self.model_config.observe_preds.n_samples, batch_index=self.trainer.num_test_batches[0]-1)
                figs_to_log.append(fig_paths)

        if self.data_config.use_gen_test_set:   
            if len(self.gen_test_step_results) != 0:
                gen_test_step_results = self.gen_test_step_results

                gen_test_keys = list(gen_test_step_results[0].keys())  # we take the first element (i.e., for the first epoch) of the list since all elements have the same keys

                # Results should contain a key for each metric (e.g., loss, acc) and the corresponding values for the single epoch seen during testing
                gen_test_results = {k: torch.cat([x[k] for x in gen_test_step_results]).cpu().numpy() for k in gen_test_keys}

                log_message += f"[Test systematic generalization epoch {self.current_epoch}] Metrics per batch: \n"
                for k, v in gen_test_results.items():
                    log_message += f"{k}: {v} \n"

                logger.info(log_message)
                
                self.gen_test_results = gen_test_results

                # Plot a few test samples (inputs, predictions, targets) of the first and last batch of testing (single epoch)
                if self.model_config.observe_preds.enabled:
                    fig_paths = plot_image_predictions(self.save_folder, "gen_test", self.gen_test_inputs, self.gen_test_preds, self.gen_test_targets, self.image_size, n_samples=self.model_config.observe_preds.n_samples, batch_index=0)
                    figs_to_log.append(fig_paths)
                    fig_paths = plot_image_predictions(self.save_folder, "gen_test", self.gen_test_inputs, self.gen_test_preds, self.gen_test_targets, self.image_size, n_samples=self.model_config.observe_preds.n_samples, batch_index=self.trainer.num_test_batches[1]-1)
                    figs_to_log.append(fig_paths)

        if len(figs_to_log) != 0:
            # Log the figures to wandb
            for fig_paths in figs_to_log:
                for fig_path in fig_paths:
                    self.logger.log_image(key="figures_image_predictions/"+fig_path.replace("./", ""),
                                          images=[fig_path]
                                          )

    def on_train_end(self):
        """
        This method is called at the end of the training phase.
        This is a default PyTorch Lightning method that we override to define the logic at the end of the training phase.
        """

        # Plot learning rate values used during training
        fig_path = plot_lr_schedule(self.save_folder, self.lr_values)

        # Log the learning rate schedule to wandb
        self.logger.log_image(key="figures_lr_schedule/"+fig_path.replace("./", ""),
                              images=[fig_path]
                              )

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """
        Override the PyTorch Lightning optimizer_step method to add custom logic before the optimizer.step() call.
        
        NOTE: We overwrite it for learning rate warm-up.
        """

        if self.model_config.training_hparams.lr_warmup.enabled:
            if self.model_config.training_hparams.lr_warmup.type == "linear":
                # Linear LR warm up
                num_lr_warmup_steps = self.model_config.training_hparams.lr_warmup.num_steps
                if self.trainer.global_step < num_lr_warmup_steps:
                    lr_scale = min(1.0, float(self.trainer.global_step + 1) / num_lr_warmup_steps)
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr_scale * self.model_config.training_hparams.lr
            else:
                raise ValueError(f"Unknown LR warmup type given: {self.model_config.training_hparams.lr_warmup.type}")

        self.lr_values.append(optimizer.param_groups[0]["lr"])
        
        # This is the content of the original optimizer_step method from PyTorch Lightning
        # TODO: Get warning due to this line? Even though it is the original PTL code?
        optimizer.step(closure=optimizer_closure)   # update params

    def configure_optimizers(self):
        """ 
        Initializes the optimizer and the learning rate scheduler. 
        The optimizer is initialized with the parameters of the model and the learning rate scheduler is initialized with the optimizer.
        
        See: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers

        Returns:
            optimizer_config (dict): A dictionary containing the optimizer and the learning rate scheduler to be used during training.
        """

        # Define the optimizer
        if self.model_config.training_hparams.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.model_config.training_hparams.lr, weight_decay=self.model_config.training_hparams.wd)
        
        elif self.model_config.training_hparams.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.model_config.training_hparams.lr, weight_decay=self.model_config.training_hparams.wd)
        
        elif self.model_config.training_hparams.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.model_config.training_hparams.lr, momentum=0.9, weight_decay=self.model_config.training_hparams.wd)
        
        else:
            raise ValueError(f"Unknown optimizer given: {self.model_config.training_hparams.optimizer}")

        # Define the learning rate scheduler
        if self.model_config.training_hparams.scheduler.type == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        elif self.model_config.training_hparams.scheduler.type == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        elif self.model_config.training_hparams.scheduler.type == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        else:
            raise ValueError(f"Unknown scheduler given: {self.model_config.training_hparams.scheduler.type}")

        optimizer_config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.model_config.training_hparams.scheduler.interval,  # 'epoch' or 'step'
                "frequency": self.model_config.training_hparams.scheduler.frequency,  # 'epoch' or 'step'; how often to call the scheduler w.r.t. the interval
                "monitor": f"metrics/{self.model_config.training_hparams.scheduler.monitored_metric}",  # metric to track for lr scheduling. E.g., metrics/val_loss or metrics/val_acc
            },
        }

        return optimizer_config


class BEFOREARCModel(VisReasModel):
    def __init__(self, base_config, model_config, data_config, backbone_network_config, head_network_config, image_size, save_folder, **kwargs):

        # Save the hyperparameters to self.hparams so that they can be stored in the model checkpoint when using torch.save()
        self.save_hyperparameters()

        # Update the max image size to take into account the visual tokens (i.e., border and newline tokens)
        if model_config.visual_tokens.enabled:
            self.image_size = image_size + 1 + 1 # +1 for the border tokens (bottom and right) and +1 for the newline token (last column of the grid)
            logger.info(f"Image grid size (with all the special visual tokens considered): {self.image_size}x{self.image_size}")
        else:
            self.image_size = image_size
            logger.info(f"Image grid size: {self.image_size}x{self.image_size}")
        

        super().__init__(base_config=base_config, 
                         model_config=model_config,
                         data_config=data_config,
                         backbone_network_config=backbone_network_config, 
                         image_size=self.image_size,
                         save_folder=save_folder
                         )

        self.model_config = model_config

        self.seq_len = self.image_size * self.image_size    # the sequence length with data tokens and any sort of padding
        self.num_channels = 1

        self.num_data_tokens = 10   # symbols in the grid (0-9)

        if self.model_config.visual_tokens.enabled:
            self.num_special_tokens = 5     # PAD_TOKEN (10), X_ENDGRID_TOKEN (11), Y_ENDGRID_TOKEN (12), XY_ENDGRID_TOKEN (13), NL_GRID_TOKEN (14)
        else:
            self.num_special_tokens = 1     # PAD_TOKEN (10)

        self.num_classes = self.num_data_tokens + self.num_special_tokens   # number of token categories that can be predicted by the _whole_ model; 10 for symbols + 1 for each special token that could be predicted

        ## Model backbone/encoder
        if model_config.backbone == "resnet":
            self.encoder, bb_num_out_features = get_resnet(base_config=base_config,
                                                           model_config=model_config,
                                                           network_config=backbone_network_config,
                                                           image_size=self.image_size,
                                                           num_classes=self.num_classes
                                                           )
            self.bb_embed_dim = bb_num_out_features   # embedding dimension backbone model


        elif model_config.backbone == "transformer":
            self.encoder = get_transformer_encoder(base_config=base_config,
                                                   model_config=self.model_config,
                                                   network_config=backbone_network_config, 
                                                   image_size=self.image_size,
                                                   num_channels=self.num_channels,
                                                   num_classes=self.num_classes 
                                                   )
            self.bb_embed_dim = backbone_network_config.embed_dim   # embedding dimension backbone model
            

        elif model_config.backbone == "vit":
            self.encoder = get_vit(base_config=base_config,
                                   model_config=model_config,
                                   network_config=backbone_network_config,
                                   image_size=self.image_size,
                                   num_channels=self.num_channels,
                                   num_classes=self.num_classes
                                   )
            self.bb_embed_dim = backbone_network_config.embed_dim   # embedding dimension backbone model

        elif model_config.backbone == "looped_vit":
            self.encoder = get_looped_vit(base_config=base_config,
                                          model_config=model_config,
                                          network_config=backbone_network_config,
                                          image_size=self.image_size,
                                          num_channels=self.num_channels,
                                          num_classes=self.num_classes
                                          )
            self.bb_embed_dim = backbone_network_config.embed_dim   # embedding dimension backbone model
        
        else:
            raise ValueError(f"Unknown model backbone given: {model_config.backbone}")
        
        self.head_input_dim = self.bb_embed_dim   # embedding dimension of the backbone model, usually the same as its input embedding dimension
        self.head_input_embed_dim = head_network_config.embed_dim   # dimension of the actual input that will be passed to the head network; initially assumed to be of dimension equal to the embedding dimension of the head model


        ## Task embedding
        if model_config.task_embedding.enabled and model_config.task_embedding.approach == "task_tokens":
            self.embed_task_tokens_seq = nn.Embedding(self.num_classes + model_config.num_elementary_tasks, embedding_dim=backbone_network_config.embed_dim, device=self.device)


        ## Encoder to Decoder projection layer; useful to handle the task embedding that is concatenated
        # TODO: We could handle the task embedding differently than by a simple concatenation. For example using FiLM. See later.
        if self.head_input_dim != self.head_input_embed_dim:
            self.enc_to_dec_proj = nn.Linear(self.head_input_dim, self.head_input_embed_dim, device=self.device)  # project the encoder output (of dimension backbone_network_config.embed_dim + task_embedding_dim) to the decoder embedding dimension

        
        ## Model head or decoder
        if model_config.head == "transformer":
            self.decoder = get_transformer_decoder(model_config=self.model_config,
                                                   network_config=head_network_config,
                                                   num_classes=self.num_classes,
                                                   seq_len=self.seq_len,
                                                   )
            
        elif model_config.head == "mytransformer":  
            self.decoder = get_mytransformer_decoder(model_config=self.model_config,
                                                     network_config=head_network_config,
                                                     num_classes=self.num_classes,
                                                     seq_len=self.seq_len,
                                                     )
              
        elif model_config.head == "xtransformer":
            self.decoder = get_xtransformer_decoder(model_config=self.model_config,
                                                    network_config=head_network_config,
                                                    num_classes=self.num_classes,
                                                    seq_len=self.seq_len
                                                    )
        

        elif model_config.head == "mlp":
            self.decoder = get_mlp_head(network_config=head_network_config, 
                                        embed_dim=self.head_input_dim, 
                                        output_dim=self.num_classes, 
                                        activation='relu',
                                        num_layers=2
                                        )
        
        else:
            raise ValueError(f"Unknown model head given: {model_config.head}")


    def check_device_placement(self):
        # Iterate through all parameters of BEFOREARCModel and its submodules and check for device placement
        log_message = "Named parameters of self: \n"
        for name, param in self.named_parameters():
            log_message += f"{name}: {param.device}\n"

        if 'cpu' in log_message:
            log_message += "WARNING: Some tensors are on CPU.\n"
            logger.error(log_message)
        logger.warning(log_message)

        # Iterate through all attributes of self and check for Tensors and their device placement
        log_message = "Attributes of self that are torch.Tensor: \n"
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                log_message += f"Tensor '{name}' is on device: {value.device}\n"

        if 'cpu' in log_message:
            log_message += "WARNING: Some tensors are on CPU.\n"
            logger.error(log_message)
        logger.warning(log_message)

        # Iterate through all attributes of self and check for nn.Modules and their parameters and buffers device placement
        log_message = "Attributes of self that are nn.Module: \n"
        for name, value in vars(self).items():
            if isinstance(value, nn.Module):
                # Check parameters and buffers of nn.Module
                for param_name, param in value.named_parameters():
                    log_message += f"Parameter '{name}.{param_name}' is on device: {param.device}\n"
                for buffer_name, buffer in value.named_buffers():
                    log_message += f"Buffer '{name}.{buffer_name}' is on device: {buffer.device}\n"
        
        if 'cpu' in log_message:
            log_message += "WARNING: Some parameters are on CPU.\n"
            logger.error(log_message)
        logger.warning(log_message)


    def forward(self, x, y, task_tokens_seq=None, example_in_context=None, x_grid_object_ids=None):
        B, H, W = x.shape
        B, seq_len = y.shape

        device = x.device

        # Handle the task embedding if applicable
        if self.model_config.task_embedding.enabled and (task_tokens_seq is not None):
            task_embedding = self.embed_task_tokens_seq(task_tokens_seq.to(device))  # [B, num_tasks, embed_dim]
        else:
            task_embedding = None
        
        # Use grid object ids for the OPE (which is used within the APE)
        if not(self.model_config.ope.enabled and (x_grid_object_ids is not None) and self.model_config.backbone in ["vit", "looped_vit", "transformer"]):
            x_grid_object_ids = None

        # Encode the input sequence
        x_encoded = self.encoder(x, task_embedding, example_in_context, x_grid_object_ids)  # [B, seq_len, embed_dim]; NOTE: the extra tokens will have been truncated so the encoded sequence will also have a dim seq_len

        # Decode the encoded input sequence
        if self.model_config.head in ["transformer", "xtransformer", "mytransformer"]:
            # Transformer Decoder

            if self.head_input_dim != self.head_input_embed_dim:
                # Map the encoded input sequence to the same embedding dimension as the decoder's
                x_encoded = self.enc_to_dec_proj(x_encoded)  # [B, seq_len, head_input_embed_dim]

            logits = self.decoder(y, x_encoded) # [B, seq_len, num_classes]

        elif self.model_config.head in ["mlp"]:
            # MLP Decoder/Head
    
            # Forward pass through the model head
            # We can treat each pixel/token independently as part of a sequence, so we can directly apply a Linear layer 
            # where the last dimension is the features dimension, instead of reshaping the tensor
            logits = self.decoder(x_encoded)   # [B, seq_len, num_classes] <-- [B, seq_len=H*W, C=self.network_config.embed_dim]

        return logits

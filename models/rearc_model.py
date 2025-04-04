import torch
from torch import nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl

# Personal codebase dependencies
from networks.backbones.resnet import get_resnet
from networks.backbones.transformer import get_transformer_encoder
from networks.backbones.vit import get_vit
from networks.heads.mlp import get_mlp_head
from networks.heads.transformer import get_transformer_decoder
from utility.utils import plot_lr_schedule
from utility.rearc.utils import observe_image_predictions, plot_attention_scores
from utility.logging import logger


class VisReasModel(pl.LightningModule):
    """
    Model module class that handles the training, validation and testing logic of the model.
    It is based on PTL's LightningModule class.
    """
    
    def __init__(self, base_config, model_config, backbone_network_config, image_size):
        super().__init__()

        self.base_config = base_config
        self.model_config = model_config
        self.backbone_network_config = backbone_network_config
        self.image_size = image_size

        # Test inputs, targets and predictions for local observation
        self.test_inputs = []
        self.gen_test_inputs = []
        self.test_preds = []
        self.test_targets = []
        self.gen_test_preds = []
        self.gen_test_targets = []

        # Train and val predictions and targets for local logging if verbose
        self.train_inputs = []
        self.train_preds = []
        self.train_targets = []

        self.val_inputs = []
        self.val_preds = []
        self.val_targets = []

        # Metrics for local plotting
        self.train_loss_step = []
        self.train_acc_step = []
        self.train_grid_acc_step = []

        self.val_loss_step = []
        self.val_acc_step = []
        self.val_grid_acc_step = []

        # Test results for logging
        self.test_step_results = [] # NOTE: this is needed to store the results of the test step for each batch (i.e., at each step), and display the final results at the end of the epoch
        self.gen_test_step_results = [] # NOTE: this is needed to store the results of the generalization test step for each batch (i.e., at each step), and display the final results at the end of the epoch

        # Learning rate values for plotting LR schedule
        self.lr_values = []

        # Store the attention scores
        if self.model_config.attention_map.enabled:
            self.train_attention_scores = []
            self.val_attention_scores = []


    def load_backbone_weights(self, checkpoint_path):
        self.model_backbone.load_state_dict(torch.load(checkpoint_path, weights_only=False)['model'], strict=False)
        logger.info(f"Loaded ckpt weights for backbone at ckpt path: {checkpoint_path}")

    def freeze_backbone_weights(self):
        for param in self.model_backbone.parameters():
            param.requires_grad = False

    def create_predictions_mask(self, B: int, H: int, W: int, y_true_size: int) -> torch.Tensor:
        """ Create a multiplicative mask to ignore all the non-symbol (e.g., padding) tokens when computing the loss and accuracy. """

        mask = torch.zeros((B, H, W), dtype=torch.bool, device=self.device)  # [B, H, W] ; initialize all as 0/False (padded)

        # Create mask for each sample of the batch
        for i in range(B):
            true_h = int(y_true_size[i][0])  # get actual height of the target y
            true_w = int(y_true_size[i][1])  # get actual width of the target y
            mask[i, :true_h, :true_w] = 1  # mark non-padding cells as 1/True

        mask = mask.view(B, -1)  # flatten the 2D mask to match y's flattended shape: [B, seq_len]

        return mask

    def shared_step(self, batch):
        """ 
        The same code is shared between training, validation and test steps. 
        In theory, we would not need to compute and return the loss for testing, hence we would only call shared_step() in the test_step() method. 
        However, we still do it, so the purpose of this shared_step() is lesser.
        """

        x, y, samples_task_id, y_true_size = batch   # [B, H, W], [B, H, W], [B], [B]

        B, H, W = x.shape

        # Flatten 2D tensor (grid image) y
        y = y.view(B, -1)  # [B, seq_len=H*W] <-- [B, H, W]

        # Forward pass of the model
        if self.model_config.task_embedding.enabled:
            # Forward pass (with the task embeddings)
            y_hat = self(x, y, samples_task_id)    # computed logits
        else:
            # Forward pass
            y_hat = self(x, y)   # computed logits

        # Permute the dimensions of y_hat to be [B, num_classes, seq_len] instead of [B, seq_len, num_classes] to match PyTorch's cross_entropy function format
        y_hat = y_hat.permute(0, 2, 1)  # [B, num_classes, seq_len] <-- [B, seq_len, num_classes]

        # Create the multiplicative mask based on the true sizes of y to only compute the metrics w.r.t. the actual tokens to predict in the target
        mask = self.create_predictions_mask(B, H, W, y_true_size)

        return x, y_hat, y, mask

    def step(self, batch, batch_idx):

        x, y_hat, y, mask = self.shared_step(batch)    # [B, num_classes, seq_len], [B, seq_len], [B, seq_len]

        B, H, W = x.shape
        B, seq_len = y.shape

        # probabilities = F.softmax(y_hat, dim=1)  # compute the probabilities (normalized logits) of the model for each sample of the batch

        # Loss per symbol (with all sorts of padding considered): compute the loss per token/symbol
        per_sample_loss = F.cross_entropy(y_hat, y.long(), reduction='none').float()  # [B, seq_len]
        loss_symbol_with_pad = (per_sample_loss.mean()).unsqueeze(0)

        # Loss per symbol (without padding): compute the loss per token/symbol and then apply the mask to ignore the padding tokens
        per_sample_loss = F.cross_entropy(y_hat, y.long(), reduction='none').float()  # [B, seq_len]
        loss_symbol_no_pad = ((per_sample_loss * mask).sum() / mask.sum()).unsqueeze(0)  # only consider non-padding elements

        # Compute predictions
        preds = torch.argmax(y_hat, dim=1)  # [B, seq_len]; predictions for each token/symbol of the model for each sample of the batch

        # Accuracy per symbol (with padding) (i.e., the accuracy of the model in predicting the correct symbol for each pixel of the grid considering the whole max. padded grid, thus also the padding tokens)
        # acc_symbol_with_pad = (torch.sum(y == preds).float() / (y.numel())).unsqueeze(0)    # same as line below
        acc_symbol_with_pad = (y == preds).float().mean().unsqueeze(0)

        # Accuracy per symbol (without padding) (i.e., the accuracy of the model in predicting the correct symbol for each pixel of the grid considering only the target grid, that is, without considering the padding tokens)
        acc_symbol_no_pad = (((preds == y) * mask).sum().float() / mask.sum()).unsqueeze(0)  # only consider non-padding elements

        # Grid accuracy (only count as correct if the entire padded grid is correct)
        # grid_acc = (torch.sum(torch.all(y == preds, dim=1)).float() / B).unsqueeze(0)    # same as line below
        acc_grid_with_pad = torch.all(y == preds, dim=1).float().mean().unsqueeze(0)

        # Grid accuracy (only count as correct if entire non-padding grid is correct)
        # acc_grid_no_pad = (torch.sum(torch.all((preds == y) | ~mask, dim=1)).float() / B).unsqueeze(0)    # same as line below
        acc_grid_no_pad = torch.all((preds == y) | ~mask, dim=1).float().mean().unsqueeze(0)   # | ~mask ensures automatically count as correct the padding tokens

        logs = {'loss': loss_symbol_with_pad, 
                'acc': acc_symbol_with_pad, 
                'acc_grid_with_pad': acc_grid_with_pad, 
                'loss_no_pad': loss_symbol_no_pad, 
                'acc_no_pad': acc_symbol_no_pad, 
                'acc_grid_no_pad': acc_grid_no_pad
                }
        
        loss = loss_symbol_with_pad

        return loss, logs, preds, y

    def training_step(self, batch, batch_idx):
        """
        This method is called for each batch during the training phase.
        This is a default PyTorch Lightning method that we override to define the training logic.
        """
        x, y, samples_task_id, y_true_size = batch

        B, H, W = x.shape

        loss, logs, preds, y = self.step(batch, batch_idx)

        # Logging
        self.log_dict({f"metrics/train_{k}": v for k,v in logs.items()}, prog_bar=True, logger=True, on_step=True, on_epoch=True)    # NOTE: this is monitored for best checkpoint and early stopping
        
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
        self.train_grid_acc_step.append(logs['acc_grid_with_pad'])

        return loss

    def validation_step(self, batch, batch_idx):
        """
        This method is called for each batch during the validation phase.
        This is a default PyTorch Lightning method that we override to define the validation logic.
        """

        x, y, samples_task_id, y_true_size = batch

        B, H, W = x.shape

        # NOTE: val_loss is the monitored metric during training
        loss, logs, preds, y = self.step(batch, batch_idx)

        # Logging
        self.log_dict({f"metrics/val_{k}": v for k, v in logs.items()}, prog_bar=True, logger=True, on_step=True, on_epoch=True)    # NOTE: this is monitored for best checkpoint and early stopping
        self.log_dict({"learning_rate": self.lr_schedulers().get_last_lr()[-1]}, prog_bar=True, logger=True, on_step=True, on_epoch=True)    # NOTE: this is monitored for best checkpoint and early stopping. This yields learning_rate in the logs

        # For the first and last validation batch of the epoch
        if (batch_idx == 0) or (batch_idx == self.trainer.num_val_batches[0] - 1):
            # Save batch (of inputs, preds, targets) for current epoch for plotting
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

        # Save to plot locally
        self.val_loss_step.append(logs['loss'])
        self.val_acc_step.append(logs['acc'])
        self.val_grid_acc_step.append(logs['acc_grid_with_pad'])

        return loss

    def on_train_epoch_end(self):
        """
        This method is called at the end of each training epoch.
        This is a default PyTorch Lightning method that we override to define the logic at the end of each training epoch.
        NOTE: It is called after the on_train_epoch_end() method of the Callback class.
        """

        # Plot attention maps if attention maps are enabled and exist
        if self.model_config.attention_map.enabled and hasattr(self.encoder, 'get_attention_scores'):
            # Plot attention maps of some training samples of the first and last batch seen during the epoch
            plot_attention_scores("train", 
                                  self.train_inputs, 
                                  self.train_attention_scores, 
                                  self.model_config.attention_map.layer, 
                                  self.backbone_network_config.num_heads, 
                                  self.image_size, 
                                  self.encoder.num_extra_tokens, 
                                  self.encoder.seq_len, 
                                  n_samples=self.model_config.attention_map.n_samples, 
                                  epoch=self.current_epoch, 
                                  batch_index=0
                                  )
            
            plot_attention_scores("train",
                                  self.train_inputs,
                                  self.train_attention_scores,
                                  self.model_config.attention_map.layer,
                                  self.backbone_network_config.num_heads,
                                  self.image_size,
                                  self.encoder.num_extra_tokens,
                                  self.encoder.seq_len,
                                  n_samples=self.model_config.attention_map.n_samples,
                                  epoch=self.current_epoch,
                                  batch_index=-1
                                  )
            
            # Plot attention maps of some validation samples of the first and last batch seen during the epoch
            plot_attention_scores("val", 
                                  self.val_inputs, 
                                  self.val_attention_scores, 
                                  self.model_config.attention_map.layer,
                                  self.backbone_network_config.num_heads,
                                  self.image_size,
                                  self.encoder.num_extra_tokens,
                                  self.encoder.seq_len,
                                  n_samples=self.model_config.attention_map.n_samples,
                                  epoch=self.current_epoch,
                                  batch_index=0
                                  )
            
            plot_attention_scores("val", 
                                  self.val_inputs, 
                                  self.val_attention_scores, 
                                  self.model_config.attention_map.layer, 
                                  self.backbone_network_config.num_heads,
                                  self.image_size,
                                  self.encoder.num_extra_tokens,
                                  self.encoder.seq_len,
                                  n_samples=self.model_config.attention_map.n_samples,
                                  epoch=self.current_epoch,
                                  batch_index=-1
                                  )

        # Plot model predictions
        if self.model_config.observe_preds.enabled:
            # Plot a few training samples (inputs, predictions, targets) of the first and last batch seen during the epoch
            observe_image_predictions("train", 
                                      self.train_inputs, 
                                      self.train_preds, 
                                      self.train_targets, 
                                      self.image_size, 
                                      n_samples=self.model_config.observe_preds.n_samples,
                                      batch_index=0,
                                      epoch=self.current_epoch
                                      )
            
            observe_image_predictions("train", 
                                      self.train_inputs, 
                                      self.train_preds, 
                                      self.train_targets, 
                                      self.image_size, 
                                      n_samples=self.model_config.observe_preds.n_samples, 
                                      batch_index=-1,
                                      epoch=self.current_epoch
                                      )
            
            # Plot a few validation samples (inputs, predictions, targets) of the first and last batch seen during the epoch
            observe_image_predictions("val", 
                                      self.val_inputs, 
                                      self.val_preds, 
                                      self.val_targets, 
                                      self.image_size, 
                                      n_samples=self.model_config.observe_preds.n_samples, 
                                      batch_index=0, 
                                      epoch=self.current_epoch
                                      )
            
            observe_image_predictions("val", 
                                      self.val_inputs, 
                                      self.val_preds, 
                                      self.val_targets, 
                                      self.image_size, 
                                      n_samples=self.model_config.observe_preds.n_samples, 
                                      batch_index=-1, 
                                      epoch=self.current_epoch
                                      )

        # Reset the lists for the next epoch
        self.train_inputs = []
        self.train_preds = []
        self.train_targets = []
        self.val_inputs = []
        self.val_preds = []
        self.val_targets = []

        self.train_attention_scores = []
        self.val_attention_scores = []

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
        This method is called for each batch during the testing phase.
        This is a default PyTorch Lightning method that we override to define the testing logic.
        """

        x, y, samples_task_id, y_true_size = batch

        x, y_hat, y, mask = self.shared_step(batch)

        B, H, W = x.shape
        B, seq_len = y.shape

        # Loss per symbol (with all sort of padding considered): compute the loss per token/symbol
        per_sample_loss = F.cross_entropy(y_hat, y.long(), reduction='none').float()  # [B, seq_len]
        loss_symbol_with_pad = per_sample_loss.mean().unsqueeze(0)

        # Loss per symbol (without padding considered): compute the loss per token/symbol and then apply the mask to ignore the padding tokens
        per_sample_loss = F.cross_entropy(y_hat, y.long(), reduction='none').float()  # [B, seq_len]
        loss_symbol_no_pad = ((per_sample_loss * mask).sum() / mask.sum()).unsqueeze(0)  # only consider non-padding elements

        # Compute predictions
        preds = torch.argmax(y_hat, dim=1)  # [B, seq_len]; predictions for each token/symbol of the model for each sample of the batch

        # Accuracy per symbol (with padding) (i.e., the accuracy of the model in predicting the correct symbol for each pixel of the grid considering the whole max. padded grid, thus also the padding tokens)
        # acc_symbol_with_pad = (torch.sum(y == preds).float() / (y.numel())).unsqueeze(0)    # same as line below
        acc_symbol_with_pad = (y == preds).float().mean().unsqueeze(0)

        # Accuracy per symbol (without padding) (i.e., the accuracy of the model in predicting the correct symbol for each pixel of the grid considering only the target grid, that is, without considering the padding tokens)
        acc_symbol_no_pad = (((preds == y) * mask).sum().float() / mask.sum()).unsqueeze(0)  # only consider non-padding elements

        # Grid accuracy with pad (only count as correct if the entire padded grid is correct)
        # grid_acc = (torch.sum(torch.all(y == preds, dim=1)).float() / B).unsqueeze(0)    # same as line below
        acc_grid_with_pad = torch.all(y == preds, dim=1).float().mean().unsqueeze(0)

        # Grid accuracy without pad (only count as correct if entire non-padding grid is correct)
        # acc_grid_no_pad = (torch.sum(torch.all((preds == y) | ~mask, dim=1)).float() / B).unsqueeze(0)    # same as line below
        acc_grid_no_pad = torch.all((preds == y) | ~mask, dim=1).float().mean().unsqueeze(0)   # | ~mask ensures automatically count as correct the padding tokens

        logs = {'loss': loss_symbol_with_pad, 
                'acc': acc_symbol_with_pad, 
                'acc_grid_with_pad': acc_grid_with_pad, 
                'loss_no_pad': loss_symbol_no_pad, 
                'acc_no_pad': acc_symbol_no_pad, 
                'acc_grid_no_pad': acc_grid_no_pad
                }

        if dataloader_idx == 0:
            self.test_inputs.append(x)
            self.test_preds.append(preds)
            self.test_targets.append(y)

            results = {f"test_{k}": v for k, v in logs.items()}
            self.log_dict(results, logger=True, on_step=True, prog_bar=True)
            self.test_step_results.append(results)

        elif dataloader_idx == 1:
            self.gen_test_inputs.append(x)
            self.gen_test_preds.append(preds)
            self.gen_test_targets.append(y)

            # TODO: Currently in the code we assume that if there is only one dataloader, it will be considered as a test dataloader and not a gen test dataloader even though the data may be of systematic generalization. Fix this to be better maybe?
            results = {f"gen_test_{k}": v for k, v in logs.items()}
            self.log_dict(results, logger=True, on_step=True, prog_bar=True)
            self.gen_test_step_results.append(results)
        
        else:
            raise ValueError(f"Unknown dataloader index given: {dataloader_idx}")
        
        return results
    
    def on_test_epoch_end(self):
        """
        This method is called at the end of the testing phase.
        This is a default PyTorch Lightning method that we override to define the logic at the end of the testing phase.
        """

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
                observe_image_predictions("test", self.test_inputs, self.test_preds, self.test_targets, self.image_size, n_samples=self.model_config.observe_preds.n_samples, batch_index=0)
                observe_image_predictions("test", self.test_inputs, self.test_preds, self.test_targets, self.image_size, n_samples=self.model_config.observe_preds.n_samples, batch_index=self.trainer.num_test_batches[0]-1)

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
                observe_image_predictions("test_gen", self.gen_test_inputs, self.gen_test_preds, self.gen_test_targets, self.image_size, n_samples=self.model_config.observe_preds.n_samples, batch_index=0)
                observe_image_predictions("test_gen", self.gen_test_inputs, self.gen_test_preds, self.gen_test_targets, self.image_size, n_samples=self.model_config.observe_preds.n_samples, batch_index=self.trainer.num_test_batches[1]-1)

    def on_train_end(self):
        """
        This method is called at the end of the training phase.
        This is a default PyTorch Lightning method that we override to define the logic at the end of the training phase.
        """

        # Plot learning rate values used during training
        plot_lr_schedule(self.lr_values)
        return

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """
        Override the PyTorch Lightning optimizer_step method to add custom logic before the optimizer.step() call.
        
        NOTE: We overwrite it for learning rate warm-up.
        TODO: See if ok to define the LR warm-up like this.
        """

        if self.model_config.training_hparams.lr_warmup.enabled:
            # Manual linear LR warm up
            num_lr_warmup_steps = self.model_config.training_hparams.lr_warmup.num_steps
            if self.trainer.global_step < num_lr_warmup_steps:
                lr_scale = min(1.0, float(self.trainer.global_step + 1) / num_lr_warmup_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * self.model_config.training_hparams.lr

        self.lr_values.append(optimizer.param_groups[0]["lr"])

        # if epoch == 0 and batch_idx == 0:
        #     logger.debug(f"Learning rate at epoch 0 and batch 0: {optimizer.param_groups[0]['lr']}")
        # if epoch == 1 and batch_idx == 0:
        #     logger.debug(f"Learning rate at epoch 1 and batch 0: {optimizer.param_groups[0]['lr']}")
        
        # This is the content of the original optimizer_step method from PyTorch Lightning
        optimizer.step(closure=optimizer_closure)   # update params

    def configure_optimizers(self):
        """ Initializes the optimizer and the learning rate scheduler. 
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
        if self.model_config.training_hparams.scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        elif self.model_config.training_hparams.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        elif self.model_config.training_hparams.scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        else:
            raise ValueError(f"Unknown scheduler given: {self.model_config.training_hparams.scheduler}")

        optimizer_config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "metrics/val_loss",  # here write the metric to track for lr scheduling. E.g., metrics/val_loss or metrics/val_acc
            },
        }

        return optimizer_config


class REARCModel(VisReasModel):
    def __init__(self, base_config, model_config, backbone_network_config, head_network_config, image_size, **kwargs):

        # Save the hyperparameters to self.hparams so that they can be stored in the model checkpoint when using torch.save()
        self.save_hyperparameters()

        # Update the max image size to take into account the border tokens
        # self.image_size = image_size + 2  # add 2 for the border tokens (top-bottom and left-right)
        self.image_size = image_size + 1  # add 1 for the border tokens (bottom and right)

        logger.info(f"Image grid size with borders and padding: {self.image_size}x{self.image_size}")

        super().__init__(base_config=base_config, 
                         model_config=model_config, 
                         backbone_network_config=backbone_network_config, 
                         image_size=self.image_size)

        self.model_config = model_config

        self.seq_len = self.image_size * self.image_size    # the sequence length with data tokens and any sort of padding
        self.num_channels = 1

        self.num_data_tokens = 10   # symbols in the grid (0-9)
        self.num_special_tokens = 4  # PAD_TOKEN (10), X_ENDGRID_TOKEN (11), Y_ENDGRID_TOKEN (12), XY_ENDGRID_TOKEN (13)

        self.num_classes = self.num_data_tokens + self.num_special_tokens   # number of token categories that can be predicted by the _whole_ model; 10 for symbols + 1 for each special token that could be predicted
        self.vocab_size = self.num_classes + 2  # number of different tokens that we consider; +2 for the BOS and EOS tokens
        # TODO: Is it ok to use vocab_size = num_classes + 2 instead of num_classes + 1 even though BOS never has to be predicted?
        #       The main reason to do +2 instead of +1 is to have a correct mapping for the layer that embeds tokens.
        #       What I had done initially was using self.vocab_size (16) and self.decoder_vocab_size (15), but then it would mean that the logits converted to a class index would not represent the same thing for the decoder and the final output layer.

        ## Model backbone/encoder
        if model_config.backbone == "resnet":
            self.encoder, bb_num_out_features = get_resnet(base_config=base_config,
                                                           model_config=model_config,
                                                           network_config=backbone_network_config,
                                                           image_size=self.image_size,
                                                           num_classes=self.num_classes,
                                                           )
            self.backbone_input_embed_dim = bb_num_out_features   # embedding dimension backbone model


        elif model_config.backbone == "transformer":
            self.encoder = get_transformer_encoder(base_config=base_config,
                                                   model_config=self.model_config,
                                                   network_config=backbone_network_config, 
                                                   image_size=self.image_size,
                                                   num_channels=self.num_channels,
                                                   num_classes=self.num_classes, 
                                                   )
            self.backbone_input_embed_dim = backbone_network_config.embed_dim   # embedding dimension backbone model
            

        elif model_config.backbone == "vit":
            self.encoder = get_vit(base_config=base_config,
                                   model_config=model_config,
                                   network_config=backbone_network_config,
                                   image_size=self.image_size,
                                   num_channels=self.num_channels,
                                   num_classes=self.num_classes,
                                   )
            self.backbone_input_embed_dim = backbone_network_config.embed_dim   # embedding dimension backbone model

        elif model_config.backbone == "looped_vit":
            raise NotImplementedError("Looped ViT not implemented yet")
        
        else:
            raise ValueError(f"Unknown model backbone given: {model_config.backbone}")
        
        self.head_input_dim = self.backbone_input_embed_dim   # embedding dimension of the backbone model, usually the same as its input embedding dimension
        self.head_input_embed_dim = head_network_config.embed_dim   # dimension of the actual input that will be passed to the head network; initially assumed to be of dimension equal to the embedding dimension of the head model


        ## Task embedding
        if model_config.task_embedding.enabled:
            task_embedding_dim = model_config.task_embedding.task_embedding_dim
            self.task_embedding = nn.Embedding(model_config.n_tasks, embedding_dim=task_embedding_dim, device=self.device)   # NOTE: 103 is the total number of tasks because the input is a task id (i.e., a number between 0 and 102)
            self.head_input_dim += task_embedding_dim
        else:
            task_embedding_dim = 0
            self.task_embedding = None


        ## Encoder to Decoder projection layer; useful to handle the task embedding that is concatenated
        # TODO: Should we handle the task embedding differently from using a concatenation? For example using FiLM or other methods?
        if self.head_input_dim != self.head_input_embed_dim:
            self.enc_to_dec_proj = nn.Linear(self.head_input_dim, self.head_input_embed_dim, device=self.device)  # project the encoder output (of dimension backbone_network_config.embed_dim + task_embedding_dim) to the decoder embedding dimension

        
        ## Model head or decoder
        if model_config.head == "transformer":
            self.decoder = get_transformer_decoder(model_config=self.model_config,
                                                   network_config=head_network_config,
                                                   num_classes=self.num_classes,
                                                   vocab_size=self.vocab_size,
                                                   seq_len=self.seq_len,
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
        # Iterate through all parameters of REARCModel and its submodules and check for device placement
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


    def forward(self, x, y, samples_task_id=None):
        B, H, W = x.shape
        B, seq_len = y.shape

        # Encode the input sequence
        x_encoded = self.encoder(x)  # [B, seq_len, backbone_input_embed_dim]; NOTE: the extra tokens will have been truncated so the encoded sequence will also have a dim seq_len 
        
        # Handle the task embedding if needed
        if samples_task_id is not None:
            task_embedding = self.task_embedding(samples_task_id)   # [B, task_embedding_dim]
            task_embedding = task_embedding.unsqueeze(1).repeat(1, x_encoded.shape[1], 1) # [B, seq_len, task_embedding_dim]
            x_encoded = torch.cat([x_encoded, task_embedding], 2)  # [B, seq_len, backbone_input_embed_dim + task_embedding_dim]

        # Decode the encoded input sequence
        if self.model_config.head in ["transformer"]:
            # Transformer Decoder

            if self.head_input_dim != self.head_input_embed_dim:
                # Map the encoded input sequence to the same embedding dimension as the decoder's
                x_encoded = self.enc_to_dec_proj(x_encoded)  # [B, seq_len, head_input_embed_dim]

            logits = self.decoder(y, x_encoded)

        elif self.model_config.head in ["mlp"]:
            # MLP Decoder/Head
    
            # Forward pass through the model head
            # We can treat each pixel/token independently as part of a sequence, so we can directly apply a Linear layer 
            # where the last dimension is the features dimension, instead of reshaping the tensor
            logits = self.decoder(x_encoded)   # [B, seq_len, num_classes] <-- [B, seq_len=H*W, C=self.network_config.embed_dim]

        return logits

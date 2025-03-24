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
from utility.utils import plot_lr_schedule, timer_decorator
from utility.rearc.utils import observe_image_predictions
from utility.logging import logger


class VisReasModel(pl.LightningModule):
    """
    Model module class that handles the training, validation and testing logic of the model.
    It is based on PTL's LightningModule class.
    """
    
    def __init__(self, model_config, image_size):
        super().__init__()

        self.model_config = model_config
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


    def load_backbone_weights(self, checkpoint_path):
        self.model_backbone.load_state_dict(torch.load(checkpoint_path, weights_only=False)['model'], strict=False)
        logger.info(f"Loaded ckpt weights for backbone at ckpt path: {checkpoint_path}")

    def freeze_backbone_model(self):
        for param in self.model_backbone.parameters():
            param.requires_grad = False

    def create_predictions_mask(self, B: int, H: int, W: int, y_true_size: int) -> torch.Tensor:
        """ Create a multiplicative mask to ignore the non-symbol (e.g., padding) tokens when computing the loss and accuracy. """
        
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

        # Loss per symbol (with padding): compute the loss per token/symbol
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
        x, y, samples_task_id, y_true_size = batch

        B, H, W = x.shape

        loss, logs, preds, y = self.step(batch, batch_idx)

        # Logging
        self.log_dict({f"metrics/train_{k}": v for k,v in logs.items()}, prog_bar=True, logger=True, on_step=True, on_epoch=True)    # NOTE: this is monitored for best checkpoint and early stopping
        
        # Save two batches (of inputs, preds, targets) per epoch for plotting of images at each epoch
        if (batch_idx == 0) or (batch_idx == self.trainer.num_training_batches - 1):
            self.train_inputs.append(x)
            self.train_preds.append(preds)
            self.train_targets.append(y)

        # Save to plot locally
        self.train_loss_step.append(logs['loss'])
        self.train_acc_step.append(logs['acc'])
        self.train_grid_acc_step.append(logs['acc_grid_with_pad'])

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, samples_task_id, y_true_size = batch

        B, H, W = x.shape

        # NOTE: val_loss is the monitored metric during training
        loss, logs, preds, y = self.step(batch, batch_idx)

        # Logging
        self.log_dict({f"metrics/val_{k}": v for k, v in logs.items()}, prog_bar=True, logger=True, on_step=True, on_epoch=True)    # NOTE: this is monitored for best checkpoint and early stopping
        self.log_dict({"learning_rate": self.lr_schedulers().get_last_lr()[-1]}, prog_bar=True, logger=True, on_step=True, on_epoch=True)    # NOTE: this is monitored for best checkpoint and early stopping. This yields learning_rate in the logs

        # Save two batches (of inputs, preds, targets) per epoch for plotting of images at each epoch
        if (batch_idx == 0) or (batch_idx == self.trainer.num_val_batches - 1):
            self.val_inputs.append(x)
            self.val_preds.append(preds)
            self.val_targets.append(y)

        # Save to plot locally
        self.val_loss_step.append(logs['loss'])
        self.val_acc_step.append(logs['acc'])
        self.val_grid_acc_step.append(logs['acc_grid_with_pad'])

        return loss


    def on_train_epoch_end(self):
        # NOTE: This method is called after the on_train_epoch_end() method of the Callback class.

        if self.model_config.observe_preds.enabled:
            # Plot a few training samples (inputs, predictions, targets) of the first and last batch seen during the epoch
            observe_image_predictions("train", self.train_inputs, self.train_preds, self.train_targets, self.image_size, n_samples=self.model_config.observe_preds.n_samples, batch_index=0, epoch=self.current_epoch)
            observe_image_predictions("train", self.train_inputs, self.train_preds, self.train_targets, self.image_size, n_samples=self.model_config.observe_preds.n_samples, batch_index=-1, epoch=self.current_epoch)
            
            # Plot a few validation samples (inputs, predictions, targets) of the first and last batch seen during the epoch
            observe_image_predictions("val", self.val_inputs, self.val_preds, self.val_targets, self.image_size, n_samples=self.model_config.observe_preds.n_samples, batch_index=0, epoch=self.current_epoch)
            observe_image_predictions("val", self.val_inputs, self.val_preds, self.val_targets, self.image_size, n_samples=self.model_config.observe_preds.n_samples, batch_index=-1, epoch=self.current_epoch)

        # Reset the lists for the next epoch
        self.train_inputs = []
        self.train_preds = []
        self.train_targets = []
        self.val_inputs = []
        self.val_preds = []
        self.val_targets = []

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, samples_task_id, y_true_size = batch

        x, y_hat, y, mask = self.shared_step(batch)

        B, H, W = x.shape
        B, seq_len = y.shape

        # Loss per symbol (with padding): compute the loss per token/symbol
        per_sample_loss = F.cross_entropy(y_hat, y.long(), reduction='none').float()  # [B, seq_len]
        loss_symbol_with_pad = per_sample_loss.mean().unsqueeze(0)

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

        # Save the hyperparameters so that they can be stored in the model checkpoint when using torch.save()
        self.save_hyperparameters() # saves all the arguments (kwargs too) of __init__() to the variable hparams

        super().__init__(model_config=model_config, image_size=image_size)

        self.model_config = model_config

        self.image_size = image_size
        self.seq_len = self.image_size * self.image_size    # the sequence length without extra tokens (but with padding)
        self.num_channels = 1

        # TODO: Should we include padding to predict due to grid size variability? Yes?
        self.num_classes = 10 + 1  # number of token categories; 1 extra token for padding as it has to be predicted

        ## Model backbone/encoder
        if model_config.backbone == "resnet":
            self.encoder, bb_num_out_features = get_resnet(base_config=base_config,
                                                           model_config=model_config,
                                                           network_config=backbone_network_config,
                                                           image_size=self.image_size,
                                                           num_classes=self.num_classes,
                                                           device=self.device
                                                           )
            self.backbone_input_embed_dim = bb_num_out_features   # embedding dimension backbone model


        elif model_config.backbone == "transformer":
            self.encoder = get_transformer_encoder(base_config=base_config,
                                                   model_config=self.model_config,
                                                   network_config=backbone_network_config, 
                                                   image_size=self.image_size,
                                                   num_channels=self.num_channels,
                                                   num_classes=self.num_classes, 
                                                   device=self.device
                                                   )
            self.backbone_input_embed_dim = backbone_network_config.embed_dim   # embedding dimension backbone model
            

        elif model_config.backbone == "vit":
            self.encoder = get_vit(base_config=base_config,
                                   model_config=model_config,
                                   network_config=backbone_network_config,
                                   image_size=self.image_size,
                                   num_channels=self.num_channels,
                                   num_classes=self.num_classes,
                                   device=self.device
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
                                                    max_seq_len=self.seq_len,
                                                    device=self.device)
            
            # Create an additive mask to prevent the Transformer Decoder from looking at the future tokens/positions in self-attention module
            # The mask is a square Tensor of size [seq_len, seq_len] with -inf where we want to mask and 0 where we want to keep
            self.tgt_mask = torch.triu(torch.full((self.seq_len, self.seq_len), float("-inf"), dtype=torch.float32, device=self.device), diagonal=1)    # [seq_len, seq_len]; diagonal=1 specifies from which diagonal to set the elements as -inf

            # Create a target projection layer to map the ground truth target tokens/sequence (obtained from y) to the decoder embedding dimension as a Transformer Decoder needs to receive the target sequence in an embedding space
            self.tgt_projection = nn.Embedding(num_embeddings=self.num_classes, embedding_dim=self.head_input_embed_dim, device=self.device)


        elif model_config.head == "mlp":
            self.decoder = get_mlp_head(network_config=head_network_config, 
                                        embed_dim=self.head_input_dim, 
                                        output_dim=self.num_classes, 
                                        activation='relu', num_layers=2)
        else:
            raise ValueError(f"Unknown model head given: {model_config.head}")


        ## Output layer to go from the decoder output to logits
        self.output_layer = nn.Linear(self.head_input_embed_dim, self.num_classes, device=self.device)


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

    # @timer_decorator
    def training_decode(self, x_encoded, y):
        B, S, D = x_encoded.shape

        self.tgt_mask = self.tgt_mask.to(device=self.device)

        # self.check_device_placement()

        # Preprocess the target sequence y so that it has the dimensions [B, seq_len, head_input_embed_dim] to be used by the Transformer decoder
        tgt = self.tgt_projection(y.long())  # [B, seq_len, head_input_embed_dim], where head_input_embed_dim is the embed dim of the decoder; ensure that tgt is of type long
        
        # We use the full target sequence y as input to the decoder with a causal mask. Thus we predict the seq_len tokens of the target sequence in parallel
        # We are using full teacher forcing
        output_target_seq = self.decoder(tgt=tgt, memory=x_encoded, tgt_mask=self.tgt_mask) # [B, seq_len, head_input_embed_dim]; NOTE: PyTorch's TransformerDecoder returns the logits for each token in the tgt sequence given

        return output_target_seq

    # @timer_decorator
    def inference_decode(self, x_encoded):
        B, S, D = x_encoded.shape

        self.tgt_mask = self.tgt_mask.to(device=self.device)

        # self.check_device_placement()

        self.decoder.eval()  # set the decoder network to evaluation mode
        with torch.no_grad():
            # We use our predicted target sequence up to token at position t as input to the decoder. Thus we predict the seq_len tokens of the target sequence one by one in an auto-regressive manner
            # We using auto-regressive decoding without teacher forcing
            output_tokens = []  # list to store the output (tensor) tokens at each step. Instead of reserving memory for all timesteps at once, we append only the necessary data step-by-step, hence avoiding storing the intermediate states as using a list detaches all states from the computation graph
            
            # Create a start token used to start the AR decoding
            # TODO: Is it correct to start with a tensor of zeros as the first token? What should the start token be?
            start_token = 0
            token_start = torch.full((B, 1), start_token, device=self.device)  # [B, 1]; start with a token start_token as the first token

            # Embed the start token
            token_start = self.tgt_projection(token_start.long())  # [B, 1, head_input_embed_dim]
            
            # Store the first token in the list; note that the list should contain the embedded tokens of each AR decoding step
            output_tokens.append(token_start)
            
            # Start the iterative AR decoding loop
            for t in range(self.seq_len-1): # we have seq_len logits/predictions to get
                prev_tokens = torch.cat(output_tokens, dim=1)   # [B, t+1, head_input_embed_dim]; create a tensor from the list and detach to save memory

                # Decode up to step t; in theory we do not need to give a causal mask as we are only interested in the last token logits, but we give it to avoid possible errors if we were to analyze the logits of all tokens at each step
                tgt_mask = self.tgt_mask[:t+1, :t+1]  # [t+1, t+1]; use the global causal mask for the first t+1 steps only

                # Compute the output embeddings up to position t
                outputs = self.decoder(prev_tokens, x_encoded, tgt_mask=tgt_mask) # [B, t+1, head_input_embed_dim]; NOTE: The logits returned by PyTorch's TransformerDecoder are that of all the tokens in the tgt sequence given. So below we take only that of the last step
                
                # Get the output token at step t; we need to get an actual (i.e., not just logits) token since we use AR decoding
                token_t_output = outputs[:, -1, :]   # [B, head_input_embed_dim]; We take the logits of the last token (as it is the token of interest)
                
                # Apply the linear output layer to get the logits from the predicted target sequence embeddings
                token_t_logits = self.output_layer(token_t_output)  # [B, seq_len, num_classes] <-- [B, seq_len, head_input_embed_dim]
                
                # Get/sample the token at step t
                # TODO: Implement better sampling? E.g., Beam search, etc. to sample the output token? 
                token_t = token_t_logits.argmax(dim=-1, keepdim=True)  # [B, 1]; This extracts only the last step’s logits and finds the most probable token; Greedy decoding: take the token with the highest probability  
                token_t = token_t.to(device=self.device, dtype=torch.float32)
            
                # Embed the output token of position/step t
                token_t = self.tgt_projection(token_t.long())  # [B, 1, head_input_embed_dim] <-- [B, 1]

                # Store in the list the (embedded) output token at position/step t
                output_tokens.append(token_t)    # TODO: Important to use .copy() here to avoid storing a reference of the tensor token_t since it is changing in the loop?
            
            # Stack the output tokens from the list to get the predicted target sequence
            output_target_seq = torch.cat(output_tokens, dim=1)   # [B, seq_len, head_input_embed_dim]

        return output_target_seq

    # @timer_decorator
    def decode_sequence(self, x_encoded, y):
        # AR Decoding
        if self.training:   # use PTL LightningModule's self.training attribute to check if the model is in training mode; could also use self.trainer.training, self.trainer.validating, self.trainer.testing
            output_target_seq = self.training_decode(x_encoded, y)
        else:
            output_target_seq = self.inference_decode(x_encoded)

        return output_target_seq

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
        if self.model_config.head in ["transformer", "vit"]:
            # Transformer Decoder

            if self.head_input_dim != self.head_input_embed_dim:
                # Map the encoded input sequence to the same embedding dimension as the decoder
                x_encoded = self.enc_to_dec_proj(x_encoded)  # [B, seq_len, head_input_embed_dim]

            # Auto-regressive decoding (with full teacher forcing and causal masking for training)
            output_target_seq = self.decode_sequence(x_encoded, y)   # [B, seq_len, head_input_embed_dim]

            # Apply the linear output layer to get the logits from the predicted target sequence embeddings
            logits = self.output_layer(output_target_seq)  # [B, seq_len, num_classes]

        elif self.model_config.head in ["mlp"]:
            # MLP Decoder/Head
    
            # Forward pass through the model head
            # We can treat each pixel/token independently as part of a sequence, so we can directly apply a Linear layer 
            # where the last dimension is the features dimension, instead of reshaping the tensor
            logits = self.decoder(x_encoded)   # [B, seq_len, num_classes] <-- [B, seq_len=H*W, C=self.network_config.embed_dim]

        return logits

import torch
from torch import nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl

# Personal codebase dependencies
from networks.backbones.resnet import get_resnet
from networks.backbones.transformer import get_transformer_encoder
from networks.backbones.my_vit import get_my_vit
from networks.heads.mlp import get_mlp_head
from networks.heads.transformer import get_transformer_decoder
from utility.utils import plot_lr_schedule, timer_decorator
from utility.rearc.utils import observe_image_predictions
from utility.logging import logger


class VisReasModel(pl.LightningModule):
    def __init__(self, model_config):
        super().__init__()

        self.model_config = model_config

        self.train_preds = []
        self.train_labels = []
        self.val_preds = []
        self.val_labels = []
        self.test_preds = []
        self.test_labels = []
        self.gen_test_preds = []
        self.gen_test_labels = []

        self.test_step_results = [] # NOTE: this is needed to store the results of the test step for each batch (i.e., at each step), and display the final results at the end of the epoch
        self.gen_test_step_results = [] # NOTE: this is needed to store the results of the generalization test step for each batch (i.e., at each step), and display the final results at the end of the epoch

        self.lr_values = []

        # Criterion / Loss Function
        # criterion = nn.CrossEntropyLoss(ignore_index=10)    # ignore_index=10 allows to ignore the symbols 10 (i.e., the padding tokens for us, currently)

    def load_backbone_weights(self, checkpoint_path):
        self.model_backbone.load_state_dict(torch.load(checkpoint_path, weights_only=False)['model'], strict=False)
        logger.info(f"Loaded ckpt weights for backbone at ckpt path: {checkpoint_path}")

    def freeze_backbone_model(self):
        for param in self.model_backbone.parameters():
            param.requires_grad = False

    def shared_step(self, batch):
        x, y, samples_task_id, y_true_size = batch   # [B, H, W], [B, H, W], [B], [B]

        B, H, W = x.shape

        if self.model_config.task_embedding.enabled:
            # Enter the model forward pass with the task embeddings
            y_hat = self(x, y, samples_task_id)    # computed logits
        else:
            # Enter the model forward pass
            y_hat = self(x, y)   # computed logits


        # Permute the dimensions of y_hat to be [B, num_classes, seq_len] instead of [B, seq_len, num_classes] to match Pytorch's cross_entropy function format
        y_hat = y_hat.permute(0, 2, 1)  # [B, num_classes, seq_len] <-- [B, seq_len, num_classes]

        # Flatten y
        y = y.view(B, -1)  # if not padded: [B, H*W] <-- [B, H, W] ; if padded: [B, seq_len=H*W] <-- [B, H, W]

        # TODO: check if my way to take into account the padding of the input grid and output grid when computing the loss and accuracy is correct
        #       Essentially, the loss and accuracy should be computed only on the non-padded part of the grid.
        #       Therefore, after having reshaping the grid to be 1D tensors, we can remove the padding from the input tensor until it reaches the real output tensor size.

        # Create the mask based on the true sizes of y
        mask = torch.zeros((B, H, W), dtype=torch.bool, device=y.device)  # [B, H, W] ; initialize all as 0/False (padded)

        for i in range(B):
            true_h = int(y_true_size[i][0])  # get actual height of the target y
            true_w = int(y_true_size[i][1])  # get actual width of the target y
            mask[i, :true_h, :true_w] = 1  # mark non-padding cells as 1/True

        mask = mask.view(B, -1)  # flatten the 2D mask to match y's flattended shape: [B, seq_len]

        return y_hat, y, mask

    def step(self, batch, batch_idx):

        y_hat, y, mask = self.shared_step(batch)    # [B, num_classes, seq_len], [B, seq_len], [B, seq_len]

        # probabilities = F.softmax(y_hat, dim=1)  # compute the probabilities (normalized logits) of the model for each sample of the batch

        # Compute the loss per token/symbol without taking into account that pad tokens should not be considered in the loss
        # loss = F.cross_entropy(y_hat, y.long())    # compute the loss (averaged over the batch)

        # Compute the loss per token/symbol and then apply the mask?
        # TODO: am I considering the loss for the batch correctly?
        loss = F.cross_entropy(y_hat, y.long(), reduction='none')  # [B, seq_len]
        loss = (loss * mask).sum() / mask.sum()  # only consider non-padded elements

        # Compute actual token predictions; we use greedy sampling with argmax to get the predicted tokens from the logits
        preds = torch.argmax(y_hat, dim=1)  # [B, seq_len]; predictions of the model for each sample of the batch

        # Reshape the sequence of predictions into a grid (i.e., 2D tensor of symbols)
        # grid_preds = preds.view(preds.shape[0], 30, 30, -1)  # [B, H=30, W=30, num_classes]
        # grid_target = y.view(y.shape[0], 30, 30, -1)  # [B, H=30, W=30, num_classes]

        # Accuracy per symbol (i.e., the accuracy of the model in predicting the correct symbol for each pixel of the grid)
        # symbol_acc = torch.sum(y == preds).float() / (y.numel())    # NOTE: (y == preds) yields a boolean tensor of shape [B, seq_len=900]. .float() converts True to 1.0 and False to 0.0, so get the sum of correctly predicted grid cells/symbols. y.numel() returns the total number of elements in the tensor y. That is, it is equivalent to dividing by B*seq_len

        # Accuracy per symbol (ignoring padding)
        # TODO: Am I considering the accuracy for the batch correctly?
        correct = (preds == y) * mask   # only consider valid positions
        symbol_acc = correct.sum().float() / mask.sum()  # only consider non-padded elements

        # Grid accuracy (i.e., the accuracy of the model to predict the whole grid correctly)
        # grid_acc = torch.sum(torch.all(y == preds, dim=1)).float() / len(y)     # torch.all(y == preds, dim=1) returns a boolean tensor of shape [B] where each element is True if all the symbols of the grid are correctly predicted and False otherwise. .float() converts True to 1.0 and False to 0.0. Then, we sum all the True values and divide by the total number of samples in the batch

        # Grid accuracy (only count if entire true grid is correct) ?
        # TODO: Am I considering the accuracy for the batch correctly?
        grid_acc = torch.sum(torch.all((preds == y) | ~mask, dim=1)).float() / len(y)   # | ~mask ensures that we ignore padding when checking if all elements match

        logs = {'loss': loss, 'acc': symbol_acc, 'grid_acc': grid_acc, 'preds': preds, 'y': y}

        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)

        self.train_preds.append(logs.pop('preds'))
        self.train_labels.append(logs.pop('y'))

        self.log_dict({f"metrics/train_{k}": v for k,v in logs.items()}, prog_bar=True, logger=True, on_step=True, on_epoch=True)    # NOTE: this is monitored for best checkpoint and early stopping
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)

        self.val_preds.append(logs.pop('preds'))
        self.val_labels.append(logs.pop('y'))

        self.log_dict({f"metrics/val_{k}": v for k, v in logs.items()}, prog_bar=True, logger=True, on_step=True, on_epoch=True)    # NOTE: this is monitored for best checkpoint and early stopping
        self.log_dict({"learning_rate": self.lr_schedulers().get_last_lr()[-1]}, prog_bar=True, logger=True, on_step=True, on_epoch=True)    # NOTE: this is monitored for best checkpoint and early stopping. This yields learning_rate in the logs

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):

        y_hat, y, mask = self.shared_step(batch)

        # Compute the loss per token/symbol and the loss
        # per_sample_loss = F.cross_entropy(y_hat, y.long(), reduction='none').float()   # loss for each sample of the batch
        # loss = per_sample_loss.mean().unsqueeze(0)

        # Compute the loss per token/symbol and then apply the mask?
        loss = F.cross_entropy(y_hat, y.long(), reduction='none')  # [B, seq_len]
        loss = (loss * mask).sum() / mask.sum()  # only consider non-padded elements
        loss = loss.unsqueeze(0)

        # Compute predictions
        preds = torch.argmax(y_hat, dim=1)  # [B, seq_len]; predictions for each token/symbol of the model for each sample of the batch

        # Accuracy per symbol (i.e., the accuracy of the model in predicting the correct symbol for each pixel of the grid)
        # symbol_acc = (y == preds).float().mean().unsqueeze(0) # same as line below
        # symbol_acc = (torch.sum(y == preds).float() / (y.numel())).unsqueeze(0)    # NOTE: (y == preds) yields a boolean tensor of shape [B, seq_len=900]. .float() converts True to 1.0 and False to 0.0, so get the sum of correctly predicted grid cells/symbols. y.numel() returns the total number of elements in the tensor y. That is, it is equivalent to dividing by B*seq_len

        # Accuracy per symbol (ignoring padding)
        # TODO: am I considering the accuracy for the batch correctly?
        correct = (preds == y) * mask   # only consider valid positions
        symbol_acc = correct.sum().float() / mask.sum()  # only consider non-padded elements
        symbol_acc = symbol_acc.unsqueeze(0)

        # Grid accuracy (i.e., the accuracy of the model to predict the whole grid correctly)
        # grid_acc = (torch.sum(torch.all(y == preds, dim=1)).float() / len(y)).unsqueeze(0)
        # grid_acc = (torch.all(y == preds, dim=1).float().mean()).unsqueeze(0)

        # Grid accuracy (only count if entire true grid is correct) ?
        # TODO: am I considering the accuracy for the batch correctly?
        grid_acc = torch.sum(torch.all((preds == y) | ~mask, dim=1)).float() / len(y)   # | ~mask ensures that we ignore padding when checking if all elements match
        grid_acc = grid_acc.unsqueeze(0)

        logs = {'loss': loss, 'acc': symbol_acc, 'grid_acc': grid_acc}

        if dataloader_idx == 0:
            self.test_preds.append(preds)
            self.test_labels.append(y)

            results = {f"test_{k}": v for k, v in logs.items()}
            self.log_dict(results, logger=True, prog_bar=True)  # log metrics for progress bar visualization
            self.test_step_results.append(results)

        elif dataloader_idx == 1:
            self.gen_test_preds.append(preds)
            self.gen_test_labels.append(y)

            # TODO: currently in the code we assume that if there is only one dataloader, it will be considered as a test dataloader and not a gen test dataloader even though the data may be of systematic generalization. Fix this to be better maybe?
            results = {f"gen_test_{k}": v for k, v in logs.items()}
            self.log_dict(results, logger=True, prog_bar=True)  # log metrics for progress bar visualization
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

            if self.model_config.observe_preds.enabled:
                observe_image_predictions(self.test_preds, self.test_labels, self.model_config.observe_preds.n_samples)

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

            if self.model_config.observe_preds.enabled:
                observe_image_predictions(self.gen_test_preds, self.gen_test_labels, self.model_config.observe_preds.n_samples)

    def on_train_end(self):
        # Plot learning rate values used during training
        plot_lr_schedule(self.lr_values)
        return

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Override the PyTorch Lightning optimizer_step method to add custom logic before the optimizer.step() call.
        
        NOTE: We overwrite it for learning rate warm-up, as it is important for Transformer model training.
        TODO: See if ok to define the LR warm-up like this.

        """
        # Manual LR warm up
        if self.trainer.global_step < 1000:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 1000.0)
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
        
        else:
            raise ValueError(f"Unknown optimizer given: {self.model_config.training_hparams.optimizer}")

        # Define the learning rate scheduler
        if self.model_config.training_hparams.scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        elif self.model_config.training_hparams.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        optimizer_config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "metrics/val_loss",  # here write the metric to track for lr scheduling. E.g., metrics/val_loss or metrics/val_acc
            },
        }

        return optimizer_config


class REARCModel(VisReasModel):
    def __init__(self, model_config, backbone_network_config, head_network_config, image_size, **kwargs):

        # Save the hyperparameters so that they can be stored in the model checkpoint when using torch.save()
        self.save_hyperparameters() # saves all the arguments (kwargs too) of __init__() to the variable hparams

        super().__init__(model_config=model_config)

        self.model_config = model_config

        self.image_size = image_size
        self.seq_len = self.image_size * self.image_size    # the sequence length without extra tokens (but with padding)
        self.num_channels = 1

        # TODO: Should we include padding to predict due to grid size variability? Yes?
        self.num_classes = 10 + 1  # number of token categories; 1 extra token for padding as it has to be predicted

        self.backbone_input_embed_dim = backbone_network_config.embed_dim   # embedding dimension backbone model
        self.head_input_dim = self.backbone_input_embed_dim   # embedding dimension of the backbone model, usually the same as its input embedding dimension
        self.head_input_embed_dim = head_network_config.embed_dim   # dimension of the actual input that will be passed to the head network; initially assumed to be of dimension equal to the embedding dimension of the head model


        ## Model backbone/encoder
        if model_config.backbone == "resnet":
            self.model_backbone, bb_num_out_features = get_resnet(model_config=model_config, 
                                                                  network_config=backbone_network_config
                                                                  )
            self.head_input_dim = bb_num_out_features

        elif model_config.backbone == "transformer":
            self.encoder = get_transformer_encoder(model_config=self.model_config,
                                                   network_config=backbone_network_config, 
                                                   image_size=self.image_size,
                                                   num_channels=self.num_channels,
                                                   num_classes=self.num_classes, 
                                                   device=self.device
                                                   )

        elif model_config.backbone == "my_vit":
            self.encoder = get_my_vit(model_config=model_config,
                                      network_config=backbone_network_config,
                                      img_size=self.image_size,
                                      num_channels=self.num_channels,
                                      num_classes=self.num_classes,
                                      device=self.device
                                      )

        elif model_config.backbone == "looped_vit":
            raise NotImplementedError("Looped ViT not implemented yet")
        
        else:
            raise ValueError(f"Unknown model backbone given: {model_config.backbone}")
        

        ## Task embedding
        if model_config.task_embedding.enabled:
            task_embedding_dim = model_config.task_embedding.task_embedding_dim
            self.task_embedding = nn.Embedding(model_config.n_tasks, embedding_dim=task_embedding_dim, device=self.device)   # NOTE: 103 is the total number of tasks because the input is a task id (i.e., a number between 0 and 102)
            self.head_input_dim += task_embedding_dim
        else:
            task_embedding_dim = 0
            self.task_embedding = None


        ## Encoder to Decoder projection layer; useful to handle the task embedding that is concatenated
        # TODO: Should we handle the task embedding differently? For example using FiLM or other methods?
        if self.head_input_dim != self.head_input_embed_dim:
            self.enc_to_dec_proj = nn.Linear(self.head_input_dim, self.head_input_embed_dim, device=self.device)  # project the encoder output (of dimension backbone_network_config.embed_dim + task_embedding_dim) to the decoder embedding dimension


        ## Model head or decoder
        # TODO: Should we use an MLP head? Or stick to a transformer decoder/head only for REARC?
        if model_config.backbone in ["transformer", "my_vit"]:

            if model_config.head == "transformer":
                self.decoder = get_transformer_decoder(model_config=self.model_config,
                                                       network_config=head_network_config,
                                                       max_seq_len=self.seq_len,
                                                       device=self.device)
            else:
                raise ValueError(f"Unknown model head given: {model_config.head}")
            
            # Create a mask to prevent the Transformer Decoder from looking at the future tokens/positions
            # Try 1: we have -inf where we want to mask and 0 where we want to keep
            self.tgt_mask = torch.triu(torch.full((self.seq_len, self.seq_len), float("-inf"), dtype=torch.float32, device=self.device), diagonal=1)  # similar to torch's generate_square_subsequent_mask
            
            # Try 2: we have 1's where we want to mask and 0's where we want to keep
            # self.tgt_mask = torch.triu(torch.ones(self.seq_len, self.seq_len), diagonal=1).bool().to(device=self.device, dtype=torch.float32)  # [seq_len, seq_len]; masked positions are filled with True, unmasked positions are filled with False
            
            # Try 3: same as Try 1
            # self.tgt_mask = torch.triu(torch.ones(self.seq_len, self.seq_len), diagonal=1)
            # self.tgt_mask = self.tgt_mask.masked_fill(self.tgt_mask == 1, float('-inf'))

            # Create a target projection layer to map the ground truth target tokens/sequence (obtained from y) to the decoder embedding dimension as a Transformer Decoder needs to receive the target sequence in an embedding space
            self.tgt_projection = nn.Embedding(num_embeddings=self.num_classes, embedding_dim=self.head_input_embed_dim, device=self.device)


        elif model_config.backbone in ["resnet"]:
            if model_config.head == "mlp":

                self.head = get_mlp_head(network_config=head_network_config, 
                                         embed_dim=self.head_input_dim, 
                                         output_dim=self.num_classes, 
                                         activation='relu', num_layers=2)
            else:
                raise ValueError(f"Unknown model head given: {model_config.head}")
        else:
            raise ValueError(f"Unknown model backbone given: {model_config.backbone}")


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

        self.tgt_projection = self.tgt_projection.to(device=self.device)
        self.tgt_mask = self.tgt_mask.to(device=self.device)

        # self.check_device_placement()

        # Preprocess the target sequence y so that it has the dimensions [B, seq_len, head_input_embed_dim] to be used by the Transformer decoder
        tgt = y.view(B, -1)  # [B, seq_len]
        tgt = tgt.to(device=self.device, dtype=torch.long)
        tgt = self.tgt_projection(tgt)  # [B, seq_len, head_input_embed_dim], where head_input_embed_dim is the embed dim of the decoder; ensure that tgt is of type long
        tgt = tgt.to(device=self.device, dtype=torch.float32)
        
        # We use the full target sequence y as input to the decoder with a causal mask. Thus we predict the seq_len tokens of the target sequence in parallel
        # We are using full teacher forcing
        output_target_seq = self.decoder(tgt=tgt, memory=x_encoded, tgt_mask=self.tgt_mask) # [B, seq_len, head_input_embed_dim]; NOTE: PyTorch's TransformerDecoder returns the logits for each token in the tgt sequence given

        return output_target_seq

    # @timer_decorator
    def inference_decode(self, x_encoded):
        B, S, D = x_encoded.shape

        self.tgt_projection = self.tgt_projection.to(device=self.device)
        self.tgt_mask = self.tgt_mask.to(device=self.device)

        # self.check_device_placement()

        self.decoder.eval()  # set the decoder network to evaluation mode
        with torch.no_grad():
            # We use our predicted target sequence up to token at position t as input to the decoder. Thus we predict the seq_len tokens of the target sequence one by one in an auto-regressive manner
            # We using auto-regressive decoding without teacher forcing
            output_tokens = []  # list to store the output (tensor) tokens at each step. Instead of reserving memory for all timesteps at once, we append only the necessary data step-by-step, hence avoiding storing the intermediate states as using a list detaches all states from the computation graph
            
            # Create a start token used to start the AR decoding
            # TODO: Is it correct to start with a tensor of zeros as the first token? What should the start token be?
            # token_start = torch.zeros(B, 1, dtype=torch.long, device=self.device)    # [B, 1]; start with a token 0 as the first token
            start_token = 0
            token_start = torch.full((B, 1), start_token, dtype=torch.long, device=self.device)  # [B, 1]; start with a token start_token as the first token
            token_start = token_start.to(device=self.device, dtype=torch.long)

            # Embed the start token
            token_start = self.tgt_projection(token_start)  # [B, 1, head_input_embed_dim]
            token_start = token_start.to(device=self.device, dtype=torch.float32)
            
            # Store the first token in the list; note that the list should contain the embedded tokens of each AR decoding step
            output_tokens.append(token_start)
            
            # Start the iterative AR decoding loop
            for t in range(self.seq_len-1): # we have seq_len logits/predictions to get
                # TODO: Here issue with shape. It is [B, 1, 1, head_input_embed_dim] instead of [B, 1, head_input_embed_dim]. 
                prev_tokens = torch.cat(output_tokens, dim=1)   # [B, t+1, head_input_embed_dim]; create a tensor from the list and detach to save memory
                prev_tokens = prev_tokens.to(device=self.device, dtype=torch.float32)

                # Decode up to step t; in theory we do not need to give a causal mask as we are only interested in the last token logits, but we give it to avoid possible errors if we were to analyze the logits of all tokens at each step
                tgt_mask = self.tgt_mask[:t+1, :t+1]  # [t+1, t+1]; use the global causal mask for the first t+1 steps only
                tgt_mask = tgt_mask.to(device=self.device)

                # Compute the output embeddings up to position t
                outputs = self.decoder(prev_tokens, x_encoded, tgt_mask=tgt_mask) # [B, t+1, head_input_embed_dim]; NOTE: The logits returned by PyTorch's TransformerDecoder are that of all the tokens in the tgt sequence given. So below we take only that of the last step
                
                # Get the output token at step t; we need to get an actual (i.e., not just logits) token since we use AR decoding
                token_t_output = outputs[:, -1, :]   # [B, ...]; We take the logits of the last token (as it is the token of interest)
                
                # Apply the linear output layer to get the logits from the predicted target sequence embeddings
                token_t_logits = self.output_layer(token_t_output)  # [B, seq_len, num_classes] <-- [B, seq_len, head_input_embed_dim]
                
                # Get/sample the token at step t
                # TODO: Implement better sampling? E.g., Beam search, etc. to sample the output token? 
                token_t = token_t_logits.argmax(dim=-1, keepdim=True)  # [B, 1]; This extracts only the last step’s logits and finds the most probable token; Greedy decoding: take the token with the highest probability  
                token_t = token_t.to(device=self.device, dtype=torch.long)
            
                # Embed the output token of position/step t
                token_t = self.tgt_projection(token_t)  # [B, 1, head_input_embed_dim] <-- [B, 1]
                token_t = token_t.to(device=self.device, dtype=torch.float32)

                # Store in the list the (embedded) output token at position/step t
                output_tokens.append(token_t)    # TODO: Important to use .copy() here to avoid storing a reference of the tensor token_t since it is changing in the loop?
            
            # Stack the output tokens from the list to get the predicted target sequence
            output_target_seq = torch.cat(output_tokens, dim=1)   # [B, seq_len, head_input_embed_dim]

        return output_target_seq

    @timer_decorator
    def decode_sequence(self, x_encoded, y):
        # AR Decoding
        if self.training:   # use PTL LightningModule's self.training attribute to check if the model is in training mode; could also use self.trainer.training, self.trainer.validating, self.trainer.testing
            output_target_seq = self.training_decode(x_encoded, y)
        else:
            output_target_seq = self.inference_decode(x_encoded)

        return output_target_seq

    def forward(self, x, y, samples_task_id=None):
        B, H, W = x.shape

        # Encode the input sequence
        x_encoded = self.encoder(src=x)  # [B, seq_len, backbone_input_embed_dim]; NOTE: the extra tokens will have been truncated so the encoded sequence will also have a dim seq_len 
        
        # Handle the task embedding if needed
        if samples_task_id is not None:
            task_embedding = self.task_embedding(samples_task_id)   # [B, task_embedding_dim]
            task_embedding = task_embedding.unsqueeze(1).repeat(1, x_encoded.shape[1], 1) # [B, seq_len, task_embedding_dim]
            x_encoded = torch.cat([x_encoded, task_embedding], 2)  # [B, seq_len, backbone_input_embed_dim + task_embedding_dim]

        # Transformer Decoder
        if self.model_config.head in ["transformer", "my_vit"]:
            if self.head_input_dim != self.head_input_embed_dim:
                # Map the encoded input sequence to the same embedding dimension as the decoder
                x_encoded = self.enc_to_dec_proj(x_encoded)  # [B, seq_len, head_input_embed_dim]

            # Auto-regressive decoding (with full teacher forcing and causal masking for training)
            output_target_seq = self.decode_sequence(x_encoded, y)   # [B, seq_len, head_input_embed_dim]

            # Apply the linear output layer to get the logits from the predicted target sequence embeddings
            logits = self.output_layer(output_target_seq)  # [B, seq_len, num_classes]

        # MLP Decoder/Head
        elif self.model_config.head in ["mlp"]:
    
            # Forward pass through the model head
            logits = self.head(x_encoded)   # [B, seq_len, num_classes]

        return logits

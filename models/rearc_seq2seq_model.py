import torch
from torch import nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl

# Personal codebase dependencies
from networks.backbones.transformers import get_transformer_encoder
from networks.heads.transformers import get_transformer_decoder
from utility.utils import plot_lr_schedule  # noqa: F401
from utility.logging import logger


class VisReasModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

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

    def load_backbone_weights(self, checkpoint_path):
        self.model_backbone.load_state_dict(torch.load(checkpoint_path)['model'], strict=False)
        logger.info(f"Loaded ckpt weights for backbone at ckpt path: {checkpoint_path}")

    def freeze_backbone_model(self):
        for param in self.model_backbone.parameters():
            param.requires_grad = False

    def shared_step(self, batch):
        x, y, samples_task_id = batch   # Vision approach: [B, C, H, W], [B, C, H, W], [B]; Seq2Seq approach: [B, H*W], [B, H*W], [B]

        x_shape = x.shape    # Seq2Seq approach: [B, H*W]

        if self.model_config.task_embedding.enabled:
            # Enter the model forward pass with the task embeddings
            y_hat = self(x, samples_task_id)    # computed logits
        else:
            # Enter the model forward pass
            y_hat = self(x)   # computed logits

        # Permute the dimensions of y_hat to be [B, num_classes=10, seq_len] instead of [B, seq_len, num_classes=10] to match Pytorch's cross_entropy function format
        y_hat = y_hat.permute(0, 2, 1)  # [B, num_classes=10, seq_len] <-- [B, seq_len, num_classes=10]

        # Flatten y
        y = y.view(y.shape[0], -1)  # [B, seq_len=H*W=900] <-- [B, H=30, W=30]

        return y_hat, y

    def step(self, batch, batch_idx):

        y_hat, y = self.shared_step(batch)  # ([B, seq_len=900, num_classes=10], [B, H=30, W=30])

        loss = F.cross_entropy(y_hat, y.long())    # compute the loss (averaged over the batch); NOTE: F.cross_entropy() applies Softmax to the (unnormalized) logits in order to obtain the probabilities (i.e., normalized logits) and then computes the loss w.r.t. the ground-truth target labels
        # normalized_logits = F.softmax(y_hat, dim=1)  # compute the probabilities (normalized logits) of the model for each sample of the batch
        preds = torch.argmax(y_hat, dim=1)  # predictions of the model for each sample of the batch; NOTE: we do not need to compute the Softmax since the ordering of the logits will be the same as the ordering of the probabilities obtained after Softmax

        # Reshape the sequence of predictions into a grid (i.e., 2D tensor of symbols)
        # grid_preds = preds.view(preds.shape[0], 30, 30, -1)  # [B, H=30, W=30, num_classes=10]
        # grid_target = y.view(y.shape[0], 30, 30, -1)  # [B, H=30, W=30, num_classes=10]

        # TODO: should we compute the accuracy and loss w.r.t. the true ground-truth grid size, that is, without padding? 

        # Accuracy per symbol (i.e., the accuracy of the model in predicting the correct symbol for each pixel of the grid)
        symbol_acc = torch.sum(y == preds).float() / (y.numel())    # NOTE: (y == preds) yields a boolean tensor of shape [B, seq_len=900]. .float() converts True to 1.0 and False to 0.0, so get the sum of correctly predicted grid cells/symbols. y.numel() returns the total number of elements in the tensor y. That is, it is equivalent to dividing by B*seq_len

        # Grid accuracy (i.e., the accuracy of the model to predict the whole grid correctly)
        grid_acc = torch.sum(torch.all(y == preds, dim=1)).float() / len(y)     # torch.all(y == preds, dim=1) returns a boolean tensor of shape [B] where each element is True if all the symbols of the grid are correctly predicted and False otherwise. .float() converts True to 1.0 and False to 0.0. Then, we sum all the True values and divide by the total number of samples in the batch

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

        y_hat, y = self.shared_step(batch)

        per_sample_loss = F.cross_entropy(y_hat, y.long(), reduction='none').float()   # loss for each sample of the batch
        loss = per_sample_loss.mean().unsqueeze(0)
        preds = torch.argmax(y_hat, dim=1)

        # TODO: should we compute the accuracy and loss w.r.t. the true ground-truth grid size, that is, without padding? 

        # Accuracy per symbol (i.e., the accuracy of the model in predicting the correct symbol for each pixel of the grid)
        symbol_acc = (y == preds).float().mean().unsqueeze(0)
        symbol_acc = (torch.sum(y == preds).float() / (y.numel())).unsqueeze(0)    # NOTE: (y == preds) yields a boolean tensor of shape [B, seq_len=900]. .float() converts True to 1.0 and False to 0.0, so get the sum of correctly predicted grid cells/symbols. y.numel() returns the total number of elements in the tensor y. That is, it is equivalent to dividing by B*seq_len

        # Grid accuracy (i.e., the accuracy of the model to predict the whole grid correctly)
        grid_acc = (torch.sum(torch.all(y == preds, dim=1)).float() / len(y)).unsqueeze(0)
        grid_acc = (torch.all(y == preds, dim=1).float().mean()).unsqueeze(0)  

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

    def on_train_end(self):
        # Plot learning rate values used during training
        plot_lr_schedule(self.lr_values)
        return

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Override the PyTorch Lightning optimizer_step method to add custom logic before the optimizer.step() call.
        
        We use learning rate warm-up, as it is important for Transformer model training.
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


class REARCSeq2SeqModel(VisReasModel):
    def __init__(self, model_config, backbone_network_config, head_network_config, **kwargs):

        self.save_hyperparameters() # saves all the arguments (kwargs too) of __init__() to the variable hparams

        super().__init__()

        self.model_config = model_config

        self.img_size = 30
        self.num_classes = 10

        self.seq_len = self.img_size * self.img_size

        self.backbone_input_embed_dim = backbone_network_config.embed_dim   # embedding dimension backbone model
        self.head_input_dim = self.backbone_input_embed_dim   # embedding dimension of the backbone model, usually the same as its input embedding dimension
        self.head_input_embed_dim = head_network_config.embed_dim   # dimension of the actual input that will be passed to the head network; initially assumed to be of dimension equal to the embedding dimension of the head model

        # Input embedding
        self.input_embedding = nn.Embedding(self.num_classes, embedding_dim=self.backbone_input_embed_dim)

        # Model backbone or encoder
        if model_config.backbone == "transformer":
            self.encoder = get_transformer_encoder(backbone_network_config, device=self.device)

        elif model_config.backbone == "looped_vit":
            raise NotImplementedError("Looped ViT not implemented yet")
        
        else:
            raise ValueError(f"Unknown model backbone given: {model_config.backbone}")
        

        # Task embedding
        if model_config.task_embedding.enabled:
            task_embedding_dim = model_config.task_embedding.task_embedding_dim
            self.task_embedding = nn.Embedding(model_config.n_tasks, embedding_dim=task_embedding_dim)   # NOTE: 103 is the total number of tasks because the input is a task id (i.e., a number between 0 and 102)
            self.head_input_dim += task_embedding_dim
        else:
            task_embedding_dim = 0
            self.task_embedding = None

        if self.head_input_dim != self.head_input_embed_dim:
            self.enc_to_dec_proj = nn.Linear(self.backbone_input_embed_dim + task_embedding_dim, self.head_input_embed_dim)  # project the encoder output (of dimension backbone_network_config.embed_dim + task_embedding_dim) to the decoder embedding dimension


        # Model head or decoder
        if model_config.backbone in ["transformer"]:
            self.decoder = get_transformer_decoder(head_network_config, device=self.device)


        # Output layer to go from the decoder output to logits
        self.output_layer = nn.Linear(self.head_input_embed_dim, self.num_classes)


    def forward(self, x, samples_task_id=None):
        # Flatten the input grid into a sequence
        x_flat = x.view(x.shape[0], self.seq_len)  # [B, seq_len=900] <-- [B, H=30, W=30]

        # Embed the input tokens
        x_embed = self.input_embedding(x_flat.long())  # [B, seq_len, backbone_input_embed_dim]; each token/symbol has its embedding of dim backbone_input_embed_dim

        # Encode the input sequence
        x_encoded = self.encoder(x_embed)  # [B, seq_len, embed_dim]

        # Handle the task embedding if needed
        if samples_task_id is not None:
            task_embedding = self.task_embedding(samples_task_id)   # [B, task_embedding_dim]
            task_embedding = task_embedding.unsqueeze(1).repeat(1, x_encoded.shape[1], 1) # [B, seq_len, task_embedding_dim]
            x_encoded = torch.cat([x_encoded, task_embedding], 2)  # [B, seq_len, embed_dim + task_embedding_dim]

        if self.head_input_dim != self.head_input_embed_dim:
            # Map the encoded input sequence to the same embedding dimension as the decoder
            x_encoded = self.enc_to_dec_proj(x_encoded)  # [B, seq_len, embed_dim]


        # Auto-regressive decoding
        
        # # Decoding method 1: memory inefficient
        # Initialize the target sequence (start with zeros or a learned start token)
        # output_target_seq = torch.zeros(x.shape[0], self.seq_len, self.head_input_embed_dim).to(x.device)  # [B, seq_len, embed_dim (+task_embedding_dim)]
        # output_target_seq_shape = output_target_seq.shape
        # for t in range(self.seq_len):  # 900 iterations/tokens
        #     # Decode up to step t
        #     output_t = self.decoder(output_target_seq[:, :t+1, :], x_encoded)  # [B, t+1, head_input_embed_dim]

        #     # Update the target sequence with the tokens predicted so far
        #     output_target_seq[:, t, :] = output_t[:, -1, :] # [B, head_input_embed_dim]


        # Decoding method 2: memory efficient
        output_tokens = []  # list to store the output tokens at each step. Instead of reserving memory for all timesteps at once, we append only the necessary data step-by-step, hence avoiding storing the intermediate states as using a list detaches all states from the computation graph
        for t in range(self.seq_len):
            if output_tokens:
                prev_tokens = torch.stack(output_tokens, dim=1).detach()  # create a tensor from the list and detach to save memory
            else:
                prev_tokens = torch.zeros(x.shape[0], 1, self.head_input_embed_dim, device=x.device)    # [B, 1, head_input_embed_dim]

            # Decode up to step t
            output_t = self.decoder(prev_tokens, x_encoded) # [B, 1, head_input_embed_dim]

            # Store in the list the output token at step t
            output_tokens.append(output_t[:, -1, :].detach())  # detach to save memory and store the output token at step t
        
        # Stack the output tokens from the list to get the predicted target sequence
        output_target_seq = torch.stack(output_tokens, dim=1)   # [B, seq_len, head_input_embed_dim]
        output_target_seq_shape = output_target_seq.shape

        # Apply the linear output layer to get the logits from the predicted target sequence
        logits = self.output_layer(output_target_seq)  # [B, seq_len, num_classes=10]
        logits_shape = logits.shape

        return logits
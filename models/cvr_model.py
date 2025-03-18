import torch
from torch import nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl

# Personal codebase dependencies
from networks.backbones.vit import get_vit
from networks.backbones.resnet import get_resnet
from networks.heads.mlp import get_mlp_head
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
        self.model_backbone.load_state_dict(torch.load(checkpoint_path, weights_only=False)['model'], strict=False)
        logger.info(f"Loaded ckpt weights for backbone at ckpt path: {checkpoint_path}")

    def freeze_backbone_model(self):
        for param in self.model_backbone.parameters():
            param.requires_grad = False

    # TODO: optimize method for speed. E.g.: remove logging statements, remove unnecessary code such as the variable affectation of nb_images_in_one_sample and of y, put somewhere else the creation of artificial labels
    # So, create the artificial label at random for all samples before training, as doing it at each step decreases performance unnecessarily?
    def shared_step(self, batch):
        # The input batch is a tuple of 2 elements (samples, labels), where a label is the task name
        x, samples_task_id = batch  # ([B, nb_images_in_one_sample, C, H, W], B)

        x_shape = x.shape    # B x nb_images in the sample (4) x C (3) x H (128) x W (128)

        # Create artificial labels. That is, randomly permute the images in each sample so that the odd image is not always the last one (which we don't want the model to learn)
        nb_images_in_one_sample = 4 
        perms = torch.stack([torch.randperm(nb_images_in_one_sample, device=self.device) for _ in range(x_shape[0])], 0)     # for each sample in the batch, we randomly permute the four images contained in a sample
        y = perms.argmax(1)   # get the new index of the odd image in the sample for each sample of the batch
        perms = perms + torch.arange(x_shape[0], device=self.device)[:, None]*nb_images_in_one_sample
        perms = perms.flatten()
        x = x.reshape([x_shape[0]*nb_images_in_one_sample, x_shape[2], x_shape[3], x_shape[4]])[perms].reshape([x_shape[0], nb_images_in_one_sample, x_shape[2], x_shape[3], x_shape[4]])

        # logger.debug(f"Actual model input x after reshaping has dimensions: {x.size()}")    # NOTE: size should be [C, nb_images_in_one_sample, C, H, W])

        if self.model_config.task_embedding.enabled:
            # Enter the model forward pass with the task embeddings
            y_hat = self(x, samples_task_id)
        else:
            y_hat = self(x)

        # logger.debug(f"Shape of y_hat: {y_hat.shape}")
        # logger.debug(f"Shape of y: {y.shape}")

        return y_hat, y

    def step(self, batch, batch_idx):

        y_hat, y = self.shared_step(batch)

        loss = F.cross_entropy(y_hat, y)    # compute the loss (averaged over the batch)
        preds = torch.argmax(y_hat, dim=1)  # predictions of the model for each sample of the batch

        acc = (torch.sum((y == preds)).float() / len(y))  # compute the accuracy (averaged over the batch)

        logs = {"loss": loss, "acc": acc, 'preds': preds, 'y': y}

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
        per_sample_loss = F.cross_entropy(y_hat, y, reduction='none').float()   # loss for each sample of the batch
        loss = per_sample_loss.mean().unsqueeze(0)
        preds = torch.argmax(y_hat, dim=1)
        acc = (y == preds).float().mean().unsqueeze(0)

        logs = {"loss": loss, "acc": acc}

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
        
        NOTE: We use it for learning rate warm-up, as it is important for Transformer model training.
        """
        # Manual LR warm up
        if self.trainer.global_step < 500:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 500.0)
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

# Vision approach
class CVRModel(VisReasModel):

    def __init__(self, model_config, backbone_network_config, head_network_config, **kwargs):

        # Save the hyperparameters so that they can be stored in the model checkpoint when using torch.save()
        self.save_hyperparameters() # saves all the arguments (kwargs too) of __init__() to the variable hparams

        super().__init__()

        self.model_config = model_config

        self.img_size = 128
        self.num_channels = 3
        self.num_classes = 4
        self.nb_images_in_one_sample = 4

        assert self.num_classes == self.nb_images_in_one_sample, f"Number of classes ({self.num_classes}) should be equal to the number of images within one sample ({self.nb_images_in_one_sample})"

        # Model backbone or encoder
        if model_config.backbone == "resnet":
            self.model_backbone, bb_num_out_features = get_resnet(model_config, backbone_network_config)
            self.head_input_dim = bb_num_out_features

        elif model_config.backbone == "vit":
            self.model_backbone, bb_num_out_features = get_vit(model_config, backbone_network_config, self.img_size, self.num_channels, self.num_classes)
            self.head_input_dim = bb_num_out_features

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


        # Model head or decoder (depending on the backbone chosen)
        if model_config.backbone in ["resnet", "vit"]: 
            if model_config.dp_sim.enabled:
                # NOTE: the approach here from the CVR code is to give feature embeddings (obtained from the backbone model)
                # to the MLP head which will then create latent embeddings that are used to compute the pairwise dot products
                # for some sort of contrastive learning. That is why the num_classes variable is not used as the output dimension of the MLP head
                self.head_output_layer_dim = head_network_config.hidden_dim//2
            else:
                self.head_output_layer_dim = self.num_classes
        
            self.head = get_mlp_head(head_network_config, embed_dim=self.head_input_dim, output_dim=self.head_output_layer_dim, activation='relu', num_layers=2)
        
        else:
            raise ValueError(f"Unknown model backbone given: {model_config.backbone}")


    def forward(self, x, samples_task_id=None):

        x_shape = x.shape   # [B, nb_images_in_one_sample=4, C=3, H, W]

        # Reshape so that the 4 images within a sample are put on the batch size dimension so that the backbone model can process the batched samples which should have three dimensions max. (in addition to considering the batch size). That is, the 4 images wihin a sample are processed at the same time, as we should.
        x = x.reshape([x_shape[0]*self.nb_images_in_one_sample, x_shape[2], x_shape[3], x_shape[4]])    # [B*4, C, H, W]

        # Forward pass through the model backbone
        x_encoded = self.model_backbone(x)  # [B*4, nb_features_backbone]

        # Handle the task embedding if needed
        if samples_task_id is not None:
            task_embedding = self.task_embedding(samples_task_id.repeat_interleave(self.nb_images_in_one_sample))     # [B*4, embed_dim] <-- .repeat_interleave(4) is used to repeat 4 times the element in the tensor samples_task_id. This allows to associate the same embedding with each of the four images within a sample and then allow to concatenate it with the features obtained from the backbone model, as the batch dimension is B*4
            x_encoded = torch.cat([x_encoded, task_embedding], 1)  # [B*4, nb_features_backbone + embed_dim]; Note that dimensions other than the one along which it is concatenated must be the same between encoded x and the task_embedding

        if self.model_config.dp_sim.enabled:

            # Forward pass through the model head to get latent embeddings
            x = self.head(x_encoded)    # [B*4, output_layer_dim]

            # Normalize the latent embeddings (over the features dimension)
            x = nn.functional.normalize(x, dim=1)   # [B*4, output_layer_dim]

            # Compute pairwise dot products between the 4 [latent embeddings of the] images within each sample of the batch
            x = x.reshape([-1, self.nb_images_in_one_sample, self.head_output_layer_dim])    # [B, 4, output_layer_dim] <-- we get back the batch size dimension by grouping the 4 images within a sample for each sample in the batch
            x = x[:, :, None, :] * x[:, None, :, :]     # [B, 4, 4, output_layer_dim] = [B, 4, 1, output_layer_dim]*[B, 1, 4, output_layer_dim] <-- using None allows to add a dimension of size 1 at the specified dimension
            # x = x.sum(3)    # [B, 4, 4] <-- sum(3) sums across the features dimension
            x = x.mean(3)    # [B, 4, 4] <-- mean(3) averages across the features dimension
            # x = x.sum(2)    # [B, 4] <-- sum(2) sums across the dimension with the number of images in the sample (so across the images comparisons)
            x = x.mean(2)    # [B, 4] <-- mean(2) averages across the dimension with the number of images in the sample (so across the images comparisons)
            
            # To signify some sort of contrastive learning
            x = -x  # [B, num_classes]  <--  we write num_classes because by construction it is equal to the number of images within a sample
        
        else:

            # Reshape back to group the 4 images (thus also classes) within a single sample, for each sample of the batch
            x = x_encoded.reshape(x_shape[0], self.nb_images_in_one_sample, -1)  # [B, 4, nb_features]

            # Aggregate features for the 4 images within a sample, for each sample in the batch
            x = x.mean(dim=1)  # [B, nb_features]. Average pooling
            # x = x.max(dim=1)[0]  # [B, nb_features]. Max pooling

            # Forward pass through the model head
            x = self.head(x)   # [B, num_classes] == [B, output_layer_dim]

        return x

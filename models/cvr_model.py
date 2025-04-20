import os
import torch
from torch import nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl

# Personal codebase dependencies
from networks.backbones.vit_timm import get_vit_timm
from networks.backbones.vit import get_vit
from networks.backbones.looped_vit import get_looped_vit
from networks.backbones.transformer import get_transformer_encoder
from networks.backbones.resnet import get_resnet
from networks.heads.mlp import get_mlp_head
from utility.cvr.utils import plot_image_predictions
from utility.utils import plot_lr_schedule  # noqa: F401
from utility.logging import logger


class VisReasModel(pl.LightningModule):
    """
    Model module class that handles the training, validation and testing logic of the model.
    """

    def __init__(self, base_config, model_config, data_config, image_size, save_folder):
        super().__init__()

        self.base_config = base_config
        self.model_config = model_config
        self.data_config = data_config
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
        
        self.val_loss_step = []
        self.val_acc_step = []

        if data_config.validate_in_and_out_domain:
            self.gen_val_loss_step = []
            self.gen_val_acc_step = []

        # Test and OOD test (if applicable) results for logging
        self.test_step_results = [] # NOTE: this is needed to store the results of the test step for each batch (i.e., at each step), and display the final results at the end of the epoch
        if data_config.use_gen_test_set:
            self.gen_test_step_results = [] # NOTE: this is needed to store the results of the generalization test step for each batch (i.e., at each step), and display the final results at the end of the epoch

        # Learning rate values for plotting LR schedule
        self.lr_values = []

    def load_backbone_weights(self, checkpoint_path):
        self.model_backbone.load_state_dict(torch.load(checkpoint_path, weights_only=False)['model'], strict=False)
        logger.info(f"Loaded ckpt weights for backbone at ckpt path: {checkpoint_path}")

    def freeze_backbone_weights(self):
        for param in self.model_backbone.parameters():
            param.requires_grad = False

    def shared_step(self, batch):
        x, y, samples_task_id = batch  # ([B, nb_images_in_one_sample, C, H, W], [B], [B])

        x_shape = x.shape    # B x nb_images in the sample (4) x C (3) x H (128) x W (128)

        # Handle task embedding
        if not self.model_config.task_embedding.enabled:
            samples_task_id = None
        
        # Forward pass through the whole model
        y_hat = self(x, samples_task_id=samples_task_id)

        return x, y_hat, y

    def step(self, batch, batch_idx):

        x, y, samples_task_id = batch  # ([B, nb_images_in_one_sample, C, H, W], [B], [B])

        x, y_hat, y = self.shared_step(batch)

        B, N, C, H, W = x.shape

        loss = F.cross_entropy(y_hat, y)    # compute the loss (averaged over the batch)
        preds = torch.argmax(y_hat, dim=1)  # predictions of the model for each sample of the batch

        acc = (torch.sum((y == preds)).float() / len(y))  # compute the accuracy (averaged over the batch)

        logs = {"loss": loss, "acc": acc}

        return loss, logs, preds

    def training_step(self, batch, batch_idx):
        """
        This method is called for each batch during the training phase.
        This is a default PyTorch Lightning method that we override to define the training logic.
        """

        x, y, samples_task_id = batch  # ([B, nb_images_in_one_sample, C, H, W], [B], [B])

        loss, logs, preds = self.step(batch, batch_idx)

        self.train_preds.append(preds)
        self.train_targets.append(y)

        self.log_dict({f"metrics/train_{k}": v for k,v in logs.items()}, 
                      prog_bar=True, 
                      logger=True, 
                      on_step=True, 
                      on_epoch=True, 
                      add_dataloader_idx=False
                      )

        # Save two batches (of inputs, preds, targets) per epoch for plotting of images at each epoch
        if (batch_idx == 0) or (batch_idx == self.trainer.num_training_batches - 1):
            self.train_inputs.append(x)
            self.train_preds.append(preds)
            self.train_targets.append(y)

        # Save to plot locally
        self.train_loss_step.append(logs['loss'])
        self.train_acc_step.append(logs['acc'])

        # Log the current learning rate
        self.log_dict({"learning_rate": self.lr_schedulers().get_last_lr()[-1]}, 
                    prog_bar=True, 
                    logger=True, 
                    on_step=True, 
                    on_epoch=True
                    )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        This method is called for each batch during the validation phase.
        This is a default PyTorch Lightning method that we override to define the validation logic.
        
        NOTE: Currently val_loss is the monitored metric during training
        """

        x, y, samples_task_id = batch  # ([B, nb_images_in_one_sample, C, H, W], [B], [B])

        loss, logs, preds = self.step(batch, batch_idx)

        if dataloader_idx == 0:
            # Logging
            self.log_dict({f"metrics/val_{k}": v for k, v in logs.items()}, 
                          prog_bar=True, 
                          logger=True, 
                          on_step=True, 
                          on_epoch=True,
                          add_dataloader_idx=False
                          )

            # Save to plot locally
            self.val_loss_step.append(logs['loss'])
            self.val_acc_step.append(logs['acc'])

            # For the first and last batch validation batch of the epoch
            if (batch_idx == 0) or (batch_idx == self.trainer.num_val_batches[0] - 1):
                # Store batch (of inputs, preds, targets) for current epoch for plotting
                self.val_inputs.append(x)
                self.val_preds.append(preds)
                self.val_targets.append(y)

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

            # For the first and last batch validation batch of the epoch
            if (batch_idx == 0) or (batch_idx == self.trainer.num_val_batches[0] - 1):
                # Store batch (of inputs, preds, targets) for current epoch for plotting
                self.gen_val_preds.append(preds)
                self.gen_val_targets.append(y)


        return loss

    def on_train_epoch_end(self):
        """
        This method is called at the end of each training epoch.
        This is a default PyTorch Lightning method that we override to define the logic to be executed at the end of each training epoch.
        NOTE: This method is called after the on_train_epoch_end() method of the Callback class.
        """

        if self.model_config.observe_preds.enabled:
            for batch_index, split in zip([0, -1], ["train", "val"]):
                # Plot a few training and validation samples of the first and last batch seen during the epoch
                plot_image_predictions(self.save_folder, 
                                       split, 
                                       self.train_inputs, 
                                       self.train_preds, 
                                       self.train_targets, 
                                       n_samples=self.model_config.observe_preds.n_samples, 
                                       batch_index=batch_index, 
                                       epoch=self.current_epoch
                                       )

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

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
        This method is called for each batch during the testing phase.
        This is a default PyTorch Lightning method that we override to define the testing logic.
        """

        x, y, samples_task_id = batch  # ([B, nb_images_in_one_sample, C, H, W], [B], [B])

        x, y_hat, y = self.shared_step(batch)
        
        # Loss
        per_sample_loss = F.cross_entropy(y_hat, y, reduction='none').float()   # loss for each sample of the batch
        loss = per_sample_loss.mean().unsqueeze(0)
        
        # Compute predictions
        preds = torch.argmax(y_hat, dim=1)
        
        # Accuracy
        acc = (y == preds).float().mean().unsqueeze(0)

        logs = {"loss": loss, "acc": acc}

        if dataloader_idx == 0:
            # Logging
            results = {f"test_{k}": v for k, v in logs.items()}
            self.log_dict(results, logger=True, prog_bar=True, add_dataloader_idx=False)
            self.test_step_results.append(results)

            self.test_inputs.append(x)
            self.test_preds.append(preds)
            self.test_targets.append(y)

        elif dataloader_idx == 1:
            # Logging
            results = {f"gen_test_{k}": v for k, v in logs.items()}
            self.log_dict(results, logger=True, prog_bar=True, add_dataloader_idx=False)
            self.gen_test_step_results.append(results)

            self.gen_test_inputs.append(x)
            self.gen_test_preds.append(preds)
            self.gen_test_targets.append(y)
        
        return results

    def on_test_epoch_end(self):
        """
        This method is called at the end of the testing phase.
        This is a default PyTorch Lightning method that we override to define the logic to be executed at the end of the testing phase.
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
                plot_image_predictions(self.save_folder, "test", self.test_inputs, self.test_preds, self.test_targets, n_samples=self.model_config.observe_preds.n_samples, batch_index=0)
                plot_image_predictions(self.save_folder, "test", self.test_inputs, self.test_preds, self.test_targets, n_samples=self.model_config.observe_preds.n_samples, batch_index=self.trainer.num_test_batches[0]-1)

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
                    plot_image_predictions(self.save_folder, "gen_test", self.gen_test_inputs, self.gen_test_preds, self.gen_test_targets, n_samples=self.model_config.observe_preds.n_samples, batch_index=0)
                    plot_image_predictions(self.save_folder, "gen_test", self.gen_test_inputs, self.gen_test_preds, self.gen_test_targets, n_samples=self.model_config.observe_preds.n_samples, batch_index=self.trainer.num_test_batches[1]-1)

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

class CVRModel(VisReasModel):

    def __init__(self, base_config, model_config, data_config, backbone_network_config, head_network_config, image_size, save_folder):

        # Save the hyperparameters to self.hparams so that they can be stored in the model checkpoint when using torch.save()
        self.save_hyperparameters()

        super().__init__(base_config, model_config, data_config, image_size, save_folder)

        self.model_config = model_config

        self.image_size = image_size
        self.num_channels = 3
        self.num_classes = 4
        self.nb_images_in_one_sample = 4

        assert self.num_classes == self.nb_images_in_one_sample, f"Number of classes ({self.num_classes}) should be equal to the number of images within one sample ({self.nb_images_in_one_sample})"

        ## Model backbone or encoder
        if model_config.backbone == "resnet":
            self.encoder, bb_num_out_features = get_resnet(base_config=base_config,
                                                           model_config=model_config,
                                                           network_config=backbone_network_config,
                                                           image_size=self.image_size,
                                                           num_classes=self.num_classes,
                                                           )
            self.head_input_dim = bb_num_out_features

        elif model_config.backbone == "vit_timm":
            self.encoder, bb_num_out_features = get_vit_timm(base_config=base_config,
                                                             model_config=model_config,
                                                             network_config=backbone_network_config,
                                                             image_size=self.image_size,
                                                             num_channels=self.num_channels,
                                                             num_classes=self.num_classes
                                                             )
            self.head_input_dim = bb_num_out_features

        elif model_config.backbone == "transformer":
            self.encoder = get_transformer_encoder(base_config=base_config,
                                                   model_config=model_config,
                                                   network_config=backbone_network_config,
                                                   image_size=self.image_size,
                                                   num_channels=self.num_channels,
                                                   num_classes=self.num_classes,
                                                   )
            self.head_input_dim = backbone_network_config.embed_dim   # embedding dimension backbone model 
            
        elif model_config.backbone == "vit":
            self.encoder = get_vit(base_config=base_config,
                                   model_config=model_config,
                                   network_config=backbone_network_config,
                                   image_size=self.image_size,
                                   num_channels=self.num_channels,
                                   num_classes=self.num_classes,
                                   )
            self.head_input_dim = backbone_network_config.embed_dim   # embedding dimension backbone model

        elif model_config.backbone == "looped_vit":
            self.encoder = get_looped_vit(base_config=base_config,
                                   model_config=model_config,
                                   network_config=backbone_network_config,
                                   image_size=self.image_size,
                                   num_channels=self.num_channels,
                                   num_classes=self.num_classes,
                                   )
            self.backbone_input_embed_dim = backbone_network_config.embed_dim   # embedding dimension backbone model
        
        else:
            raise ValueError(f"Unknown model backbone given: {model_config.backbone}")
        

        ## Task embedding
        if model_config.task_embedding.enabled:
            task_embedding_dim = model_config.task_embedding.task_embedding_dim
            self.task_embedding = nn.Embedding(model_config.n_tasks, embedding_dim=task_embedding_dim)   # NOTE: 103 is the total number of tasks because the input is a task id (i.e., a number between 0 and 102)
            self.head_input_dim += task_embedding_dim
        else:
            task_embedding_dim = 0
            self.task_embedding = None


        ## Model head or decoder (depending on the backbone chosen)
        if model_config.head == "mlp": 
            if model_config.dp_sim.enabled:
                # NOTE: the approach here from the CVR code is to give feature embeddings (obtained from the backbone model)
                # to the MLP head which will then create latent embeddings that are used to compute the pairwise dot products
                # for some sort of contrastive learning. That is why the num_classes variable is not used as the output dimension of the MLP head
                self.head_output_layer_dim = head_network_config.hidden_dim//2
            else:
                self.head_output_layer_dim = self.num_classes
        
            self.head = get_mlp_head(head_network_config, 
                                     embed_dim=self.head_input_dim, 
                                     output_dim=self.head_output_layer_dim, 
                                     activation='relu', 
                                     num_layers=2
                                     )
        
        else:
            raise ValueError(f"Unknown model head given: {model_config.head}")


    def forward(self, x, samples_task_id=None):

        x_shape = x.shape   # [B, nb_images_in_one_sample=4, C=3, H, W]

        # Reshape so that the 4 images within a sample are put on the batch size dimension so that the backbone model can process the batched samples which should have three dimensions max. (in addition to considering the batch size). That is, the 4 images wihin a sample are processed at the same time, as we should.
        x = x.reshape([x_shape[0]*self.nb_images_in_one_sample, x_shape[2], x_shape[3], x_shape[4]])    # [B*4, C, H, W]

        # Forward pass through the model backbone
        x_encoded = self.encoder(x)  # [B*4, nb_features_backbone]

        # Handle the task embedding if applicable
        if samples_task_id is not None:
            task_embedding = self.task_embedding(samples_task_id.repeat_interleave(self.nb_images_in_one_sample))     # [B*4, embed_dim] <-- .repeat_interleave(4) is used to repeat 4 times the element in the tensor samples_task_id. This allows to associate the same embedding with each of the four images within a sample and then allow to concatenate it with the features obtained from the backbone model, as the batch dimension is B*4
            x_encoded = torch.cat([x_encoded, task_embedding], dim=1)  # [B*4, nb_features_backbone + embed_dim]; Note that dimensions other than the one along which it is concatenated must be the same between encoded x and the task_embedding

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

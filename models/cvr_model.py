import yaml
import pytorch_lightning as pl
import torch
from torch import nn as nn
from torch.nn import functional as F
from torchvision import models

# Personal codebase dependencies
from networks.vanilla_vits import get_vanilla_vit_base
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

    def load_backbone_weights(self, checkpoint_path):
        self.model_backbone.load_state_dict(torch.load(checkpoint_path)['model'], strict=False)
        logger.info(f"Loaded ckpt weights for backbone at chkpt path: {checkpoint_path}")

    def freeze_backbone_model(self):
        for param in self.model_backbone.parameters():
            param.requires_grad = False

    # TODO: optimize method for speed. E.g.: remove logging statements, remove unnecessary code such as the variable affectation of nb_images_in_one_sample and of y, put somewhere else the creation of artificial labels
    # So, create the artificial label at random for all samples before training, as doing it at each step decreases performance unnecessarily?
    def shared_step(self, batch):
        # The input batch is a tuple of 2 elements (samples, labels), where a label is the task name
        x, batch_samples_task_names = batch  # ([bs, nb_images_in_one_sample, nb_channels, H, W], bs)

        x_shape = x.shape    # bs x nb_images in the sample (4) x channels (3) x H (128) x W (128)

        # logger.debug(f"Actual model input x has dimensions: {x_shape}")

        # Create artificial labels. That is, randomly permute the images in each sample so that the odd image is not always the last one (which we don't want the model to learn)
        nb_images_in_one_sample = 4 
        perms = torch.stack([torch.randperm(nb_images_in_one_sample, device=self.device) for _ in range(x_shape[0])], 0)     # for each sample in the batch, we randomly permute the four images contained in a sample
        y = perms.argmax(1)   # get the new index of the odd image in the sample for each sample of the batch
        perms = perms + torch.arange(x_shape[0], device=self.device)[:, None]*nb_images_in_one_sample
        perms = perms.flatten()
        x = x.reshape([x_shape[0]*nb_images_in_one_sample, x_shape[2], x_shape[3], x_shape[4]])[perms].reshape([x_shape[0], nb_images_in_one_sample, x_shape[2], x_shape[3], x_shape[4]])

        # logger.debug(f"Actual model input x after reshaping has dimensions: {x.size()}")    # NOTE: size should be [batch_size, nb_images_in_one_sample, nb_of_channels, img_height, img_width])

        # Enter the model forward pass function
        if self.use_task_embedding: # NOTE: this is used to indicate to the model which task it has to consider
            y_hat = self(x, batch_samples_task_names)
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

        if dataloader_idx == 0:
            y_hat, y = self.shared_step(batch)
            per_sample_loss = F.cross_entropy(y_hat, y, reduction='none').float()   # loss for each sample of the batch
            loss = per_sample_loss.mean().unsqueeze(0)
            preds = torch.argmax(y_hat, dim=1)
            acc = (y == preds).float().mean().unsqueeze(0)

            self.test_preds.append(preds)
            self.test_labels.append(y)

            logs = {"loss": loss, "acc": acc}
            results = {f"test_{k}": v for k, v in logs.items()}
            self.log_dict(results, logger=True, prog_bar=True)  # log metrics for progress bar visualization
            self.test_step_results.append(results)

        elif dataloader_idx == 1:
            y_hat, y = self.shared_step(batch)
            per_sample_loss = F.cross_entropy(y_hat, y, reduction='none').float()   # loss for each sample of the batch
            loss = per_sample_loss.mean().unsqueeze(0)
            preds = torch.argmax(y_hat, dim=1)
            acc = (y == preds).float().mean().unsqueeze(0)

            self.gen_test_preds.append(preds)
            self.gen_test_labels.append(y)

            # TODO: currently in the code we assume that if there is only one dataloader, it will be considered as a test dataloader and not a gen test dataloader even though the data may be of systematic generalization. Fix this to be better maybe?
            logs = {"loss": loss, "acc": acc}
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


    def configure_optimizers(self):
        """ Initializes the optimizer and the learning rate scheduler. 
        The optimizer is initialized with the parameters of the model and the learning rate scheduler is initialized with the optimizer.
        
        See: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers


        Returns:
            optimizer_config (dict): A dictionary containing the optimizer and the learning rate scheduler to be used during training.
        """

        # Define the optimizer
        if self.hparams.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        
        elif self.hparams.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        
        else:
            raise ValueError(f"Unknown optimizer given: {self.hparams.optimizer}")

        # Define the learning rate scheduler
        if self.hparams.scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        elif self.hparams.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        optimizer_config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "metrics/val_loss",  # here write the metric to track for lr scheduling. E.g., metrics/val_loss or metrics/val_acc
            },
        }

        return optimizer_config

def get_backbone_network_config(model_config):
    # Load the architecture parameters of the backbone model from the <network_name>.yaml config file
    with open(f"./configs/networks/{model_config['model_backbone']}.yaml", "r") as f:
        network_config = yaml.safe_load(f)

    logger.info(f"Network config of {model_config['model_backbone']} used for backbone: {network_config}")

    return network_config


class CVRModel(VisReasModel):

    def __init__(
        self,
        model_backbone: str = 'resnet18',
        mlp_dim: int = 128,
        mlp_hidden_dim: int = 2048,
        use_task_embedding: bool = False,
        **kwargs
    ):

        self.save_hyperparameters() # saves all the arguments (kwargs too) of __init__() to the variable hparams

        super().__init__()

        self.backbone_network_config = get_backbone_network_config(self.hparams)

        self.num_classes = 4    # NOTE: this should be equal to the number of images within one sample, as they define the number of classes for the odd-one-out problem
        self.mlp_dim = mlp_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.use_task_embedding = use_task_embedding

        self.nb_images_in_one_sample = 4

        if model_backbone == "resnet18":
            self.model_backbone = models.resnet18(progress=False, weights=self.hparams.pretrained)
            nb_features = self.model_backbone.fc.in_features
            self.model_backbone.fc = nn.Identity()

        elif model_backbone == "resnet50":
            self.model_backbone = models.resnet50(progress=False, weights=self.hparams.pretrained)
            nb_features = self.model_backbone.fc.in_features
            self.model_backbone.fc = nn.Identity()

        elif model_backbone == "vanilla_vit_base":
            self.model_backbone = get_vanilla_vit_base(self.backbone_network_config, img_size=128)
            nb_features = self.model_backbone.embed_dim
            self.model_backbone.head = nn.Identity()

        elif model_backbone == "looped_vit":
            raise NotImplementedError("Looped ViT not implemented yet")
        
        else:
            raise ValueError(f"Unknown model backbone given: {model_backbone}")

        if self.use_task_embedding:
            task_embedding_dim = self.hparams.task_embedding_dim
            self.task_embedding = nn.Embedding(self.hparams.n_tasks, embedding_dim=self.hparams.task_embedding_dim)   # NOTE: 103 is the total number of tasks
        else:
            task_embedding_dim = 0
            self.task_embedding = None

        if self.hparams.activation == 'relu':
            self.activation = nn.ReLU()
        elif self.hparams.activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif self.hparams.activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation function given: {self.hparams.activation}")
        
        # TODO: see if need to use batch norm or not. Currently unused
        if self.hparams.use_batchnorm:
            self.batch_norm = nn.BatchNorm1d(nb_features)

        # NOTE: the approach here from the CVR code is to give feature embeddings (obtained from the backbone model)
        # to the MLP head which will then create latent embeddings that are used to compute the pairwise dot products
        # for some sort of contrastive learning. That is why the num_classes variable is not used as the output dimension of the MLP head
        if self.hparams.use_dp_sim:
            self.mlp = nn.Sequential(nn.Linear(nb_features+task_embedding_dim, self.mlp_hidden_dim),
                                    self.activation, 
                                    nn.Linear(self.mlp_hidden_dim, self.mlp_dim))    # NOTE: no Softmax needed here as it is included in the cross-entropy loss function

        else:
            self.mlp = nn.Sequential(nn.Linear(nb_features+task_embedding_dim, self.mlp_hidden_dim),
                                    self.activation, 
                                    nn.Linear(self.mlp_hidden_dim, self.num_classes))    # NOTE: no Softmax needed here as it is included in the cross-entropy loss function

    def forward(self, x, sample_task_name=None):

        x_shape = x.shape

        # Reshape so that the 4 images within a sample are put on the batch size dimension so that the model can process the batched samples which should have three dimensions max. (in addition to considering the batch size). That is, the 4 images wihin a sample are processed at the same time, as we should.
        x = x.reshape([x_shape[0]*self.nb_images_in_one_sample, x_shape[2], x_shape[3], x_shape[4]])

        # Forward pass through the model backbone
        x = self.model_backbone(x)
        
        if self.hparams.use_dp_sim:

            # Handle the task embedding if needed
            if sample_task_name is not None:
                x_task = self.task_embedding(sample_task_name.repeat_interleave(self.nb_images_in_one_sample))     # repeat_interleave(4) is used to repeat 4 times the elements in the tensor sample_task_name? The output x_task is a tensor of shape (batch_size * 4, embedding_dim) ?
                logger.debug(f"Shape of x_task: {x_task.size()}")
                x = torch.cat([x, x_task], 1)
                logger.debug(f"Shape of x after concatenation with task embedding x_task: {x.size()}")

            # Forward pass through the model head to get latent embeddings
            x = self.mlp(x)

            # Normalize the latent embeddings
            x = nn.functional.normalize(x, dim=1)

            # Compute pairwise dot products between the 4 [latent embeddings of the] images within each sample of the batch
            x = x.reshape([-1, self.nb_images_in_one_sample, self.mlp_dim])    # reshape to have the 4 [latent embeddings of the] images in the sample dimension
            x = (x[:, :, None, :] * x[:, None, :, :]).sum(3).sum(2)     # using None allows to add a dimension of size 1 at the specified position (i.e., 2nd and 3rd dimensions here). sum(3) sums across the features dimension, and sum(2) sums across the dimension with the number of images in the sample (so across the images comparisons)
            
            # To signify some sort of contrastive learning where lower value means better match?
            x = -x
        
        else:

            # Reshape back to group the 4 images within a single sample, for each sample of the batch
            x = x.reshape(x_shape[0], self.nb_images_in_one_sample, -1)  # [batch_size, 4, nb_features]

            # Aggregate features
            # TODO: see what approach would be appropriate here
            x = x.mean(dim=1)  # [batch_size, nb_features]. Average pooling
            # x = x.max(dim=1)[0]  # [batch_size, nb_features]. Max pooling

            # logger.warning(f"Shape of x after backbone and pooling: {x.size()}")

            # Forward pass through the model head
            x = self.mlp(x)

            # logger.warning(f"Shape of x after MLP head: {x.size()}")
            # logger.warning(f"x: {x}")

        return x

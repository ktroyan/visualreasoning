import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

# Personal codebase dependencies
from utility.rearc.utils import one_hot_encode


__all__ = ['get_resnet']

class PatchEmbed(nn.Module):
    """ 
    Embedding used for the in-context input-output pair example.
    Go from grid image to a sequence of patch embeddings.

    If patch_size=1, the input is essentially flattened (pixels) and projected linearly to the embedding dimension (by the convolutional layer).
    If patch_size>1, the input is divided into patches and each patch is projected to the embedding dimension using a convolutional layer.

    If num_token_categories is given (not None), the input is a 2D Tensor with possible token values that are represented as OHE artificial channels so that the tensor can be flattened and projected linearly (1x1 conv) to the embedding dimension.
    If num_token_categories is None, the input is a 2D Tensor for which we create an artificial single channel so that the tensor can be flattened and projected linearly (1x1 conv) to the embedding dimension.
    
    Essentially, giving num_token_categories is for REARC and BEFOREARC, not for CVR (as it is a continuous image space and not categorical values)
    """
    def __init__(self, base_config, img_size, patch_size, in_channels, embed_dim, num_token_categories=None):
        super().__init__()
        
        self.base_config = base_config

        # NOTE: The image is assumed to be square
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = (self.grid_size) ** 2

        self.patch_proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        if self.base_config.data_env in ['REARC', 'BEFOREARC']:
            x = x.unsqueeze(1).float()  # add a channel dimension for the input image: [B, C=1, H, W] <-- [B, H, W]; x is int64, so we need to convert it to float32 for the Conv2d layer (otherwise issue with the conv bias)

        x = self.patch_proj(x)  # [B, embed_dim, H_out//patch_size, W_out//patch_size], where H_out = ((H-1)/1)+1) = H and W_out = ((W-1)/1)+1) = W if the spatial dimensions have not been reduced, by having patch_size=1 and stride=patch_size
        x = x.flatten(2)        # [B, embed_dim, num_patches]; flatten the spatial dimensions to get a sequence of patches
        x = x.transpose(1, 2)   # [B, num_patches, embed_dim], where num_patches = H*W = seq_len
        return x

class ResNet(nn.Module):
    """
    Custom wrapper for the torchvision ResNet model.

    We remove some original components and perform some computation steps before and after
    the original forward pass of the model in order to get required inputs and outputs.
    """

    def __init__(self, base_config, model_config, network_config, resnet, image_size, num_classes):
        super().__init__()

        self.base_config = base_config
        self.model_config = model_config
        self.network_config = network_config

        self.image_size = image_size

        self.patch_size = self.model_config.patch_size
        self.embed_dim = network_config.embed_dim

        # Decide the number of main layers to use from the original ResNet network
        self.resnet_num_main_layers = 1   # hardcoded; number of convolutional layers to use from the original ResNet network

        ## Modify origina ResNet encoder architecture
        if self.base_config.data_env in ['REARC', 'BEFOREARC']:
            self.num_token_categories = num_classes

            self.input_projection = nn.Conv2d(in_channels=self.num_token_categories, out_channels=3, kernel_size=1, stride=1)  # 11 channels (from the one-hot encoding) to 3 channels
            
            if self.resnet_num_main_layers == 1:
                # Remove the last main layer of the original resnet
                resnet.layer2 = nn.Identity()
                resnet.layer3 = nn.Identity()
                resnet.layer4 = nn.Identity()
                self.original_num_out_features = 64

            if self.resnet_num_main_layers == 2:
                # Remove the last two main layers of the original resnet
                resnet.layer3 = nn.Identity()
                resnet.layer4 = nn.Identity()
                self.original_num_out_features = 128

            if self.resnet_num_main_layers == 3:
                # Remove the last main layer of the original resnet
                resnet.layer4 = nn.Identity()
                self.original_num_out_features = 256

            if self.resnet_num_main_layers == 4:
                self.original_num_out_features = resnet.fc.in_features   # get the original number of output features of the backbone

            # Modify the original Resnet network from torchvision in order to output the required dimensions
            resnet.avgpool = nn.Identity()  # remove spatial pooling of the original resnet as we do not want the last downsampling (followed by flattening if downsampling occurs) since we want to get one embedding per position (H,W) in the grid image

        elif self.base_config.data_env == 'CVR':
            # For CVR, we do not need to modify the original ResNet network except for the classification head
            self.num_token_categories = None
            self.original_num_out_features = resnet.fc.in_features
                    
        # Remove the original ResNet classification head
        # We thus output feature maps which can then be used for classification with the correct dimensions
        resnet.fc = nn.Identity()    # remove the final fully connected layer (i.e., classification layer or head of the original resnet)
        
        # Resnet encoder
        self.resnet = resnet

        # Projection layer to go from the embedding dimension of the original ResNet to the embedding dimension specified for our network
        self.projection_to_embed_dim = nn.Linear(self.original_num_out_features, self.network_config.embed_dim)

        # self.dropout = nn.Dropout(0.1)   # dropout layer to be used in the forward pass

        ## Patch Embeddings for the example_in_context approach
        if (
            base_config.data_env in ['REARC', 'BEFOREARC'] and
            model_config.task_embedding.enabled and
            model_config.task_embedding.approach == 'example_in_context'
        ):
            # NOTE: When patch_size=1 (i.e., consider pixels), this is equivalent to flattening the input image and projecting it to the embedding dimension
            self.patch_embed = PatchEmbed(base_config, img_size=self.image_size, patch_size=1, in_channels=1, embed_dim=self.embed_dim)


    def forward(self, x, task_tokens=None, example_in_context=None):
        """
        Modified forward pass of the original ResNet encoder.

        NOTE:
        For REARC, we need to transform our input so that it has 3 channels as expected by the torchvision ResNet model.
        Hence, we can OHE the input (the number of possible symbols is the number of channels) and then use a 1x1 convolution to transform the input to the expected number of channels.
        Otherwise, we can just add an artificial channel dimension to the input. However, the OHE approach is in general better.
        """

        B, H, W = x.shape   # [B, H, W]

        if self.base_config.data_env in ['REARC', 'BEFOREARC']:
            # Add an artificial channel dimension to the input
            # x = x.unsqueeze(1).float()    # [B, C=1, H, W] <-- [B, H, W]
            x = one_hot_encode(x, self.num_token_categories)  # add a channel dimension for the input image: [B, C=num_token_categories, H, W] <-- [B, H, W]

            # Use a 1x1 convolution to transform the input (with some number of channels) to an "image" with the expected number of channels
            x = self.input_projection(x)    # [B, C=3, H, W] <-- [B, C=num_token_categories, H, W]

            # Recreate the forward pass of the original ResNet by accessing its layers but before the flattening and avgpool
            x = self.resnet.conv1(x)        # [B, C=64, H/2, W/2]
            # x_shape = x.shape
            x = self.resnet.bn1(x)          # [B, C=64, H/2, W/2]
            # x_shape = x.shape
            x = self.resnet.relu(x)         # [B, C=64, H/2, W/2]
            # x_shape = x.shape
            x = self.resnet.maxpool(x)      # [B, C=64, H', W']; formula for H (similar for W) is floor((H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1), so it depends on the arguments of the maxpool layer

            # x_shape = x.shape

            x = self.resnet.layer1(x)       # [B, C=64, H'/4, W'/4], or 1 if the division yields a size less than 1
            x_shape = x.shape

            if self.resnet_num_main_layers >= 2:
                x = self.resnet.layer2(x)       # [B, C=128, H'/8, W'/8], or 1 if the division yields a size less than 1
                x_shape = x.shape
            
            if self.resnet_num_main_layers >= 3:
                x = self.resnet.layer3(x)   # [B, C=256, H'/16, W'/16], or 1 if the division yields a size less than 1
                x_shape = x.shape
            
            if self.resnet_num_main_layers == 4:
                x = self.resnet.layer4(x)   # [B, C=512, H'/32, W'/32], or 1 if the division yields a size less than 1
                x_shape = x.shape
            
            num_features = x_shape[1]       # C=512; number of feature maps after the last convolutional layer of the original ResNet
            h_ds = x_shape[2]               # H'/32; height of the downsampled image
            w_ds = x_shape[3]               # W'/32; width of the downsampled image

            # Dynamic upsampling
            x = nn.functional.interpolate(x, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)

            # NOTE:
            # Now, to change the embedding dimension to the desired one (self.network_config.embed_dim). 
            # We can either use a Conv2d layer (which is basically a linear projection for (batched) 2D data) or
            # we can flatten the spatial dimensions into the batch dimension and then use a Linear layer.

            # We want the image's flattened spatial dimensions to act as independent data points
            # Reshape the tensor to be able to pass it through the fully connected layer
            x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C'=512] <-- [B, C'=512, H, W]; put the channel at the end and ensure that the Tensor is contiguous
            x = x.view(B * self.image_size * self.image_size, num_features)    # [B*H*W, C'=512] <-- [B, H, W, C'=512]; flatten spatial dimensions into the batch dimension;
            
            # Linear projection to the desired embedding dimension
            x = self.projection_to_embed_dim(x)   # [B*H*W, C=self.network_config.embed_dim] <-- [B*H*W, C'=512]; linear projection to the desired embedding dimension
            
            # Reshape back to an image with spatial dimensions independent of the batch dimension and with a number of channels equal to the embedding dimension
            x = x.view(B, self.image_size, self.image_size, -1)  # [B, H, W, C=self.network_config.embed_dim] <-- [B*H*W, C=self.network_config.embed_dim]
            x = x.view(B, -1, x.shape[3])   # [B, seq_len=H*W, C=self.network_config.embed_dim] <-- [B, H, W, C=self.network_config.embed_dim]
            

        elif self.base_config.data_env == 'CVR':
            x = self.resnet(x)  # [B, C=self.original_num_out_features] <-- [B, C=3, H, W]
            x = self.projection_to_embed_dim(x)  # [B, C=self.network_config.embed_dim] <-- [B, C=self.original_num_out_features]

        # Handle the task embedding for the task_tokens approach
        if task_tokens is not None and self.base_config.data_env in ['BEFOREARC']:
            x = torch.cat((x, task_tokens), dim=1)  # [B, seq_len + num_task_tokens, embed_dim]
            
        # Handle the task embedding for the example_in_context approach
        if example_in_context is not None and self.base_config.data_env in ['BEFOREARC']:
            example_input = self.patch_embed(example_in_context[0])
            example_output = self.patch_embed(example_in_context[1])

            # Append the example input and output to the input sequence
            x = torch.cat((x, example_input, example_output), dim=1)    # [B, 3*seq_len, embed_dim]

        return x

class CustomBasicBlock(nn.Module):
    """ A smaller version of ResNet BasicBlock without aggressive downsampling since we use image grids of small size. """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1):
        super().__init__()

        # Sub-block 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)  # consider using dilation=dilation for the first conv layer in order to increase the receptive field
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Sub-block 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        x_res = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x_res)

class CustomResNet(nn.Module):
    """ 
    Custom ResNet model for the REARC and BEFOREARC tasks. 
    Think of using Average Pooling and dilation for increased receptive field.

    We do not use Max Pooling (nn.MaxPool2d(kernel_size=2, stride=2), which downsamples as: [B, C, H/2, W/2])
    for better receptive field because it is not ideal to downsample since we want to have the same spatial dimensions at the end, which means we would need to upsample
    """
    def __init__(self, base_config, model_config, network_config, image_size, num_classes):
        super().__init__()
        self.base_config = base_config
        self.model_config = model_config
        self.network_config = network_config
        self.image_size = image_size
        self.embed_dim = network_config.embed_dim
        self.num_token_categories = num_classes if base_config.data_env in ['REARC', 'BEFOREARC'] else 3    # 3 for RGB

        self.backbone = nn.Sequential(
            nn.Conv2d(self.num_token_categories, 32, kernel_size=1, stride=1, padding=0),   # initial convolutional layer
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            CustomBasicBlock(32, 64, kernel_size=3, stride=1, padding=1),
            CustomBasicBlock(64, 128, kernel_size=3, stride=1, padding=1),
            CustomBasicBlock(128, 128, kernel_size=1, stride=1, padding=0),
            CustomBasicBlock(128, 256, kernel_size=3, stride=1, padding=1), # try: dilation=2, padding=2 (since P = D * (K - 1) / 2)
            CustomBasicBlock(256, 128, kernel_size=1, stride=1, padding=0), # try: dilation=4, padding=4 (since P = D * (K - 1) / 2)
            nn.Conv2d(128, self.embed_dim, kernel_size=1, stride=1),  # project to embed_dim
        )

        self.patch_embed = None
        if (
            base_config.data_env in ['REARC', 'BEFOREARC'] and
            model_config.task_embedding.enabled and
            model_config.task_embedding.approach == 'example_in_context'
        ):
            self.patch_embed = PatchEmbed(base_config, image_size, patch_size=1, in_channels=1, embed_dim=self.embed_dim)

    def forward(self, x, task_tokens=None, example_in_context=None):
        B, H, W = x.shape

        # Create a channel dimension for the input image
        if self.base_config.data_env in ['REARC', 'BEFOREARC']:
            x = one_hot_encode(x, self.num_token_categories)    # [B, C, H, W] <-- [B, H, W]; for CVR there is already a channel dimension

        # Encode the input image by passing it through the backbone
        x = self.backbone(x)    # [B, embed_dim, H, W]

        # Flatten the spatial dimensions to get a sequence of pixels
        x = x.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)  # [B, H*W, embed_dim]

        # Handle the task embedding for the task_tokens approach
        if task_tokens is not None and self.base_config.data_env == 'BEFOREARC':
            x = torch.cat((x, task_tokens), dim=1)

        # Handle the task embedding for the example_in_context approach
        if example_in_context is not None and self.base_config.data_env == 'BEFOREARC':
            example_input = self.patch_embed(example_in_context[0])
            example_output = self.patch_embed(example_in_context[1])
            x = torch.cat((x, example_input, example_output), dim=1)

        return x

def get_resnet(base_config, model_config, network_config, image_size, num_classes):
    if network_config.network_size == 18:
        model = models.resnet18(progress=False, weights=model_config.pretrained) # get the model from torchvision
        model = ResNet(base_config, model_config, network_config, model, image_size, num_classes)   # modify the torchvision model architecture to obtain suitable backbone output dimensions
    elif network_config.network_size == 50:
        model = models.resnet50(progress=False, weights=model_config.pretrained) # get the model from torchvision
        model = ResNet(base_config, model_config, network_config, model, image_size, num_classes)   # modify the torchvision model architecture to obtain suitable backbone output dimensions
    else:
        model = CustomResNet(base_config, model_config, network_config, image_size, num_classes)    # define our custom ResNet model

    bb_num_out_features = network_config.embed_dim

    return model, bb_num_out_features

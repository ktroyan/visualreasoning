import torch
from torch import nn
from torchvision import models

# Personal codebase dependencies
from utility.rearc.utils import one_hot_encode


__all__ = ['get_resnet']


class ResNet(nn.Module):
    """
    Custom wrapper for the torchvision ResNet model.
    We perform some computation steps before and after the original forward pass of the model in order to get required inputs and outputs.
    """

    def __init__(self, base_config, model_config, network_config, resnet, image_size, num_classes):
        super().__init__()

        self.base_config = base_config
        self.model_config = model_config
        self.network_config = network_config

        self.image_size = image_size

        self.original_num_out_features = resnet.fc.in_features   # get the original number of output features of the backbone

        if self.base_config.data_env =='REARC':
            patch_size = 1
            if self.model_config.use_ohe_repr:
                self.num_channels = num_classes
            else:
                self.num_channels = 1

            self.input_projection = nn.Conv2d(in_channels=self.num_channels, out_channels=3, kernel_size=patch_size, stride=patch_size)  # 10 channels (from the one-hot encoding) to 3 channels
            
            # Modify the original Resnet network from torchvision in order to output the required dimensions
            resnet.avgpool = nn.Identity()  # remove spatial pooling of the original resnet as we do not want the last downsampling (followed by flattening if downsampling occurs) since we want to get one embedding per position (H,W) in the grid image
        
            # TODO: See if upsampling correctly? Maybe we have to use dynamic upsampling (see below in forward()) as we do not know the number of feature maps after the layer4?
            # Upsample the output to the original grid image size. 
            # The scale factor is the factor by how much we want to upsample the image.
            # Since for the original ResNet it is downsampled by a factor of at most 32 (at most because if a dimension cannot be smaller than 1), we want to upsample it by the same factor.
            # Hence the factor is image_size//H' and image_size//W' where H' and W' are the height and width of the downsampled image after layer4.
            # self.upsample = nn.Upsample(scale_factor=image_size, mode='bicubic', align_corners=False)   # bicubic interpolation is like in timm's VisionTransformer
            # self.upsample = nn.ConvTranspose2d(self.network_config.embed_dim, self.network_config.embed_dim, kernel_size=image_size, stride=image_size)   # "deconvolution" or "transposed convolution"

        # Modify the original Resnet network from torchvision in order to output the required dimensions
        resnet.fc = nn.Identity()    # remove the final fully connected layer (i.e., classification layer or head of the original resnet)
        
        self.resnet = resnet

        # Projection layer to go from the embedding dimension of the original ResNet to the embedding dimension specified for our network
        self.projection_to_embed_dim = nn.Linear(self.original_num_out_features, self.network_config.embed_dim)


    def forward(self, x):
        """
        NOTE:
        For REARC, we need to transform our input so that it has 3 channels as expected by the torchvision ResNet model.
        Hence, we can OHE the input (the number of possible symbols is the number of channels) and then use a 1x1 convolution to transform the input to the expected number of channels.
        Otherwise, we can just add an artificial channel dimension to the input. However, the OHE approach is in general better.
        """

        B = x.shape[0]

        if self.base_config.data_env == 'REARC':
            # Prepare REARC input data to be used by a ResNet network
            if self.model_config.use_ohe_repr:
                # One-hot encode the input to add an artificial channel dimension (of size equal to the number of classes) to the input
                x = one_hot_encode(x, num_token_categories=self.num_channels)   # [B, C=num_classes, H, W] <-- [B, H, W]
            else:
                # Add an artificial channel dimension (of size 1) to the input
                x = x.unsqueeze(1)   # [B, C=1, H, W] <-- [B, H, W]

            # Use a 1x1 convolution to transform the input (with some number of channels) to an "image" with the expected number of channels
            x = self.input_projection(x)    # [B, C=3, H, W] <-- [B, C, H, W]
        

            # Approach 1: does not work as there is flattening occurring after the avgpool layer and before the fully connected layer
            # x = self.resnet(x)  # [B, C=self.original_num_out_features, H/32, W/32] <-- [B, C=3, H, W]

            # Approach 2: recreate the forward pass of the original ResNet by accessing its layers but before the flattening and avgpool
            x = self.resnet.conv1(x)   # [B, C=64, H/2, W/2]
            x_shape = x.shape
            x = self.resnet.bn1(x)   # [B, C=64, H/2, W/2]
            x_shape = x.shape
            x = self.resnet.relu(x)   # [B, C=64, H/2, W/2]
            x_shape = x.shape
            x = self.resnet.maxpool(x)   # [B, C=64, H', W']; formula for H (similar for W) is floor((H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1), so it depends on the arguments of the maxpool layer

            x_shape = x.shape

            x = self.resnet.layer1(x)   # [B, C=64, H'/4, W'/4], or 1 if the division yields a size less than 1
            x_shape = x.shape
            x = self.resnet.layer2(x)   # [B, C=128, H'/8, W'/8], or 1 if the division yields a size less than 1
            x_shape = x.shape
            x = self.resnet.layer3(x)   # [B, C=256, H'/16, W'/16], or 1 if the division yields a size less than 1
            x_shape = x.shape
            x = self.resnet.layer4(x)   # [B, C=512, H'/32, W'/32], or 1 if the division yields a size less than 1
            x_shape = x.shape
            
            num_features = x_shape[1]   # C=512; number of feature maps after the last convolutional layer of the original ResNet
            h_ds = x_shape[2]   # H'/32; height of the downsampled image
            w_ds = x_shape[3]   # W'/32; width of the downsampled image

            # TODO: Upsampling approach 1: does not work (?) if we do not know the number of feature maps after the layer4 at __init__ time
            # x = self.upsample(x)    # [B, C, H, W] <-- [B, C, H'/32, W'/32]
            
            # TODO: Upsampling approach 2: dynamic upsampling
            x = nn.functional.interpolate(x, scale_factor=(self.image_size//h_ds, self.image_size//w_ds), mode="bicubic", align_corners=False) # [B, C=self.network_config.embed_dim, H, W] <-- [B, C=self.network_config.embed_dim, H'/32, W'/32]; the scale factor is the factor by how much we want to upsample the image
            
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


        return x

def get_resnet(base_config, model_config, network_config, image_size, num_classes):
    if network_config.network_size == 18:
        model = models.resnet18(progress=False, weights=model_config.pretrained) # get the model from torchvision
    
    elif network_config.network_size == 50:
        model = models.resnet50(progress=False, weights=model_config.pretrained) # get the model from torchvision

    # Modify the model architecture to obtain suitable backbone output dimensions
    model = ResNet(base_config, model_config, network_config, model, image_size, num_classes)

    bb_num_out_features = network_config.embed_dim

    return model, bb_num_out_features

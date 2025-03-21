import torch
from torch import nn
from torchvision import models

# Personal codebase dependencies
from utility.rearc.utils import one_hot_encode


__all__ = ['get_resnet']


class ResNet(nn.Module):
    """
    Custom wrapper for the torchvision ResNet model.
    Essentially, we perform some steps before the original forward pass of the model.
    """

    def __init__(self, base_config, resnet, image_size, num_classes):
        super().__init__()

        self.base_config = base_config

        if self.base_config.data_env =='REARC':
            self.num_token_categories = num_classes
            patch_size = 1
            self.input_projection = nn.Conv2d(in_channels=self.num_token_categories, out_channels=3, kernel_size=patch_size, stride=patch_size)  # 10 channels (from the one-hot encoding) to 3 channels
        
        self.resnet = resnet

        self.seq_len = image_size * image_size        

    def forward(self, x):
        """
        For REARC, we need to transform our input so that it has 3 channels as expected by the torchvision ResNet model.
        Hence, we can OHE the input (the number of possible symbols is the number of channels) and then use a 1x1 convolution to transform the input to the expected number of channels.
        """

        if self.base_config.data_env == 'REARC':
            if self.num_token_categories is not None:
                # One-hot encode the input
                x = one_hot_encode(x, num_token_categories=self.num_token_categories)
            else:
                x = x.unsqueeze(1)  # add a channel dimension to the input

            # Use a 1x1 convolution to transform the input to the expected number of channels
            x = self.input_projection(x)
        
        x_encoded = self.resnet(x)

        return x_encoded

def get_resnet(base_config, model_config, network_config, image_size, num_classes, device):
    if network_config.network_size == 18:
        model = models.resnet18(progress=False, weights=model_config.pretrained) # get the model from torchvision
    
    elif network_config.network_size == 50:
        model = models.resnet50(progress=False, weights=model_config.pretrained) # get the model from torchvision

    bb_num_out_feature = model.fc.in_features   # get the number of output features of the backbone
    model.fc = nn.Identity()    # remove the final classification layer (i.e., the head)
    
    # Wrap the model with the custom wrapper
    if model_config.use_ohe_repr:
        model = ResNet(base_config, model, image_size, num_classes).to(device)
    else:
        model = ResNet(base_config, model, image_size, num_classes).to(device)

    return model, bb_num_out_feature

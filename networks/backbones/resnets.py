from torch import nn
from torchvision import models

__all__ = ['get_resnet']

def get_resnet(model_config, backbone_network_config):
    if backbone_network_config.network_size == 50:
        model = models.resnet18(progress=False, weights=model_config.pretrained) # get the model from torchvision
    
    elif backbone_network_config.backbone_network.network_size == 18:
        model = models.resnet50(progress=False, weights=model_config.pretrained) # get the model from torchvision

    bb_num_out_feature = model.fc.in_features   # get the number of output features of the backbone
    model.fc = nn.Identity()    # remove the final classification layer (i.e., the head)
    
    return model, bb_num_out_feature

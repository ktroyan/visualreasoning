# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: the latest Facebook Research code can be found at https://github.com/facebookresearch/moco-v3/blob/main/vits.py

import math
import torch
import torch.nn as nn
from functools import partial, reduce
from operator import mul
import matplotlib.pyplot as plt

from timm.models.vision_transformer import VisionTransformer, _cfg, init_weights_vit_moco
from timm.layers import PatchEmbed

# Personal codebase dependencies
from utility.logging import logger
from utility.utils import plot_absolute_positional_embeddings


__all__ = ['get_vit_timm']

class VitTimm(VisionTransformer):
    """
    Vanilla Vision Transformer (ViT) model from the timm library.
    This code is principally used for safety check reproduction of the results of the CVR paper. 

    Args:
        VisionTransformer (nn.Module): VisionTransformer class from timm library
    """
    def __init__(self, 
                 img_size,
                 num_channels,
                 num_classes,
                 patch_size,
                 embed_dim,
                 num_layers,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 norm_layer=None,
                 ape='',
                 use_cls_token=True,
                 reg_tokens=0,
                 cls_aggreg_method='',
                 ):
        
        super().__init__(img_size=img_size,
                         in_chans=num_channels,
                         num_classes=num_classes,
                         patch_size=patch_size,
                         embed_dim=embed_dim,
                         depth=num_layers,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias,
                         norm_layer=norm_layer,
                         pos_embed='',  # NOTE: the only choice from timm is 'learn' or '' (?). Hence we handle the APE below.
                         class_token=use_cls_token,
                         global_pool=cls_aggreg_method
                         )
        
        self.use_cls_token = use_cls_token
        self.num_prefix_tokens = 1+reg_tokens if use_cls_token else reg_tokens  # NOTE: the cls and reg tokens are defined as cls_token and reg_token respectively in the timm VisionTransformer class

        # TODO: Do similar as what I did in REARCModel to get the selected APE
        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # Weights initialization
        self._init_weights()

    # 2D sin-cos PE
    # NOTE: taken and updated from CVR code and thus from the Facebook Research repo (https://github.com/facebookresearch/moco-v3/blob/main/vits.py)
    # NOTE: see also the PE's in timm.layers.pos_embed_sincos.py (e.g. build_sincos2d_pos_embed)
    # NOTE: we overwrite the variable self.pos_embed defined in the VisionTransformer class of the timm library
    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        # NOTE: https://github.com/facebookresearch/moco-v3/issues/36Note
        # Handle [cls] token
        if self.use_cls_token:
            assert self.num_prefix_tokens == 1, 'Assuming one and only one token, [cls]'
            pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)     # define the PE of the [cls] token to be concatenated at the beginning of the PE of the patches
            self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))    # [1, (num_patches+1), embed_dim]; the PE will be added to the input embeddings of the ViT in the forward pass of the ViT backbone (VisionTransformer)
        else:
            self.pos_embed = nn.Parameter(pos_emb)    # [1, num_patches, embed_dim]; the PE will be added to the input embeddings of the ViT in the forward pass of the ViT backbone (VisionTransformer)
        
        self.pos_embed.requires_grad = False    # NOTE: set to False for the PE to not be learned

        # Visualize PE
        plot_absolute_positional_embeddings(self.pos_embed, self.num_prefix_tokens)


    def _init_weights(self,):
        for name, m in self.named_modules():
            init_weights_vit_moco(m, name)
            # self.init_weights_vit_timm(m, name)

        # NOTE: not sure if needed or already done by init_weights_vit_moco
        if self.use_cls_token:
            nn.init.normal_(self.cls_token, std=1e-6)

        # NOTE: Patch embedding initialization as per CVR code and thus from the Facebook Research repo (https://github.com/facebookresearch/moco-v3/blob/main/vits.py)
        if isinstance(self.patch_embed, PatchEmbed):
            # self.patch_embed.proj.weight gives the weights of the linear layer that projects the flattened patches to the embedding dimension
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)


def get_vit_timm(base_config, model_config, network_config, image_size, num_channels, num_classes):    

    ## Method 1 to get the model
    # model = timm.create_model(
    #     vit_model_name,
    #     pretrained=False,  # use pre-trained weights
    #     in_chans=3,  # number of input channels (1 for grayscale images, 3 for RGB)
    #     num_classes=0,  # no final classification head and thus get feature embeddings
    # )

    ## Method 2 to get the model
    model = VitTimm(
        img_size=image_size,
        num_channels=num_channels,
        num_classes=num_classes,
        patch_size=model_config['patch_size'], 
        embed_dim=network_config['embed_dim'], 
        num_layers=network_config['num_layers'],
        num_heads=network_config['num_heads'], 
        mlp_ratio=network_config['mlp_ratio'], 
        qkv_bias=network_config['qkv_bias'],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        ape=model_config['ape_type'],
        use_cls_token=model_config['use_cls_token'],  # whether to use or not the cls token
        reg_tokens=model_config['num_reg_tokens'],  # number of register tokens
        cls_aggreg_method=model_config['encoder_aggregation']['method'],  # "" for no global pooling to get a sequence as output, "mean" for global average pooling, "token" for using the [cls] token to aggregate features
        )

    default_cfg = _cfg()

    model.default_cfg = default_cfg

    bb_num_out_features = model.embed_dim   # we can also do: model.num_features
    model.head = nn.Identity()  # remove the final classification layer (i.e., the head); this ensures that the model outputs features (i.e., patch embeddings, or token embeddings if patch size is 1) instead of class logits

    return model, bb_num_out_features

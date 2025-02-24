# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: the latest code can be found at https://github.com/facebookresearch/moco-v3/blob/main/vits.py
# The CVR authors used FAIR code in 2022.
# I also modified the code.

import math
import torch
import torch.nn as nn
from functools import partial, reduce
from operator import mul

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.layers import PatchEmbed

__all__ = [
    'get_vanilla_vit_small',
    'get_vanilla_vit_base',
]

class VanillaVit(VisionTransformer):
    def __init__(self, stop_grad_conv1=False, **kwargs):
        super().__init__(**kwargs)

        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # Weights initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(
                        6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(
                6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

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
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(
            out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        # NOTE: https://github.com/facebookresearch/moco-v3/issues/36Note
        assert self.num_prefix_tokens == 1, 'Assuming one and only one token, [cls]'
        # assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False


def get_vanilla_vit_base(bb_net_config, **kwargs):    

    model = VanillaVit(
        patch_size=bb_net_config['patch_size'], 
        embed_dim=bb_net_config['embed_dim'], 
        depth=bb_net_config['depth'], 
        num_heads=bb_net_config['num_heads'], 
        mlp_ratio=bb_net_config['mlp_ratio'], 
        qkv_bias=bb_net_config['qkv_bias'],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)

    model.default_cfg = _cfg()
    return model

# NOTE: the only difference with base vanilla vit is the embedding dimension
def get_vanilla_vit_small(bb_net_config, **kwargs):

    model = VanillaVit(
        patch_size=bb_net_config['patch_size'], 
        embed_dim=bb_net_config['embed_dim'], 
        depth=bb_net_config['depth'], 
        num_heads=bb_net_config['num_heads'], 
        mlp_ratio=bb_net_config['mlp_ratio'], 
        qkv_bias=bb_net_config['qkv_bias'],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    model.default_cfg = _cfg()
    return model

"""
Vision Transformer (ViT) (encoder).
It works for REARC, BEFOREARC, and CVR.
It supports different types of: 2D APE, OPE, PE Mixer, RPE.
It supports register tokens.
"""

import math
import torch
import torch.nn as nn
# from torchtune.modules.position_embeddings import VisionRotaryPositionalEmbeddings
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from einops import rearrange
import matplotlib.pyplot as plt
import torch.nn.functional as F


# Personal codebase dependencies
from utility.utils import plot_absolute_positional_embeddings
from utility.rearc.utils import one_hot_encode
from utility.custom_logging import logger

# For flash / memory-efficient kernels when available (e.g., A100, RTX40xx, etc.), otherwise
# use math kernels for older GPUs.
torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)


__all__ = ['get_vit']


class PatchEmbed(nn.Module):
    """ 
    Input image to Patch Embeddings.

    If patch_size=1, the input is essentially flattened (pixels) and projected linearly to the embedding dimension (by the convolutional layer).
    If patch_size>1, the input is divided into patches and each patch is projected to the embedding dimension using a convolutional layer.

    If num_token_categories is given (not None), the input is a 2D Tensor with possible token values that are represented as OHE artificial channels so that the tensor can be flattened and projected linearly (1x1 conv) to the embedding dimension.
    If num_token_categories is None, the input is a 2D Tensor for which we create an artificial single channel so that the tensor can be flattened and projected linearly (1x1 conv) to the embedding dimension.
    
    Eseentially, giving num_token_categories is for REARC and BEFOREARC, not for CVR (as it is a continuous image space and not categorical values)
    """
    def __init__(self, base_config, img_size, patch_size, in_channels, embed_dim, num_token_categories=None):
        super().__init__()
        
        self.base_config = base_config
        if self.base_config.data_env in ['REARC', 'BEFOREARC']:
            self.num_token_categories = num_token_categories
            if self.num_token_categories is not None:
                in_channels = self.num_token_categories
        else:
            self.num_token_categories = None

        # NOTE: The image is assumed to be square
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = (self.grid_size) ** 2

        # TODO: For CVR, consider a stride smaller than the patch to get overlapping patches that may be useful for their types of objects
        self.patch_proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        if self.base_config.data_env in ['REARC', 'BEFOREARC']:
            if self.num_token_categories is not None:
                x = one_hot_encode(x, self.num_token_categories)  # add a channel dimension for the input image: [B, C=num_token_categories, H, W] <-- [B, H, W]
            else:
                x = x.unsqueeze(1).float()  # add a channel dimension for the input image: [B, C=1, H, W] <-- [B, H, W]; x is int64, so we need to convert it to float32 for the Conv2d layer (otherwise issue with the conv bias)

        x = self.patch_proj(x)    # [B, embed_dim, H_out//patch_size, W_out//patch_size], where H_out = ((H-1)/1)+1) = H and W_out = ((W-1)/1)+1) = W if the spatial dimensions have not been reduced, by having patch_size=1 and stride=patch_size
        x = x.flatten(2) # [B, embed_dim, num_patches]; flatten the spatial dimensions to get a sequence of patches
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim], where num_patches = H*W = seq_len
        return x

class AbsolutePositionalEncoding(nn.Module):
    """
    Absolute Positional Encoding (APE) with PEMixer.
    
    Support APEs:
        - learned APE
        - 2D sin-cos APE
        - 2D sin-cos APE with OPE (Object Positional Encoding)

    Supported PE mixer strategies:
        - 'hardcoded_normalization'
        - 'learnable_scaling'
        - 'weighted_sum'
        - 'weighted_sum_no_norm'
        - 'learnable_scaling_vec'
        - 'weighted_sum_vec'
        - 'weighted_sum_no_norm_vec'
        - 'layer_norm'
        - 'sum' (simple addition of the APE to the input embeddings)

    NOTE: Code for PEMixer adapted from the ViTARC repo. See their original class ViTARCEmbedding.
    """

    def __init__(self, model_config, network_config, image_size, seq_len, patch_embed, embed_dim, num_prepended_extra_tokens, ape_type, mixer_strategy='sum'):
        super().__init__()

        self.model_config = model_config
        self.network_config = network_config
        self.image_size = image_size
        self.seq_len = seq_len
        self.patch_embed = patch_embed
        self.num_prepended_extra_tokens = num_prepended_extra_tokens
        self.embed_dim = embed_dim

        self.mixer_strategy = mixer_strategy

        ## APE (static creation, as there is no OPE)
        if not model_config.ope.enabled:
            # We can already create the APE as it does not have to changed depending on the input sample
            self.static_ape = self.create_ape(image_size=image_size,
                                              seq_len=seq_len,
                                              patch_embed=patch_embed,
                                              embed_dim=embed_dim,
                                              num_prepended_extra_tokens=num_prepended_extra_tokens,
                                              ape_type=ape_type
                                              )

        ## PE Mixer
        if self.mixer_strategy in ['learnable_scaling_vec', 'weighted_sum_vec', 'weighted_sum_no_norm_vec']:
            self.position_scale = nn.Parameter(torch.ones(1, embed_dim))
            self.input_weight = nn.Parameter(torch.ones(1, embed_dim))
            self.position_weight = nn.Parameter(torch.ones(1, embed_dim))

        if self.mixer_strategy in ['learnable_scaling', 'weighted_sum', 'weighted_sum_no_norm']:
            self.position_scale = nn.Parameter(torch.ones(1))
            self.input_weight = nn.Parameter(torch.ones(1))
            self.position_weight = nn.Parameter(torch.ones(1))

        if self.mixer_strategy == 'layer_norm':
            self.layer_norm = nn.LayerNorm(embed_dim)


        ## Dropout post APE
        self.ape_drop = nn.Dropout(p=self.network_config.ape_dropout_rate)


    def create_ape(self, image_size, seq_len, patch_embed, embed_dim, num_prepended_extra_tokens, ape_type) -> nn.Parameter:
        """ 
        Create and return the desired APE (absolute positional embedding).
        The APE will be added to the input embeddings of the backbone/encoder mode (e.g., ViT) during the forward pass.
        
        NOTE:
        The PE is used for the grid tokens comprising special types of tokens to predict (e.g., pad tokens).
        However, the extra tokens (e.g., [cls] and register tokens) are (currently) not considered for the PE.
        The PE should still be of the correct size since it is added to the whole input embeddings which include the extra tokens.
        Currently, we only need to consider the prepended extra tokens as the APE is added to the input embeddings before any token is possibly appended.

        """
        
        num_pos_embeds = seq_len + num_prepended_extra_tokens

        if ape_type == 'learn':    # learned APE

            pos_embed_learned = nn.Parameter(torch.randn(1, seq_len, embed_dim))  # [1, seq_len, embed_dim]; by default requires_grad=True

            if num_prepended_extra_tokens > 0:
                # Set the PE for the extra tokens to zero (i.e., no PE for the extra tokens since PE is added to the input embeddings and the PE is not learned)
                pos_embed_extra_tokens = torch.zeros(1, num_prepended_extra_tokens, embed_dim, dtype=torch.float32, requires_grad=False)     # define the PE for the [cls] and register tokens to be concatenated at the beginning of the PE for the patches
                pos_embed = torch.cat([pos_embed_extra_tokens, pos_embed_learned], dim=1)   # [1, num_pos_embeds, embed_dim]
            
            else:
                pos_embed = pos_embed_learned

        elif ape_type == '2dsincos':    # fixed 2D sin-cos APE 

            # PE dimension per axis and sin/cos
            assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
            pos_dim = embed_dim // 4

            # Frequencies
            omega = 1. / (10000** (torch.arange(pos_dim, dtype=torch.float32) / pos_dim))   # [pos_dim]
    
            # Create grid xy coordinates
            H, W = (patch_embed.grid_size, patch_embed.grid_size)   # assume square input
            seq_h = torch.arange(H, dtype=torch.float32)  # [H]; we will want y/vertical/rows
            seq_w = torch.arange(W, dtype=torch.float32)  # [W]; we will want x/horizontal/cols
            grid_y, grid_x = torch.meshgrid(seq_h, seq_w, indexing='ij')  # [H, W], [H, W]; ij indexing for usual/consistent image layout, so row is y, col is x

            grid_x_flat = grid_x.flatten()  # [seq_len]
            grid_y_flat = grid_y.flatten()  # [seq_len]

            # Compute the input values for sincos and xy positions
            out_x = torch.einsum('m,d->md', grid_x_flat, omega)  # [seq_len, pos_dim]; outer product
            out_y = torch.einsum('m,d->md', grid_y_flat, omega)  # [seq_len, pos_dim]; outer product

            # Concatenate the computed sin/cos of the x and y positions
            pos_embed = torch.cat([torch.sin(out_x),
                                   torch.cos(out_x),
                                   torch.sin(out_y),
                                   torch.cos(out_y)], dim=1)  # [seq_len, embed_dim]
            
            pos_embed = pos_embed.unsqueeze(0)  # [1, seq_len, embed_dim]; create a batch dimension so that it can be used for batched samples

            # Handle extra tokens (e.g., cls, register tokens)
            if num_prepended_extra_tokens > 0:
                extra_pe = torch.zeros([1, num_prepended_extra_tokens, embed_dim], dtype=torch.float32)
                pos_embed = nn.Parameter(torch.cat([extra_pe, pos_embed], dim=1))  # [1, seq_len+num_prepended_extra_tokens, embed_dim]; concatenate along sequence dimension
            else:
                pos_embed = nn.Parameter(pos_embed) # [1, seq_len, embed_dim]

            # Ensure PE is fixed/non-learnable
            pos_embed.requires_grad = False
            
        else:
            raise ValueError(f"Invalid positional encoding type: {ape_type}. Choose from ['learn', '2dsincos']")

        # Visualize PE
        plot_absolute_positional_embeddings(pos_embed, num_prepended_tokens=num_prepended_extra_tokens)

        return pos_embed    # [1, seq_len(+num_prepended_extra_tokens), embed_dim]

    def create_ape_with_ope(self, image_size, seq_len, patch_embed, embed_dim, num_prepended_extra_tokens, x_grid_object_ids) -> nn.Parameter:
        """
        Dynamic PE.
        Build 2D Sin-Cos Absolute Positional Embeddings (APE) + Object Positional Encoding (OPE).

        seq_len is the number of positions to consider for the positional encoding, without the extra tokens.
        For example, the length of the input grid with all sorts of special (not extra) tokens added, padded to a fixed size and flattened.
        
        The object IDs grid x_grid_object_ids is [B, seq_len] (i.e., already flattened).

        Note that this method computes an APE (with OPE considered) for each input sample in the batch since
        it is a dynamic APE computed (i.e., computed during each forward pass, and for each sample of a given batch).

        TODO:
        Check if how I attribute the dimensions is correct.
        """

        num_pos_embeds = seq_len + num_prepended_extra_tokens

        # Check if x_grid_object_ids is flattened in the spatial dimensions
        assert x_grid_object_ids.ndim == 2, "x_grid_object_ids should be a 2D tensor [B, seq_len]"
        
        B = x_grid_object_ids.shape[0]
        device = x_grid_object_ids.device

        # Allocate the embedding dimension for the xy positions and the object positions
        # We want embed_dim = embed_dim//2 (object) + embed_dim//4 (x) + embed_dim//4 (y)
        # But each sin,cos pair gets half of their allocation, so we divide more by 2
        assert embed_dim % 8 == 0, 'Embed dimension must be divisible by 8 for 2D sin-cos PE with OPE'
        obj_dim = embed_dim // 4    # dimension for the object PE for each sinusoid
        xy_pos_dim = embed_dim // 8 # dimension for the x and y positions for each sinusoid

        # Frequencies
        omega_xy_pos = (1. / (10000 ** (torch.arange(xy_pos_dim, dtype=torch.float32) / xy_pos_dim))).to(device)    # [embed_dim//8]
        omega_obj = (1. / (10000 ** (torch.arange(obj_dim, dtype=torch.float32) / obj_dim))).to(device)             # [embed_dim//4]

        # Create grid xy coordinates
        H, W = (patch_embed.grid_size, patch_embed.grid_size)  # assume square input
        seq_h = torch.arange(H, dtype=torch.float32)   # [B, H]; for y-coordinates
        seq_w = torch.arange(W, dtype=torch.float32)   # [B, W]; for x-coordinates
        grid_x, grid_y = torch.meshgrid(seq_w, seq_h, indexing='ij')  # indexing 'ij' for (x,y) instead of (row,col), and repeat the coordinates across rows and columns respectively

        grid_x_flat = grid_x.flatten().to(device)
        grid_y_flat = grid_y.flatten().to(device)

        # 2D-sincos APE for x and y positions (same across batch samples)
        out_x = torch.einsum('n,d->nd', grid_x_flat, omega_xy_pos)      # [seq_len, embed_dim//8]; outer product; frequency-transformed positions for x
        out_y = torch.einsum('n,d->nd', grid_y_flat, omega_xy_pos)      # [seq_len, embed_dim//8]; outer product; frequency-transformed positions for y
        x_pe = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)   # [seq_len, embed_dim//4]; x-coordinate encoding
        y_pe = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)   # [seq_len, embed_dim//4]; y-coordinate encoding

        # Expand to batch
        x_pe = x_pe.unsqueeze(0).expand(B, -1, -1)  # [B, seq_len, embed_dim//4]
        y_pe = y_pe.unsqueeze(0).expand(B, -1, -1)  # [B, seq_len, embed_dim//4]

        # OPE (per sample in the batch)
        x_grid_object_ids = x_grid_object_ids.to(dtype=torch.float32)           # [B, seq_len]
        out_obj = x_grid_object_ids.unsqueeze(-1) * omega_obj                   # [B, seq_len, embed_dim//4]
        obj_pe = torch.cat([torch.sin(out_obj), torch.cos(out_obj)], dim=-1)    # [B, seq_len, embed_dim//2]

        # Combined PE: APE with OPE
        pos_embed = torch.cat([obj_pe, x_pe, y_pe], dim=-1)  # [B, seq_len, embed_dim]; NOTE: ViTARC prepends the OPE

        # Handle extra tokens (e.g., cls, register tokens). 
        # We simply increase the number of positions in the PE, so we don't create a useful PE for the extra tokens
        if num_prepended_extra_tokens > 0:
            extra_pe = torch.zeros((B, num_prepended_extra_tokens, embed_dim), dtype=torch.float32, device=pos_embed.device)
            pos_embed = torch.cat([extra_pe, pos_embed], dim=1)  # [B, num_prepended_extra_tokens + seq_len, embed_dim]

        # Make sure the APE (with OPE) is fixed/non-learnable
        pos_embed = nn.Parameter(pos_embed, requires_grad=False)  # [B, seq_len (+num_prepended_extra_tokens), embed_dim]; fixed/non-learnable

        # Visualize PE
        # NOTE: Should not uncomment this line if the goal is not to observe the PE, as this would plot for each batch during training
        # plot_absolute_positional_embeddings(pos_embed, num_prefix_tokens=num_prepended_extra_tokens)

        return pos_embed


    def forward(self, inputs_embeds, x_grid_object_ids):
        """
        Args:
            inputs_embeds (torch.Tensor): [batch_size, seq_len, embed_dim]
            x_grid_object_ids (torch.Tensor): [batch_size, seq_len, 1] or [batch_size, seq_len]

        Returns:
            output_embeds (torch.Tensor): [batch_size, seq_len, embed_dim]
        """

        # Get the APE
        if x_grid_object_ids is not None:
            # Dynamic creation
            position_embeds = self.create_ape_with_ope(image_size=self.image_size,
                                                       seq_len=self.seq_len,
                                                       patch_embed=self.patch_embed,
                                                       embed_dim=self.embed_dim,
                                                       num_prepended_extra_tokens=self.num_prepended_extra_tokens,
                                                       x_grid_object_ids=x_grid_object_ids
                                                       )    # [batch_size, seq_len, embed_dim]; created for each sample in the batch
        else:
            # Static creation
            position_embeds = self.static_ape   # [1, seq_len, embed_dim]; create for a single sample, so need to be expanded to the batch size
            position_embeds = position_embeds.expand(inputs_embeds.shape[0], -1, -1)  # [batch_size, seq_len, embed_dim]

        position_embeds = position_embeds.to(inputs_embeds.device)  # make sure the PE is on the same device as the input embeddings

        # PE Mixer strategy
        strategy = self.mixer_strategy

        if strategy == 'hardcoded_normalization':
            inputs_embeds_norm = F.normalize(inputs_embeds, p=2, dim=-1)
            position_embeds_norm = F.normalize(position_embeds, p=2, dim=-1)
            output_embeds = inputs_embeds_norm + position_embeds_norm

        elif strategy in ['learnable_scaling', 'learnable_scaling_vec']:
            scaled_position_embeds = self.position_scale * position_embeds
            output_embeds = inputs_embeds + scaled_position_embeds

        elif strategy in ['weighted_sum', 'weighted_sum_vec']:
            inputs_embeds_norm = F.normalize(inputs_embeds, p=2, dim=-1)
            position_embeds_norm = F.normalize(position_embeds, p=2, dim=-1)
            output_embeds = (self.input_weight * inputs_embeds_norm) + (self.position_weight * position_embeds_norm)

        elif strategy in ['weighted_sum_no_norm', 'weighted_sum_no_norm_vec']:
            output_embeds = (self.input_weight * inputs_embeds) + (self.position_weight * position_embeds)

        elif strategy == 'layer_norm':
            combined_embeds = inputs_embeds + position_embeds
            output_embeds = self.layer_norm(combined_embeds)

        elif strategy == 'sum':
            output_embeds = inputs_embeds + position_embeds

        else:
            raise ValueError(f"Unsupported mixer_strategy: {strategy}")
        
        # Dropout after positional encoding
        output_embeds = self.ape_drop(output_embeds)

        return output_embeds

class MHSA(nn.Module):
    """ (Vanilla) Multi-Head Self-Attention block. """

    def __init__(self, model_config, image_size, embed_dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0., num_prepended_extra_tokens=0):
        super().__init__()

        self.model_config = model_config
        self.image_size = image_size
        self.num_prepended_extra_tokens = num_prepended_extra_tokens

        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)  # W_qkv; *3 because we want embeddings for q, k, v from the input; this is equivalent to defining three separate linear layers (W_q, W_k, W_v) for q, k, and v
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim) # W_o
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, seq_len, embed_dim = x.shape
        
        # Compute the Queries, Keys, Values from the input embeddings by a linear projection
        x_qkv = self.qkv_proj(x) # [B, seq_len, 3*embed_dim]

        # Reshape the Queries, Keys, Values for multi-head
        head_embed_dim = embed_dim // self.num_heads
        x_qkv = x_qkv.reshape(B, seq_len, 3, self.num_heads, head_embed_dim).permute(2, 0, 3, 1, 4)  # [3, B, num_heads, seq_len, head_embed_dim]

        # Get the Queries, Keys, Values for all heads
        x_q, x_k, x_v = x_qkv[0], x_qkv[1], x_qkv[2]    # ([B, num_heads, seq_len, head_embed_dim], [B, num_heads, seq_len, head_embed_dim], [B, num_heads, seq_len, head_embed_dim])

        # # Method 1: Raw compute of the attention scores
        # # Compute the attention scores
        # attn = (x_q @ x_k.transpose(-2, -1))    # [B, num_heads, seq_len, seq_len]; attention matrix/logits
        # attn_scaled = attn * self.scale   # [B, num_heads, seq_len, seq_len]; scaled attention logits
        # # NOTE: no masking
        # attn_scores = self.softmax(attn_scaled)   # [B, num_heads, seq_len, seq_len]; attention scores/weights
        # attn_scores = self.attn_drop(attn_scores)   # [B, num_heads, seq_len, seq_len]; dropout
        # self.attn_scores = attn_scores    # store the attention scores for visualization
        # attn_out = attn_scores @ x_v

        # Method 2: Memory-efficient attention (SDPA)
        attn_p = self.attn_drop.p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            x_q.contiguous(), x_k.contiguous(), x_v.contiguous(),
            attn_mask=None,
            dropout_p=attn_p,
            is_causal=False
        )  # [B, H, S, D]

        # Get the new embeddings from the Values and Attention scores, and concatenate back the heads through reshaping
        x = attn_out.transpose(1, 2).reshape(B, seq_len, embed_dim)  # [B, seq_len, embed_dim] <-- [B, seq_len, num_heads, head_embed_dim] <-- [B, num_heads, seq_len, head_embed_dim] 
        x = self.proj(x)    # [B, seq_len, embed_dim]; linearly project the new embeddings
        x = self.proj_drop(x)   # [B, seq_len, embed_dim]; dropout
        return x

class ViTARCMHSA(nn.Module):
    """ 
    Multi-Head Self-Attention block with ViTARC (Two/Four-slope Alibi) RPE (Relative Positional Encoding).

    NOTE: See ViTARC for RPE code. The code here was modified.
    https://github.com/khalil-research/ViTARC/blob/effa665802946de83807143797247e81e8e7c4b3/vitarc/models/model.py#L195

    """
    def __init__(self, model_config, image_size, embed_dim, num_heads, rpe_type, grid_2d_distance_matrix, qkv_bias=False, attn_drop=0., proj_drop=0., num_prepended_extra_tokens=0, num_appended_extra_tokens=0):
        super().__init__()

        self.model_config = model_config
        self.image_size = image_size

        self.num_prepended_extra_tokens = num_prepended_extra_tokens
        self.num_appended_extra_tokens = num_appended_extra_tokens

        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)  # W_qkv; *3 because we want embeddings for q, k, v from the input; this is equivalent to defining three separate linear layers (W_q, W_k, W_v) for q, k, and v
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim) # W_o
        self.proj_drop = nn.Dropout(proj_drop)

        ## ViTARC RPE (modified code from ViTARC)
        self.rpe_type = rpe_type
        self.rpe_abs = False    # for now fixed to False

        # Two slopes are sufficient here, since we manipulate the distance matrix with pre-added per-diag-direction ratios
        self.register_buffer("slopes_l", torch.Tensor(self.get_slopes(self.num_heads, start_exponent=1)) * -1)
        self.register_buffer("slopes_r", torch.Tensor(self.get_slopes(self.num_heads, start_exponent=0.5)) * -1)    # TODO: Why 0.5 and not 1 like the left slope?

        ## Relative positions
        # Calculate relative positions for the full sequence (2D grid possibly with extra tokens) based on the 2D distance matrix pre-computed 
        grid_height = self.image_size   # VITARC wrote: 33
        grid_width = self.image_size    # VITARC wrote: 34
        
        # TODO: Why + 10 ? Just an arbitrary large distance? Why not much more? when we observe the matrix during a run it also contains distance values such as 61 (which is greater than 44) ?
        large_dist = self.image_size + 100 # VITARC wrote +10 and get 44

        ## Calculate matrix of x and y differences
        total_length = grid_height * grid_width + num_prepended_extra_tokens + num_appended_extra_tokens
        
        distance_matrix = torch.full((total_length, total_length), fill_value=large_dist, dtype=torch.float)  # 100 as a large distance

        # Assign the 2D relative positions to the correct part of the matrix
        distance_matrix[num_prepended_extra_tokens:total_length-num_appended_extra_tokens, num_prepended_extra_tokens:total_length-num_appended_extra_tokens] = grid_2d_distance_matrix

        # Handle extra tokens relative positions. Extra tokens are considered to be far from everything
        if num_prepended_extra_tokens > 0:
            distance_matrix[:num_prepended_extra_tokens, :] = large_dist
            distance_matrix[:, :num_prepended_extra_tokens] = large_dist
        
        if num_appended_extra_tokens > 0:
            distance_matrix[total_length-num_appended_extra_tokens:, :] = large_dist
            distance_matrix[:, total_length-num_appended_extra_tokens:] = large_dist

        self.register_buffer("distance_matrix", distance_matrix)   # [total_length, total_length]; distance matrix for the 2D grid with extra tokens

        
    def get_slopes(self, n, start_exponent=1):
        """ ViTARC. Create the slopes for the relative position bias. """
        def get_geometric_slopes(n, start_exponent):
            start = 2 ** (-start_exponent)  # starting value 2^(-start_exponent)
            ratio = 2 ** -1     # halving each step
            return [start * (ratio ** i) for i in range(n)]

        if math.log2(n).is_integer():
            return get_geometric_slopes(n, start_exponent)
        
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (get_geometric_slopes(closest_power_of_2, start_exponent) +
                    self.get_slopes(2 * closest_power_of_2, start_exponent)[0::2][:n - closest_power_of_2])    


    def compute_bias(self, query_length, key_length, relative_position, device):
        """ ViTARC. Compute binned relative position bias. """

        if self.rpe_abs:
            relative_position = torch.abs(relative_position).unsqueeze(0).expand(self.num_heads, -1, -1)
        else:
            relative_position = relative_position.unsqueeze(0).expand(self.num_heads, -1, -1)

        self.slopes_l = self.slopes_l.to(device)
        self.slopes_r = self.slopes_r.to(device)

        # relative_position is pre-mult with factor 2**0.25 for top-right, down-right
        alibi_left = self.slopes_l.unsqueeze(1).unsqueeze(1) * relative_position
        alibi_right = self.slopes_r.unsqueeze(1).unsqueeze(1) * relative_position

        values = torch.triu(alibi_right) + torch.tril(alibi_left)

        # Slice the relevant part of the bias before reshaping
        values = values[:, :query_length, :key_length]  # [num_heads, query_length, key_length]
        values = values.view(1, self.num_heads, query_length, key_length)  # [1, num_heads, query_length, key_length]          

        return values            

    def forward(self, x):
        
        ## As usual
        B, seq_len, embed_dim = x.shape
        device = x.device
        
        # Compute the Queries, Keys, Values from the input embeddings by a linear projection
        x_qkv = self.qkv_proj(x) # [B, seq_len, 3*embed_dim]

        # Reshape the Queries, Keys, Values for multi-head
        head_embed_dim = embed_dim // self.num_heads
        x_qkv = x_qkv.reshape(B, seq_len, 3, self.num_heads, head_embed_dim).permute(2, 0, 3, 1, 4)  # [3, B, num_heads, seq_len, head_embed_dim]

        # Get the Queries, Keys, Values for all heads
        x_q, x_k, x_v = x_qkv[0], x_qkv[1], x_qkv[2]    # ([B, num_heads, seq_len, head_embed_dim], [B, num_heads, seq_len, head_embed_dim], [B, num_heads, seq_len, head_embed_dim])

        # Compute the attention scores
        attn = (x_q @ x_k.transpose(-2, -1))    # [B, num_heads, seq_len, seq_len]; logits
        attn_scaled = attn * self.scale   # [B, num_heads, seq_len, seq_len]; scaled logits

        ## ViTARC. Compute relative bias and add it to the attention logits for RPE
        # NOTE: We compute the RPE bias over all the sequence x, which possibly includes extra tokens as they are handled in the distance matrix by setting a large distance
        position_bias = self.compute_bias(seq_len, seq_len, relative_position=self.distance_matrix, device=device)  # [B, num_heads, seq_len, seq_len]
        
        # NOTE: We add the position bias to the unscaled logits as per ViTARC Alibi
        attn += position_bias

        logits = attn   # [B, num_heads, seq_len, seq_len]

        ## As usual
        attn_scores = self.softmax(logits)   # [B, num_heads, seq_len, seq_len]; attention weights
        attn_scores = self.attn_drop(attn_scores)   # [B, num_heads, seq_len, seq_len]; dropout
        self.attn_scores = attn_scores    # store the attention scores for visualization
        
        # Get the new embeddings from the Values and Attention scores, and concatenate back the heads through reshaping
        x = (attn_scores @ x_v).transpose(1, 2).reshape(B, seq_len, embed_dim)  # [B, seq_len, embed_dim] <-- [B, seq_len, num_heads, head_embed_dim] <-- [B, num_heads, seq_len, head_embed_dim] 
        x = self.proj(x)    # [B, seq_len, embed_dim]; linearly project the new embeddings
        x = self.proj_drop(x)   # [B, seq_len, embed_dim]; dropout

        ## ViTARC.
        # TODO: They return the position bias as well, but I don't (?)

        return x

class RoPEMHSA(nn.Module):
    """ 
    Multi-Head Self-Attention block including RoPE (RPE).
    """
    def __init__(self, model_config, image_size, embed_dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0., num_prepended_extra_tokens=0):
        super().__init__()

        self.model_config = model_config
        self.image_size = image_size
        self.num_prepended_extra_tokens = num_prepended_extra_tokens

        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)  # *3 because we want embeddings for q, k, v
        self.head_embed_dim = embed_dim // self.num_heads
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        ## 2D RoPE (Rotary Positional Embedding)
        # Choose RoPE dim (half features per axis)
        rope_dim = self.head_embed_dim // 4
        rope_dim = rope_dim if rope_dim % 2 == 0 else max(2, rope_dim - (rope_dim % 2)) # ensure rope_dim is even

        self.rotary_emb = RotaryEmbedding(
            dim=rope_dim,   # half the dims rotated per axis; total rotated features is dim*4
            freqs_for='pixel',  # use 'pixel' for 2D images
            max_freq=image_size // 2,   # adjust max_freq based on the grid image size
            )

        # Get and store flattened axial frequencies
        axial_freqs = self.rotary_emb.get_axial_freqs(image_size, image_size) # [H=image_size, W=image_size, rope_dim * 2]
        flat_axial_freqs = rearrange(axial_freqs, 'h w d -> (h w) d') # [num_tokens, rope_dim * 2], where num_tokens is the sequence length that the axial frequencies are computed for
        self.register_buffer('rope_axial_freqs', flat_axial_freqs, persistent=False)

    def forward(self, x, num_appended_extra_tokens):
        B, seq_len, embed_dim = x.shape
        
        # Compute the Queries, Keys, Values from the input embeddings by a linear projection
        x_qkv = self.qkv_proj(x) # [B, seq_len, 3*embed_dim]

        # Reshape the Queries, Keys, Values for multi-head
        x_qkv = x_qkv.reshape(B, seq_len, 3, self.num_heads, self.head_embed_dim).permute(2, 0, 3, 1, 4)  # [3, B, num_heads, seq_len, head_embed_dim]

        # Get the Queries, Keys, Values
        x_q, x_k, x_v = x_qkv[0], x_qkv[1], x_qkv[2]    # ([B, num_heads, seq_len, head_embed_dim], [B, num_heads, seq_len, head_embed_dim], [B, num_heads, seq_len, head_embed_dim])

        ## Use RoPE (2D with Axial Rotary Embedding). We rotate queries and keys
        
        # Remove the extra tokens temporarily as we do not want to rotate for them
        # First store q,k extra tokens before removing them
        prepended_extra_q = None
        prepended_extra_k = None
        appended_extra_q = None
        appended_extra_k = None

        if self.num_prepended_extra_tokens > 0:
            prepended_extra_q = x_q[:, :, :self.num_prepended_extra_tokens, :]
            prepended_extra_k = x_k[:, :, :self.num_prepended_extra_tokens, :]
            x_q = x_q[:, :, self.num_prepended_extra_tokens:, :]
            x_k = x_k[:, :, self.num_prepended_extra_tokens:, :]

        if num_appended_extra_tokens > 0:
            appended_extra_q = x_q[:, :, -num_appended_extra_tokens:, :]
            appended_extra_k = x_k[:, :, -num_appended_extra_tokens:, :]
            x_q = x_q[:, :, :-num_appended_extra_tokens, :]
            x_k = x_k[:, :, :-num_appended_extra_tokens, :]

        # Rotate the queries and keys
        # Reshape q, k (to have seq_len and head_dim as the the two last dimensions) for apply_rotary_emb: [B*num_heads, seq_len, head_dim]
        q_reshaped = rearrange(x_q, 'b h s d -> (b h) s d') # [B*num_heads, seq_len, head_dim]
        k_reshaped = rearrange(x_k, 'b h s d -> (b h) s d') # [B*num_heads, seq_len, head_dim]

        # Apply 2D RoPE using the precomputed axial freqs
        q_rotated = apply_rotary_emb(self.rope_axial_freqs, q_reshaped, seq_dim=1)  # [B*num_heads, seq_len, head_dim]; seq_dim=1 aligns freqs axis 0 (seq_len) with input axis 1 (seq_len)
        k_rotated = apply_rotary_emb(self.rope_axial_freqs, k_reshaped, seq_dim=1)  # [B*num_heads, seq_len, head_dim]; seq_dim=1 aligns freqs axis 0 (seq_len) with input axis 1 (seq_len)

        # Reshape back after rotation
        x_q = rearrange(q_rotated, '(b h) s d -> b h s d', h=self.num_heads)    # [B, num_heads, seq_len, head_embed_dim]
        x_k = rearrange(k_rotated, '(b h) s d -> b h s d', h=self.num_heads)    # [B, num_heads, seq_len, head_embed_dim]

        # Concatenate back the extra tokens removed before RoPE
        if self.num_prepended_extra_tokens > 0:
            # Concatenate back the extra tokens to the embeddings after applying RoPE
            x_q = torch.cat((prepended_extra_q, x_q), dim=2)
            x_k = torch.cat((prepended_extra_k, x_k), dim=2)

        if num_appended_extra_tokens > 0:
            # Concatenate back the extra tokens to the embeddings after applying RoPE
            x_q = torch.cat((x_q, appended_extra_q), dim=2)
            x_k = torch.cat((x_k, appended_extra_k), dim=2)

        # Method 1: Raw compute of the attention scores
        # attn = (x_q @ x_k.transpose(-2, -1))    # [B, num_heads, seq_len, seq_len]; unscaled attention logits
        # attn_scaled = attn * self.scale   # [B, num_heads, seq_len, seq_len]; scaled attention logits
        # attn_scores = self.softmax(attn_scaled)   # [B, num_heads, seq_len, seq_len]; attention scores/weights (normalized scaled attention logits)
        # # NOTE: no masking
        # attn_scores = self.attn_drop(attn_scores)   # [B, num_heads, seq_len, seq_len]; dropout
        # self.attn_scores = attn_scores    # store the attention scores/weights for visualization
        # attn_out = attn_scores @ x_v

        # Method 2: Memory-efficient attention (SDPA)
        attn_p = self.attn_drop.p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            x_q.contiguous(), x_k.contiguous(), x_v.contiguous(),
            attn_mask=None,
            dropout_p=attn_p,
            is_causal=False
        )  # [B, H, S, D]

        # Get the new embeddings from the Values and Attention scores, and concatenate back the heads through reshaping
        # NOTE: Attention used for the full sequence, so no slicing required on x_v
        x = attn_out.transpose(1, 2).reshape(B, seq_len, embed_dim)  # [B, seq_len, embed_dim] <-- [B, seq_len, num_heads, head_embed_dim] <-- [B, num_heads, seq_len, head_embed_dim] 
        x = self.proj(x)    # [B, seq_len, embed_dim]; linearly project the new embeddings
        x = self.proj_drop(x)   # [B, seq_len, embed_dim]; dropout
        
        return x

class FeedForward(nn.Module):
    """ Feed-Forward Neural Network block """
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, activation=nn.GELU(), ff_drop=0.0):
        super().__init__()
        output_dim = output_dim or input_dim
        hidden_dim = hidden_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = activation
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(ff_drop)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerLayer(nn.Module):
    """ Transformer layer """
    def __init__(self, model_config, image_size, embed_dim, num_heads, mlp_ratio=4., qkv_bias=False, attn_drop=0., attn_proj_drop=0., ff_drop=0., num_prepended_extra_tokens=0, num_appended_extra_tokens=0, rpe_type=None, grid_2d_distance_matrix=None):
        super().__init__()

        self.first_norm = nn.LayerNorm(embed_dim)

        self.rpe_type = rpe_type

        if rpe_type == 'rope':
            self.attn = RoPEMHSA(model_config,
                                 image_size,
                                 embed_dim,
                                 num_heads=num_heads,
                                 qkv_bias=qkv_bias,
                                 attn_drop=attn_drop,
                                 proj_drop=attn_proj_drop,
                                 num_prepended_extra_tokens=num_prepended_extra_tokens
                                 )

        elif rpe_type in ["Four-diag-slope-Alibi", "Two-slope-Alibi"]:
            self.attn = ViTARCMHSA(model_config,
                                   image_size,
                                   embed_dim,
                                   num_heads=num_heads,
                                   rpe_type=rpe_type,
                                   grid_2d_distance_matrix=grid_2d_distance_matrix,
                                   qkv_bias=qkv_bias,
                                   attn_drop=attn_drop,
                                   proj_drop=attn_proj_drop,
                                   num_prepended_extra_tokens=num_prepended_extra_tokens,
                                   num_appended_extra_tokens=num_appended_extra_tokens
                                   )

        else:
            self.attn = MHSA(model_config,
                             image_size,
                             embed_dim,
                             num_heads=num_heads,
                             qkv_bias=qkv_bias,
                             attn_drop=attn_drop,
                             proj_drop=attn_proj_drop,
                             num_prepended_extra_tokens=num_prepended_extra_tokens
                             )
        
        self.second_norm = nn.LayerNorm(embed_dim)

        self.ff = FeedForward(input_dim=embed_dim,
                              hidden_dim=int(embed_dim * mlp_ratio),
                              ff_drop=ff_drop
                              )

    def forward(self, x, num_appended_extra_tokens=0):
        if self.rpe_type == 'rope':
            x = x + self.attn(self.first_norm(x), num_appended_extra_tokens)   # [B, seq_len, embed_dim]; this uses norm first and then attention
        else:
            x = x + self.attn(self.first_norm(x))   # [B, seq_len, embed_dim]; this uses norm first and then attention

        x = x + self.ff(self.second_norm(x))    # [B, seq_len, embed_dim]
        # return x, self.attn.attn_scores
        return x

def calculate_2d_relative_positions(grid_height, grid_width, rpe_type):
    """ ViTARC. Calculate the 2D relative positions of all the pixels in the grid. """
    if rpe_type == "Four-diag-slope-Alibi":
        # Define direction-specific factors
        # Pre-mult those to diagonal directions
        top_right_factor = 2 ** 0.25
        down_right_factor = 2 ** 0.25
    else:
        top_right_factor = 1.0
        down_right_factor = 1.0
    
    # Create grid coordinates
    x_coords, y_coords = torch.meshgrid(
        torch.arange(grid_height, dtype=torch.long),
        torch.arange(grid_width, dtype=torch.long),
        indexing='ij'
    )

    # Flatten the 2D grid coordinates
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()

    # Initialize the relative position matrix
    num_positions = grid_height * grid_width
    relative_position = torch.zeros((num_positions, num_positions), dtype=torch.float)

    # Calculate Manhattan distance between each pair of points
    for i in range(num_positions):
        for j in range(num_positions):
            x_diff = x_flat[i] - x_flat[j]
            y_diff = y_flat[i] - y_flat[j]
            manhattan_distance = float(abs(x_diff) + abs(y_diff))  # Convert to float

            # Adjust the distance based on the direction
            if x_diff < 0 and y_diff < 0:  # Top-right
                manhattan_distance *= top_right_factor
            elif x_diff > 0 and y_diff < 0:  # Down-right
                manhattan_distance *= down_right_factor

            relative_position[i, j] = manhattan_distance

    return relative_position

class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self,
                 base_config,
                 model_config,
                 network_config,
                 image_size,
                 num_channels,
                 num_classes,
                 ):

        super().__init__()

        self.model_config = model_config
        self.network_config = network_config

        self.image_size = image_size
        self.seq_len = self.image_size * self.image_size    # NOTE: here image_size also takes into account border padding if relevant
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_token_categories = self.num_classes
        
        self.patch_size = model_config.patch_size

        self.num_all_tokens = self.seq_len  # essentially seq_len + extra tokens; number of tokens/positions in a sequence (i.e., the extra tokens are also accounted for)

        self.embed_dim = network_config.embed_dim
        assert self.embed_dim % network_config.num_heads == 0, 'Embedding dimension must be divisible by the number of heads'

        if base_config.data_env == 'BEFOREARC' and model_config.task_embedding.enabled and model_config.task_embedding.approach == 'task_tokens':
            self.num_token_categories = self.num_token_categories + model_config.num_elementary_tasks

        ### [Optional] Extra tokens
        # NOTE: They are concatenated to the input embeddings during the forward pass of the model
        self.num_prepended_extra_tokens = 0 # coming from the cls token or register tokens
        self.num_appended_extra_tokens = 0  # coming from the task embedding

        ## Prepended extra tokens
        # Create cls token
        if model_config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) # create the [cls] token
            self.num_prepended_extra_tokens += 1
            self.num_all_tokens += 1
            if base_config.data_env in ['REARC', 'BEFOREARC']:
                self.num_token_categories += 1  # + 1 cls token

        # Create register tokens
        if model_config.num_reg_tokens > 0:
            self.reg_tokens = nn.Parameter(torch.zeros(1, model_config.num_reg_tokens, self.embed_dim)) # create the register tokens
            self.num_prepended_extra_tokens += model_config.num_reg_tokens
            self.num_all_tokens += model_config.num_reg_tokens
            if base_config.data_env in ['REARC', 'BEFOREARC']:
                self.num_token_categories += 1  # + 1 register token

        ## Appended extra tokens
        # Task embedding: task tokens
        if model_config.task_embedding.enabled and model_config.task_embedding.approach == 'task_tokens':
            self.num_appended_extra_tokens += 4 # TODO: Fix to 4 for now but it should be obtained from the data module
            self.num_all_tokens += 4 # TODO: Fix to 4 for now but it should be obtained from the data module

        # Task embedding: example in-context
        if model_config.task_embedding.enabled and model_config.task_embedding.approach == 'example_in_context':
            self.num_appended_extra_tokens += (self.image_size * self.image_size) * 2   # we append one input-output example (padded to the max image size as the input-output)
            self.num_all_tokens += (self.image_size * self.image_size) * 2

        ## Patch Embeddings
        if base_config.data_env == 'CVR':
            # We are in CVR and we want to use the RGB channels for the input image, so we create patches using convolutions
            # Using channels: input x is [B, C=3, H, W] and we should get [B, seq_len=num_patches, embed_dim]
            self.patch_embed = PatchEmbed(base_config, img_size=self.image_size, patch_size=self.patch_size, in_channels=self.num_channels, embed_dim=self.embed_dim)
        
        elif base_config.data_env in ['REARC', 'BEFOREARC']:
            # NOTE: When patch_size=1 (i.e., consider pixels), this is equivalent to flattening the input image and projecting it to the embedding dimension
            if model_config.use_ohe_repr:
                # We are in REARC/BEFOREARC and we want to use OHE (resulting in channels) for the possible tokens and thus create artificial channels for the linear projection of patches/pixels/tokens
                # Using Channels: we create input x [B, C=num_token_categories, H, W] and we should get [B, seq_len=num_patches, embed_dim]
                self.patch_embed = PatchEmbed(base_config, img_size=self.image_size, patch_size=self.patch_size, in_channels=self.num_channels, embed_dim=self.embed_dim, num_token_categories=self.num_token_categories)
            else:
                # We are in REARC/BEFOREARC and we want to simply flatten the input image (resulting in a sequence of tokens) and thus create an artificial channel for the linear projection of patches/pixels/tokens
                # Using Seq2Seq: we create input x [B, C=1, H, W] and we should get [B, seq_len, embed_dim]
                self.patch_embed = PatchEmbed(base_config, img_size=self.image_size, patch_size=self.patch_size, in_channels=self.num_channels, embed_dim=self.embed_dim)

        log_message = f"The max sequence length (with padding) not embedded as patches is: {image_size * image_size}\n"
        log_message += f"The actual max sequence length (with padding) embedded as patches is: {self.patch_embed.num_patches}\n"
        log_message += "The two numbers should be equal if the image size is a square and the patch size is 1."
        logger.info(log_message)
        
        self.seq_len = self.patch_embed.num_patches


        ## [Optional] Absolute Positional Embedding (APE) with Mixer
        # APE is applied right after the Patch Embedding
        if self.model_config.ape.enabled:
            self.ape = AbsolutePositionalEncoding(model_config=model_config,
                                                  network_config=network_config,
                                                  image_size=self.image_size,
                                                  seq_len=self.seq_len,
                                                  patch_embed=self.patch_embed,
                                                  embed_dim=self.embed_dim,
                                                  num_prepended_extra_tokens=self.num_prepended_extra_tokens,
                                                  ape_type=model_config.ape.ape_type,
                                                  mixer_strategy=model_config.ape.mixer
                                                  )

        ## [Optional] Relative Positional Embeddings (RPE)
        # RPE is applied in the Attention layer
        if model_config.rpe.enabled:
            rpe_type = model_config.rpe.rpe_type

            if rpe_type in ["Four-diag-slope-Alibi", "Two-slope-Alibi"]:
                grid_2d_distance_matrix = calculate_2d_relative_positions(self.image_size, self.image_size, rpe_type)
            else:
                grid_2d_distance_matrix = None

        else:
            rpe_type = None
            grid_2d_distance_matrix = None

        ## Transformer Encoder Layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(
                model_config=model_config,
                image_size=self.image_size,
                embed_dim=self.embed_dim,
                num_heads=network_config.num_heads,
                mlp_ratio=network_config.mlp_ratio,
                qkv_bias=network_config.qkv_bias,
                attn_drop=network_config.attn_dropout_rate,
                attn_proj_drop=network_config.attn_proj_dropout_rate,
                ff_drop=network_config.ff_dropout_rate,
                num_prepended_extra_tokens=self.num_prepended_extra_tokens,
                num_appended_extra_tokens=self.num_appended_extra_tokens,
                rpe_type=rpe_type,
                grid_2d_distance_matrix=grid_2d_distance_matrix
            )
            for _ in range(self.network_config.num_layers)
            ])

        self.last_layer_norm = nn.LayerNorm(self.embed_dim)


        ## Weights initialization of all the learnable components
        self.init_type = self.network_config.init_weights_cfg.type
        self.distribution = self.network_config.init_weights_cfg.distribution
        self.init_weights()


    def _init_weights(self, m):
        # Linear modules
        if isinstance(m, nn.Linear):
            if m.bias is not None:
                nn.init.zeros_(m.bias)

            if self.init_type == 'trunc_normal':
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            
            elif self.init_type == 'normal':
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            
            elif self.init_type == 'xavier':
                if self.distribution == 'uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif self.distribution == 'normal':
                    nn.init.xavier_normal_(m.weight)

            elif self.init_type == 'kaiming':
                    if self.distribution == 'uniform':
                        nn.init.kaiming_uniform_(m.weight)
                    elif self.distribution == 'normal':
                        nn.init.kaiming_normal_(m.weight)
            
        # LayerNorm modules
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def init_weights(self):

        if self.init_type not in ['trunc_normal', 'normal', 'xavier', 'kaiming']:
            raise ValueError(f"Invalid initialization type: {self.init_type}. Choose from ['trunc_normal', 'normal', 'xavier', 'kaiming']")

        if self.init_type in ['xavier', 'kaiming'] and self.distribution not in ['uniform', 'normal']:
            raise ValueError(f"Invalid initialization distribution: {self.distribution}. Choose from ['uniform', 'normal']")

        # Transformer Encoder modules
        self.apply(self._init_weights)

        # TODO:
        # I removed the initialization of the APE parameters here.
        # See if OK and need to do it in the class AbsolutePositionalEncoding directly

        # Extra tokens
        if hasattr(self, 'reg_tokens'):
            nn.init.trunc_normal_(self.reg_tokens, std=0.02)
        if hasattr(self, 'cls_token'):
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward_embed(self, x, task_tokens=None, example_in_context=None, x_grid_object_ids=None):
        """ Embed the original input x and concatenate extra tokens to the created sequence if applicable """

        x_shape = x.shape   # [B, H, W]
        B = x_shape[0]

        ## Embed image input to a sequence of patches
        x = self.patch_embed(x) # [B, num_patches, embed_dim]; embed the input

        ## Handle extra prepended tokens
        # Concatenate the [cls] token and register tokens (if any) to the input embeddings
        # TODO: Depending on if we want to use PE for the extra tokens, we should add first the PE and then concatenate or concatenate first and then add the PE
        if hasattr(self, 'reg_tokens'):
            reg_tokens = self.reg_tokens.expand(B, -1, -1)   # [B, num_register_tokens, embed_dim]; use expand to create and add the tokens to each sample in the batch instead of just one sample
            x = torch.cat((reg_tokens, x), dim=1)   # [B, num_all_tokens, embed_dim]
        
        if hasattr(self, 'cls_token'):
            cls_tokens = self.cls_token.expand(B, -1, -1)   # [B, 1, embed_dim]; use expand to create and add the token for each sample in the batch instead of just one sample
            x = torch.cat((cls_tokens, x), dim=1)   # [B, num_all_tokens, embed_dim]

        ## Absolute Positional Encoding
        if self.model_config.ape.enabled:
            # Apply APE
            x = self.ape(x, x_grid_object_ids)  # [B, num_all_tokens, embed_dim], where num_all_tokens depends on the number of patches and extra tokens (if any) (e.g., cls, register tokens)

        ## Handle extra appended tokens
        # NOTE: We do not use an APE for the extra appended tokens (e.g., task tokens or input-output example in context)

        # Handle the task embedding for the example_in_context approach
        if self.model_config.task_embedding.enabled and example_in_context is not None:
            example_input = self.patch_embed(example_in_context[0])     # [B, num_patches, embed_dim]
            example_output = self.patch_embed(example_in_context[1])    # [B, num_patches, embed_dim]

            # Append the example input and output to the input sequence
            x = torch.cat((x, example_input, example_output), dim=1)    # [B, num_all_tokens + 2*num_patches, embed_dim]

        # Handle the task embedding for the task_tokens approach
        if self.model_config.task_embedding.enabled and task_tokens is not None:
            x = torch.cat((x, task_tokens), dim=1)  # [B, num_all_tokens + num_task_tokens, embed_dim]

        return x

    def forward_encode(self, x, num_appended_extra_tokens):
        """ Encode the input sequence x """
        
        x_shape = x.shape    # [B, num_all_tokens, embed_dim]

        if self.model_config.attention_map.enabled:
            self.attn_scores = []

        # Transformer layers/blocks
        for i, layer in enumerate(self.transformer_layers):
            # x, attn_scores = layer(x, num_appended_extra_tokens)  # [B, num_all_tokens, embed_dim], [B, num_heads, seq_len, seq_len]
            x = layer(x, num_appended_extra_tokens)

            # if self.model_config.attention_map.enabled:
            #     # Store the attention scores for visualization
            #     self.attn_scores.append(attn_scores)  # list of elements of dimensions [B, num_heads, seq_len, seq_len]
        
        # Last layer norm
        x = self.last_layer_norm(x) # [B, num_all_tokens, embed_dim]
        
        return x
    
    def forward_features(self, x):
        """ Output relevant feature embeddings of the encoded sequence x """
        
        x_shape = x.shape    # [B, num_all_tokens, embed_dim]
        
        if self.model_config.use_cls_token and self.model_config.encoder_aggregation.enabled:
            if self.model_config.encoder_aggregation.method == "token":
                # Aggregate features to the cls token
                x = x[:, 0, :]  # [B, embed_dim]
            elif self.model_config.encoder_aggregation.method == "mean":
                # Average pooling (i.e., average the features)
                x = x.mean(dim=1)   # [B, embed_dim]
            elif self.model_config.encoder_aggregation.method == "max":
                # Max pooling
                x = x.max(dim=1).values    # [B, embed_dim]
        else:
            # Discard the embeddings of the prepended extra tokens (e.g., cls, register tokens)
            x = x[:, self.num_prepended_extra_tokens:, :]    # [B, seq_len, embed_dim] <-- [B, num_all_tokens - num_prepended_extra_tokens, embed_dim]

            # Discard the embeddings of the appended extra tokens (e.g., task tokens or input-output example in context)
            if self.num_appended_extra_tokens > 0:
                x = x[:, :-self.num_appended_extra_tokens, :]

        return x

    def forward(self, src, task_tokens=None, example_in_context=None, x_grid_object_ids=None):
        """ 
        Forward pass entry point of the network 
        The input is an image of shape [B, C, H, W] and the output is a sequence of embeddings.
        
        NOTE: Sequence of single element (i.e., not actually a sequence) for CVR as we predict a single label.
        """

        # NOTE: Masks for the forward() method
        # 1) [Not needed] (Self-)Attention mask. Use mask in forward() of nn.TransformerEncoder. It is used for custom attention masking. Use value 0 for allowed and -inf for masked positions.
        # 2) [Not needed] Padding mask. Use src_key_padding_mask in forward() of nn.TransformerEncoder. It is used to mask the padding (or other) tokens in the input sequence. Use value True for padding tokens and False for actual tokens.

        src_shape = src.shape

        # Embed the input image to an input sequence
        x = self.forward_embed(src, task_tokens, example_in_context, x_grid_object_ids)  # [B, num_all_tokens(+task_embedding_length), embed_dim] <-- [B, C, H, W]

        # Encode the sequence (i.e., the embedded input image)
        x = self.forward_encode(x, self.num_appended_extra_tokens)  # [B, num_all_tokens, embed_dim] <-- [B, num_all_tokens, embed_dim]

        # Output feature embeddings from the encoded input sequence
        x = self.forward_features(x)    # [B, embed_dim] <-- [B, num_all_tokens, embed_dim]

        return x

    def get_attention_scores(self):
        """ Get the attention scores from the config-selected ViT layer """
        
        if hasattr(self, 'attn_scores'):
            return self.attn_scores
        else:
            logger.warning("Attention scores not available. Ensure that the attention map is enabled in the model configuration.")
            return None

def get_vit(base_config, model_config, network_config, image_size, num_channels, num_classes):
    """ Return a ViT encoder instance """
    
    vit_encoder = VisionTransformer(base_config,
                                    model_config, 
                                    network_config, 
                                    image_size, 
                                    num_channels, 
                                    num_classes
                                    )
    
    return vit_encoder
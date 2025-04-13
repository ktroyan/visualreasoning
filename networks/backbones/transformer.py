"""
This Transformer encoder does not support RPE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Personal codebase dependencies
from utility.utils import plot_absolute_positional_embeddings
from utility.rearc.utils import one_hot_encode
from utility.logging import logger


__all__ = ['get_transformer_encoder']


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

        # TODO: See if it is correct to create patches (in the forward pass) without the extra tokens (so not considered when creating the artificial channels)

        self.base_config = base_config
        if self.base_config.data_env == 'REARC':
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
        self.patch_proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        if self.base_config.data_env == 'REARC':
            if self.num_token_categories is not None:
                x = one_hot_encode(x, self.num_token_categories)  # add a channel dimension for the input image: [B, C=num_token_categories, H, W] <-- [B, H, W]
            else:
                x = x.unsqueeze(1).float()  # add a channel dimension for the input image: [B, C=1, H, W] <-- [B, H, W]; x is int64, so we need to convert it to float32 for the Conv2d layer (otherwise issue with the conv bias)

        x = self.patch_proj(x)    # [B, embed_dim, H_out//patch_size, W_out//patch_size], where H_out = ((H-1)/1)+1) = H and W_out = ((W-1)/1)+1) = W if the spatial dimensions have not been reduced, by having patch_size=1 and stride=patch_size
        x = x.flatten(2) # [B, embed_dim, num_patches]  # flatten the spatial dimensions to get a sequence of patches
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim], where num_patches = H*W = seq_len
        return x

class AbsolutePositionalEncoding(nn.Module):
    """
    NOTE: Code adapted from the ViTARC repo. See their original class ViTARCEmbedding.

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
        - 'default' (simple addition of the APE to the input embeddings)
    """
    def __init__(self, model_config, network_config, image_size, seq_len, patch_embed, embed_dim, num_extra_tokens, ape_type, mixer_strategy='default'):
        super().__init__()

        self.model_config = model_config
        self.network_config = network_config
        self.image_size = image_size
        self.seq_len = seq_len
        self.patch_embed = patch_embed
        self.num_extra_tokens = num_extra_tokens
        self.embed_dim = embed_dim

        self.mixer_strategy = mixer_strategy

        ## APE (static creation, as there is no OPE)
        if not model_config.ope.enabled:
            # We can already create the APE as it does not have to changed depending on the input sample
            self.static_ape = self.create_ape(image_size=image_size,
                                              seq_len=seq_len,
                                              patch_embed=patch_embed,
                                              embed_dim=embed_dim,
                                              num_extra_tokens=num_extra_tokens,
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


    def create_ape(self, image_size, seq_len, patch_embed, embed_dim, num_extra_tokens, ape_type) -> nn.Parameter:
        """ 
        Create and return the desired APE (absolute positional embedding).
        The APE will be added to the input embeddings of the backbone/encoder mode (e.g., ViT) during the forward pass.
        
        NOTE: The PE is considered for the actual tokens and other special types of tokens to predict (e.g., pad tokens)
            However, the extra tokens (e.g., [cls] and register tokens) are not considered for the PE.
            The PE should still be of the correct size since it is added to the whole input embeddings which include the extra tokens.

        TODO:
        Check if 2D sin-cos APE is correctly implemented.
        """
        
        num_pos_embeds = seq_len + num_extra_tokens

        if ape_type == 'learn':    # learned APE

            pos_embed_learned = nn.Parameter(torch.randn(1, seq_len, embed_dim))  # [1, seq_len, embed_dim]; by default requires_grad=True

            if num_extra_tokens > 0:
                # Set the PE for the extra tokens to zero (i.e., no PE for the extra tokens since PE is added to the input embeddings and the PE is not learned)
                pos_embed_extra_tokens = torch.zeros(1, num_extra_tokens, embed_dim, dtype=torch.float32, requires_grad=False)     # define the PE for the [cls] and register tokens to be concatenated at the beginning of the PE for the patches
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
            if num_extra_tokens > 0:
                extra_pe = torch.zeros([1, num_extra_tokens, embed_dim], dtype=torch.float32)
                pos_embed = nn.Parameter(torch.cat([extra_pe, pos_embed], dim=1))  # [1, seq_len+num_extra_tokens, embed_dim]; concatenate along sequence dimension
            else:
                pos_embed = nn.Parameter(pos_embed) # [1, seq_len, embed_dim]

            # Ensure PE is fixed/non-learnable
            pos_embed.requires_grad = False
            
        else:
            raise ValueError(f"Invalid positional encoding type: {ape_type}. Choose from ['learn', '2dsincos']")

        # Visualize PE
        plot_absolute_positional_embeddings(pos_embed, num_prefix_tokens=num_extra_tokens, viz_as_heatmap=False)

        return pos_embed    # [1, seq_len(+num_extra_tokens), embed_dim]

    def create_ape_with_ope(self, image_size, seq_len, patch_embed, embed_dim, num_extra_tokens, x_grid_object_ids) -> nn.Parameter:
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

        num_pos_embeds = seq_len + num_extra_tokens

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
        if num_extra_tokens > 0:
            extra_pe = torch.zeros((B, num_extra_tokens, embed_dim), dtype=torch.float32, device=pos_embed.device)
            pos_embed = torch.cat([extra_pe, pos_embed], dim=1)  # [B, num_extra_tokens + seq_len, embed_dim]

        # Make sure the APE (with OPE) is fixed/non-learnable
        pos_embed = nn.Parameter(pos_embed, requires_grad=False)  # [B, seq_len (+num_extra_tokens), embed_dim]; fixed/non-learnable

        # Visualize PE
        # NOTE: Should not uncomment this line if the goal is not to observe the PE, as this would plot for each batch during training
        # plot_absolute_positional_embeddings(pos_embed, num_prefix_tokens=num_extra_tokens)

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
                                                       num_extra_tokens=self.num_extra_tokens,
                                                       x_grid_object_ids=x_grid_object_ids
                                                       )    # [batch_size, seq_len, embed_dim]; created for each sample in the batch
        else:
            # Static creation
            position_embeds = self.static_ape   # [1, seq_len, embed_dim]; create for a single sample, so need to be expanded to the batch size
            position_embeds = position_embeds.expand(inputs_embeds.shape[0], -1, -1)  # [batch_size, seq_len, embed_dim]


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

        elif strategy == 'default':
            output_embeds = inputs_embeds + position_embeds

        else:
            raise ValueError(f"Unsupported mixer_strategy: {strategy}")
        
        # Dropout after positional encoding
        output_embeds = self.ape_drop(output_embeds)

        return output_embeds

class TransformerEncoder(nn.TransformerEncoder):
    def __init__(self,
                 base_config,
                 model_config, 
                 network_config, 
                 encoder_layer, 
                 image_size, 
                 num_channels, 
                 num_classes,
                 ):

        super().__init__(encoder_layer, network_config.num_layers)

        self.model_config = model_config
        self.network_config = network_config

        self.image_size = image_size
        self.seq_len = self.image_size * self.image_size    # NOTE: here image_size also takes into account border padding if relevant
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.patch_size = model_config.patch_size

        self.num_all_tokens = self.seq_len  # essentially seq_len + extra tokens; number of tokens/positions in a sequence (i.e., the extra tokens are also accounted for)

        self.embed_dim = network_config.embed_dim
        assert self.embed_dim % network_config.num_heads == 0, 'Embedding dimension must be divisible by the number of heads'

        if base_config.data_env == 'REARC':
            if model_config.visual_tokens.enabled:
                self.num_token_categories = self.num_classes + 1 + 4 # 10 symbols (0-9) + 1 pad token + 4 x,y,xy border tokens and newline tokens
            else:
                self.num_token_categories = self.num_classes + 1


        ## [Optional] Extra tokens
        # NOTE: They are concatenated (prepended) to the input embeddings during the forward pass of the model
        self.num_extra_tokens = 0

        # Create cls token
        if model_config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) # create the [cls] token
            self.num_all_tokens += 1
            self.num_extra_tokens += 1
            if base_config.data_env == 'REARC':
                self.num_token_categories += 1  # + 1 cls token

        # Create register tokens
        if model_config.num_reg_tokens > 0:
            self.reg_tokens = nn.Parameter(torch.zeros(1, model_config.num_reg_tokens, self.embed_dim)) # create the register tokens
            self.num_all_tokens += model_config.num_reg_tokens
            self.num_extra_tokens += model_config.num_reg_tokens
            if base_config.data_env == 'REARC':
                self.num_token_categories += 1  # + 1 register token


        ## Patch Embeddings
        if base_config.data_env == 'CVR':
            # We are in CVR and we want to use the RGB channels for the input image, so we create patches using convolutions
            # Using channels: input x is [B, C=3, H, W] and we should get [B, seq_len=num_patches, embed_dim]
            self.patch_embed = PatchEmbed(base_config, img_size=self.image_size, patch_size=self.patch_size, in_channels=self.num_channels, embed_dim=self.embed_dim)
        
        elif base_config.data_env == 'REARC':
            # NOTE: When patch_size=1 (i.e., consider pixels), this is equivalent to flattening the input image and projecting it to the embedding dimension
            if model_config.use_ohe_repr:
                # We are in REARC and we want to use OHE (resulting in channels) for the possible tokens and thus create artificial channels for the linear projection of patches/pixels/tokens
                # Using Channels: we create input x [B, C=num_token_categories, H, W] and we should get [B, seq_len=num_patches, embed_dim]
                self.patch_embed = PatchEmbed(base_config, img_size=self.image_size, patch_size=self.patch_size, in_channels=self.num_channels, embed_dim=self.embed_dim, num_token_categories=self.num_token_categories)
            else:
                # We are in REARC and we want to simply flatten the input image (resulting in a sequence of tokens) and thus create an artificial channel for the linear projection of patches/pixels/tokens
                # Using Seq2Seq: we create input x [B, C=1, H, W] and we should get [B, seq_len, embed_dim]
                self.patch_embed = PatchEmbed(base_config, img_size=self.image_size, patch_size=self.patch_size, in_channels=self.num_channels, embed_dim=self.embed_dim)

        log_message = f"The max sequence length (with padding) not embedded as patches is: {image_size * image_size}\n"
        log_message += f"The actual max sequence length (with padding) embedded as patches is: {self.patch_embed.num_patches}\n"
        log_message += "The two numbers should be equal if the image size is a square and the patch size is 1."
        logger.info(log_message)

        self.seq_len = self.patch_embed.num_patches


        ## [Optional] Absolute Positional Embedding (APE) Mixer
        # APE is applied right after the Patch Embedding
        if self.model_config.ape.enabled:
            self.ape = AbsolutePositionalEncoding(model_config=model_config,
                                                  network_config=network_config,
                                                  image_size=self.image_size,
                                                  seq_len=self.seq_len,
                                                  patch_embed=self.patch_embed,
                                                  embed_dim=self.embed_dim,
                                                  num_extra_tokens=self.num_extra_tokens,
                                                  ape_type=model_config.ape.ape_type,
                                                  mixer_strategy=model_config.ape.mixer
                                                  )


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

    def forward_embed(self, x, x_grid_object_ids):
        x_shape = x.shape
        B = x_shape[0]

        x = self.patch_embed(x) # [B, num_patches, embed_dim]; embed the input

        # Concatenate the [cls] token and register tokens (if any) to the input embeddings
        # TODO: Depending on if we want to use PE for the extra tokens, we should add first the PE and then concatenate or concatenate first and then add the PE
        if hasattr(self, 'reg_tokens'):
            reg_tokens = self.reg_tokens.expand(B, -1, -1)   # [B, num_register_tokens, embed_dim]; use expand to create and add the tokens to each sample in the batch instead of just one sample
            x = torch.cat((reg_tokens, x), dim=1) 
        if hasattr(self, 'cls_token'):
            cls_tokens = self.cls_token.expand(B, -1, -1)   # [B, 1, embed_dim]; use expand to create and add the token for each sample in the batch instead of just one sample
            x = torch.cat((cls_tokens, x), dim=1)   # [B, num_all_tokens, embed_dim]
        
        # Absolute Positional Encoding
        if self.model_config.ape.enabled:
            # Apply APE
            x = self.ape(x, x_grid_object_ids)  # [B, num_all_tokens, embed_dim], where num_all_tokens depends on the number of patches and extra tokens (if any) (e.g., cls ro register tokens)
        
        return x

    def forward_encode(self, x, mask, src_key_padding_mask):
        x = super().forward(src=x, mask=mask, src_key_padding_mask=src_key_padding_mask)
        return x
    
    def forward_features(self, x):
        # TODO: Here is it correct to get rid of the extra tokens (e.g., cls or register tokens) from the output? 

        x_shape = x.shape    # [B, num_all_tokens, embed_dim]
        
        # TODO: Simplify this? Only used for CVR.
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
            # Output the features for each patch/token after having removed the extra tokens (e.g., cls and register tokens) if any
            x = x[:, self.num_extra_tokens:, :]    # [B, seq_len, embed_dim] <-- [B, num_all_tokens - num_extra_tokens, embed_dim]

        return x


    def forward(self, src, x_grid_object_ids=None, mask=None, src_key_padding_mask=None):
        # NOTE: Masks for the forward() method
        # 1) [Not needed] (Self-)Attention mask. Use mask in forward() of nn.TransformerEncoder. It is used for custom attention masking. Use value 0 for allowed and -inf for masked positions.
        # 2) [Not needed] Padding mask. Use src_key_padding_mask in forward() of nn.TransformerEncoder. It is used to mask the padding (or other) tokens in the input sequence. Use value True for padding tokens and False for actual tokens.

        src_shape = src.shape

        # Embed the input image to an input sequence
        x = self.forward_embed(src, x_grid_object_ids)

        # Encode the sequence (i.e., the embedded input image)
        x = self.forward_encode(x, mask, src_key_padding_mask)  # [B, num_all_tokens, embed_dim] <-- [B, C, H, W]

        # TODO: See if and how the default forward handles the extra tokens + remove them from the output
        x = self.forward_features(x)    # [B, embed_dim] <-- [B, num_all_tokens, embed_dim]

        return x

def get_transformer_encoder(base_config, model_config, network_config, image_size, num_channels, num_classes):
    """Returns a Transformer encoder instance"""

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=network_config.embed_dim,   # the dimension of the embeddings for each input and output token
        nhead=network_config.num_heads,     # the dimension of the embeddings for each head is: d_model // num_heads
        dim_feedforward=network_config.embed_dim * network_config.mlp_ratio,  # ff_dim; the hidden dimension of the feedforward network model
        batch_first=True, 
        )

    encoder = TransformerEncoder(base_config, model_config, network_config, encoder_layer, image_size, num_channels, num_classes)
    return encoder

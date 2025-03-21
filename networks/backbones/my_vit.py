import torch
import torch.nn as nn

# Personal codebase dependencies
from utility.logging import logger
from utility.utils import plot_absolute_positional_embeddings
from utility.rearc.utils import one_hot_encode


__all__ = ['get_my_vit']


def create_relative_positional_encoding():
    # TODO: See how to build and use the 2D RPE (relative positional encoding) in the Attention module.
    #       See ViTARC
    raise NotImplementedError("Relative Positional Encoding (RPE) not yet implemented")

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

        # TODO: See if it is correct to create patches (in the forward pass) without the extra tokens but to consider them still for the number of channels

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
                x = x.unsqueeze(1)  # add a channel dimension for the input image: [B, C=1, H, W] <-- [B, H, W]

        x = self.patch_proj(x)    # [B, embed_dim, H_out//patch_size, W_out//patch_size], where H_out = ((H-1)/1)+1) = H and W_out = ((W-1)/1)+1) = W if the spatial dimensions have not been reduced, by having patch_size=1 and stride=patch_size
        x = x.flatten(2) # [B, embed_dim, num_patches]  # flatten the spatial dimensions to get a sequence of patches
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim], where num_patches = H*W = seq_len
        return x

def create_absolute_positional_encoding(ape_type: str, seq_len: int, embed_dim: int, patch_embed: PatchEmbed, num_extra_tokens: int) -> nn.Parameter:
    """ 
    Create and return the desired APE (absolute positional embedding).
    NOTE: The PE is considered for the actual tokens, the pad tokens, and also the extra tokens (e.g., [cls] and register tokens).
    TODO: See if we should consider the extra tokens for the PE or not. For example, CVR does consider the cls token for 2dsincos PE.
          However, I did not find information in the paper "ViTs Need Registers" about PE for register tokens.
    
    """
    
    num_pos_embeds = seq_len + num_extra_tokens

    if ape_type == 'learn':    # learned positional encoding
        pos_embed = nn.Parameter(torch.randn(1, num_pos_embeds, embed_dim))  # [1, seq_len, embed_dim]
        pos_embed.requires_grad = True    # NOTE: set to True for the PE to be learned

    elif ape_type == '2dsincos':    # 2D sin-cos fixed positional embedding
        h, w = (patch_embed.grid_size, patch_embed.grid_size)
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (10000**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_embed = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        # Handle extra tokens such as the [cls] and register tokens
        if num_extra_tokens > 0:
            pos_embed_extra_tokens = torch.zeros([1, num_extra_tokens, embed_dim], dtype=torch.float32)     # define the PE for the [cls] and register tokens to be concatenated at the beginning of the PE for the patches
            pos_embed = nn.Parameter(torch.cat([pos_embed_extra_tokens, pos_embed], dim=1))    # [1, (num_patches+num_extra_tokens), embed_dim]; the PE will be added to the input embeddings of the ViT in the forward pass of the ViT backbone (VisionTransformer)

        else:
            pos_embed = nn.Parameter(pos_embed)
        
        pos_embed.requires_grad = False    # NOTE: set to False for the PE to not be learned; TODO: so the PE of the extra tokens is fixed to zero?

    else:
        raise ValueError(f"Invalid positional encoding type: {ape_type}. Choose from ['learn', '2dsincos']")

    # Visualize PE
    plot_absolute_positional_embeddings(pos_embed)

    return pos_embed    # [1, num_patches(+num_extra_tokens), embed_dim]


class MHSA(nn.Module):
    """ 
    Multi-Head Self-Attention block.

    TODO: implement RPE (Relative Positional Encoding) 

    """
    def __init__(self, embed_dim, num_heads=6, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)  # *3 because we want embeddings for q, k, v
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, D = x.shape   # B is the batch size, N is the number of tokens (were a patch is considered a token), D is the embedding dimension
        
        # Compute the Queries, Keys, Values from the input embeddings by a linear projection
        x_qkv = self.qkv_proj(x) # [B, N, 3*embed_dim]

        # Reshape the Queries, Keysm Values for multi-head
        head_embed_dim = D // self.num_heads
        x_qkv = x_qkv.reshape(B, N, 3, self.num_heads, head_embed_dim).permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_embed_dim]

        # Get the Queries, Keys, Values
        x_q, x_k, x_v = x_qkv[0], x_qkv[1], x_qkv[2]    # ([B, num_heads, N, head_embed_dim], [B, num_heads, N, head_embed_dim], [B, num_heads, N, head_embed_dim])

        attn = (x_q @ x_k.transpose(-2, -1))    # [B, num_heads, N, N]
        attn_scaled = attn * self.scale   # [B, num_heads, N, N]
        attn_scores = self.softmax(attn_scaled)   # [B, num_heads, N, N]
        attn_scores = self.attn_drop(attn_scores)   # [B, num_heads, N, N]; dropout

        # Get the new embeddings from the Values and Attention scores, and concatenate back the heads through reshaping
        x = (attn_scores @ x_v).transpose(1, 2).reshape(B, N, D)  # [B, N, D] <-- [B, N, num_heads, head_embed_dim] <-- [B, num_heads, N, head_embed_dim] 
        x = self.proj(x)    # [B, N, D]; linearly project the new embeddings
        x = self.proj_drop(x)   # [B, N, D]; dropout
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
    def __init__(self, embed_dim, num_heads=6, mlp_ratio=4., qkv_bias=False, attn_drop=0., attn_proj_drop=0., ff_drop=0.):
        super().__init__()
        self.first_norm = nn.LayerNorm(embed_dim)

        self.attn = MHSA(embed_dim,
                         num_heads=num_heads,
                         qkv_bias=qkv_bias,
                         attn_drop=attn_drop,
                         proj_drop=attn_proj_drop
                         )
        
        self.second_norm = nn.LayerNorm(embed_dim)

        self.ff = FeedForward(input_dim=embed_dim,
                              hidden_dim=int(embed_dim * mlp_ratio),
                              ff_drop=ff_drop
                              )

    def forward(self, x):
        x = x + self.attn(self.first_norm(x))   # [B, seq_len, embed_dim]; this uses norm first and then attention
        x = x + self.ff(self.second_norm(x))   # [B, seq_len, embed_dim]
        return x

class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(
            self,
            base_config,
            model_config,
            network_config,
            image_size,
            num_channels,
            num_classes,
            device
            ):
        
        super().__init__()

        self.model_config = model_config
        self.network_config = network_config

        self.image_size = image_size
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.patch_size = self.model_config.patch_size

        self.num_all_tokens = 0     # essentially seq_len + extra tokens; number of tokens/positions in a sequence (i.e., the extra tokens are also accounted for)

        if base_config.data_env == 'REARC':
            self.num_token_categories = self.num_classes  # REARC: 10 symbols (0-9) + 1 pad token (<pad_token>) [+ 1 cls token] + [1 register token]

        self.embed_dim = self.network_config.embed_dim
        assert self.embed_dim % self.network_config.num_heads == 0, 'Embedding dimension must be divisible by the number of heads'


        ## [Optional] Extra tokens
        # NOTE: They are concatenated (preprended) to the input embeddings during the forward pass of the model
        self.num_extra_tokens = 0

        # Create cls token
        if model_config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) # create the [cls] token
            self.num_all_tokens += 1
            self.num_extra_tokens += 1
            if base_config.data_env == 'REARC':
                self.num_token_categories += 1

        # Create register tokens
        if self.model_config.num_reg_tokens > 0:
            self.reg_tokens = nn.Parameter(torch.zeros(1, self.model_config.num_reg_tokens, self.embed_dim)) # create the register tokens
            self.num_all_tokens += model_config.num_reg_tokens
            self.num_extra_tokens += model_config.num_reg_tokens
            if base_config.data_env == 'REARC':
                self.num_token_categories += 1


        ## Patch Embeddings
        if base_config.data_env == 'CVR':
            # We are in CVR and we want to use the RGB channels for the input image, so we create patches using convolutions
            # Using channels: input x is [B, C=3, H, W] and we should get [B, seq_len=num_patches, embed_dim]
            self.patch_embed = PatchEmbed(base_config, img_size=self.image_size, patch_size=self.patch_size, in_channels=self.num_channels, embed_dim=self.embed_dim)
        
        elif base_config.data_env == 'REARC':
            # NOTE: When patch_size=1 (i.e., consider pixels), this is equivalent to flattening the input image and projecting it to the embedding dimension
            if self.model_config.use_ohe_repr:
                # We are in REARC and we want to use OHE (resulting in channels) for the possible tokens and thus create artificial channels for the linear projection of patches/pixels/tokens
                # Using Channels: we create input x [B, C=num_token_categories, H, W] and we should get [B, seq_len=num_patches, embed_dim]
                self.patch_embed = PatchEmbed(base_config, img_size=self.image_size, patch_size=self.patch_size, in_channels=self.num_channels, embed_dim=self.embed_dim, num_token_categories=self.num_token_categories)
            else:
                # We are in REARC and we want to simply flatten the input image (resulting in a sequence of tokens) and thus create an artificial channel for the linear projection of patches/pixels/tokens
                # Using Seq2Seq: we create input x [B, C=1, H, W] and we should get [B, seq_len, embed_dim]
                self.patch_embed = PatchEmbed(base_config, img_size=self.image_size, patch_size=self.patch_size, in_channels=self.num_channels, embed_dim=self.embed_dim)

        logger.info(f"The max sequence length (with padding) not embedded as patches is: {image_size * image_size}")
        logger.info(f"The actual max sequence length (with padding) embedded as patches is: {self.patch_embed.num_patches}")
        
        self.seq_len = self.patch_embed.num_patches


        ## Absolute Positional Embeddings
        # Create positional encoding
        self.ape = create_absolute_positional_encoding(ape_type=self.model_config.ape_type,
                                                       seq_len=self.seq_len,
                                                       embed_dim=self.embed_dim,
                                                       patch_embed=self.patch_embed,
                                                       num_extra_tokens=self.num_extra_tokens
                                                       )    # [1, num_all_tokens, embed_dim]
        
        self.ape_drop = nn.Dropout(p=self.network_config.ape_dropout_rate)    # dropout right after the positional encoding and residual


        ## Transformer Encoder Layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(
                embed_dim=self.embed_dim,
                num_heads=self.network_config.num_heads,
                mlp_ratio=self.network_config.mlp_ratio,
                qkv_bias=self.network_config.qkv_bias,
                attn_drop=self.network_config.attn_dropout_rate,
                attn_proj_drop=self.network_config.attn_proj_dropout_rate,
                ff_drop=self.network_config.ff_dropout_rate
            )
            for _ in range(self.network_config.num_layers)
            ])

        self.last_layer_norm = nn.LayerNorm(self.embed_dim)


        ## Weights initialization of all the learnable components
        self.init_weights()


    def _init_weights(self, m):
        # Weights init: Linear modules
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Weights init: LayerNorm modules
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def init_weights(self):
        # Weights init: Positional Embedding
        if self.ape.requires_grad:  # check if the PE is learnable
            nn.init.trunc_normal_(self.ape, std=0.02)

        # Weights init: Extra tokens
        if hasattr(self, 'reg_tokens'):
            nn.init.trunc_normal_(self.reg_tokens, std=0.02)
        if hasattr(self, 'cls_token'):
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Weights init: Transformer Encoder
        self.apply(self._init_weights)


    def forward_embed(self, x):
        B = x.shape[0]

        x = self.patch_embed(x) # [B, num_patches, embed_dim]

        # Concatenate the [cls] token and register tokens (if any) to the input embeddings
        # TODO: Depending on if we want to use PE for the extra tokens, we should add first the PE and then concatenate or concatenate first and then add the PE
        if hasattr(self, 'reg_tokens'):
            reg_tokens = self.reg_tokens.expand(B, -1, -1)   # [B, num_register_tokens, embed_dim]; use expand to create and add the tokens to each sample in the batch instead of just one sample
            x = torch.cat((reg_tokens, x), dim=1) 
        if hasattr(self, 'cls_token'):
            cls_tokens = self.cls_token.expand(B, -1, -1)   # [B, 1, embed_dim]; use expand to create and add the token for each sample in the batch instead of just one sample
            x = torch.cat((cls_tokens, x), dim=1)   # [B, num_all_tokens, embed_dim]
        
        # Add the positional encoding
        x = x + self.ape  # [B, num_all_tokens, embed_dim], where num_all_tokens depends on the number of patches and extra tokens (if any) (e.g., cls ro register tokens)
        x = self.ape_drop(x)    # [B, num_all_tokens, embed_dim]

        return x

    def forward_encode(self, x):

        B, S, D = x.shape    # [B, num_all_tokens, embed_dim]

        # Transformer layers/blocks
        for layer in self.transformer_layers:
            x = layer(x)  # [B, num_all_tokens, embed_dim]

        # Last layer norm
        x = self.last_layer_norm(x) # [B, num_all_tokens, embed_dim]
        
        return x
    
    def forward_features(self, x):
        # TODO: Here is it correct to get rid of the extra tokens (e.g., cls or register tokens) from the output? 

        B, S, D = x.shape    # [B, num_all_tokens, embed_dim]
        
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

    # NOTE: Masks for the forward() method
    # 1) [Not needed] (Self-)Attention mask. Use mask in forward() of nn.TransformerEncoder. It is used for custom attention masking. Use value 0 for allowed and -inf for masked positions.
    # 2) [Not needed] Padding mask. Use src_key_padding_mask in forward() of nn.TransformerEncoder. It is used to mask the padding (or other) tokens in the input sequence. Use value True for padding tokens and False for actual tokens.
    def forward(self, src):

        # Embed the input image to an input sequence
        x = self.forward_embed(src)  # [B, num_all_tokens, embed_dim] <-- [B, C, H, W]

        # Encode the sequence (i.e., the embedded input image)
        x = self.forward_encode(x)  # [B, num_all_tokens, embed_dim] <-- [B, C, H, W]

        # Output feature embeddings from the encoded input sequence
        x = self.forward_features(x)    # [B, embed_dim] <-- [B, num_all_tokens, embed_dim]

        return x

def get_my_vit(base_config, model_config, network_config, image_size, num_channels, num_classes, device):
    """Returns a ViT encoder instance"""
    
    vit_encoder = VisionTransformer(base_config,
                                    model_config, 
                                    network_config, 
                                    image_size, 
                                    num_channels, 
                                    num_classes, 
                                    device
                                    )
    
    return vit_encoder
    

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test the ViT encoder
    # Load the model and network configs
    ...


    # Instance 1: CVR approach. Good if we want to predict a single class from a sequence of tokens.
    # Test instance 1
    B = 2
    C = 3
    H = 128
    W = 128
    x = torch.randn(B, C, H, W)

    # The config parameters that matter for this instance are:
    # patch_size = 16
    # encoder_aggregation.enabled = True
    # encoder_aggregation.method = "token"
    model = VisionTransformer(model_config=None,
                              network_config=None,
                              img_size=H,
                              in_channels=C,
                              num_classes=4,
                              device=device
                              )

    # Get the predictions
    print("Results using the model as a classifier outputting a single embedding for the input image")
    y = model(x)  # [B, num_classes]; raw logits for each input sequence
    print("Output shape: ", y.shape)
    y = torch.argmax(y, dim=-1) # [B]; predicted classes
    print("Predictions (output after softmax) shape: ", y.shape)
    print("Predictions: ", y)


    # Instance 2: REARC approach. Good if we want to use a Transformer decoder after a ViT encoder
    # Test instance 2
    B = 2
    H = 30
    W = 30
    x = torch.randn(B, H, W)

    # The config parameters that matter for this instance are:
    # encoder_aggregation.enabled = False
    # patch_size = 1
    model = VisionTransformer(
        model_config=None,
        network_config=None,
        img_size=H,
        in_channels=1,
        num_classes=4,
        device=device
    )

    # Get the predictions
    print("Results using the model as an encoder outputting features for each patch/token in the sequence")
    y = model(x)    # [B, num_patches, embed_dim]; features embeddings for each token in the full input sequence encoded by the transformer
    print("Output shape: ", y.shape)
    print("Output (features embeddings resulting from encoding of each patch/token): ", y)    # features embeddings for each token in the full input sequence encoded by the transformer; we do not consider the extra tokens (e.g. cls or registers) in the output
    # Then we would need to use a Transformer decoder to get predictions
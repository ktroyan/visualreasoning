import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """ Input image to Patch Embeddings """
    def __init__(self, img_size=128, patch_size=16, in_channels=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x_shape = x.shape   # [B, C, H, W]
        x = self.patch_proj(x)    # [B, embed_dim, num_patches]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]; flatten the patch dimension to get a sequence of patches
        return x

class MHSA(nn.Module):
    """ Multi-Head Self-Attention """
    def __init__(self, embed_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
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
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., qkv_bias=False, proj_drop=0., attn_drop=0., ff_drop=0.):
        super().__init__()
        self.first_norm = nn.LayerNorm(embed_dim)

        self.attn = MHSA(embed_dim,
                         num_heads=num_heads,
                         qkv_bias=qkv_bias,
                         attn_drop=attn_drop,
                         proj_drop=proj_drop
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
            img_size=30,
            patch_size=16,
            in_channels=3,
            num_classes=10,
            embed_dim=768,
            num_layers=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            pe_type='learned',  # choices: 'learned', '2d-sincos'; if 'learned', the model learns the PE; if '2d-sincos', the model uses 2D sin-cos PE
            pos_embed_dropout_rate=0.,
            attn_dropout_rate=0.,
            proj_dropout_rate=0.,
            ff_dropout_rate=0.,
            num_register_tokens=0,
            aggregation_method='cls_token', # choices: None, 'cls_token'; if None, the model outputs a feature embedding for each patch/token of the sequence
            use_as_classifier=True,    # if set to False, the model is used as a classifier through an MLP head stacked on top of the transformer backbone
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)

        self.num_pos_embeds = self.patch_embed.num_patches
        self.num_extra_tokens = 0

        self.aggregation_method = aggregation_method
        self.use_as_classifier = use_as_classifier

        # Create cls token and register tokens
        if aggregation_method == 'cls_token':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # define the [cls] token
            self.num_pos_embeds += 1
            self.num_extra_tokens += 1

        if num_register_tokens > 0:
            self.reg_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) # define the register tokens
            self.num_pos_embeds += num_register_tokens
            self.num_extra_tokens += num_register_tokens

        # Positional embeddings
        self.pos_embed = self.get_pos_embed(pe_type)    # [1, num_pos_embeds, embed_dim]
        self.pos_drop = nn.Dropout(p=pos_embed_dropout_rate)    # dropout right after the positional encoding and residual

        self.transformer_layers = nn.Sequential(*[
            TransformerLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_dropout_rate,
                attn_drop=attn_dropout_rate,
                ff_drop=ff_dropout_rate
            )
            for _ in range(num_layers)])

        self.last_layer_norm = nn.LayerNorm(embed_dim)

        if use_as_classifier:
            # Classification head
            self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights()


    def get_pos_embed(self, pe_type='learned'):
        """ Get positional embedding """

        if pe_type == 'learned':    # learned positional embedding
            pos_embed = nn.Parameter(torch.zeros(1, self.num_pos_embeds, self.embed_dim))        
            pos_embed.requires_grad = True

        elif pe_type == '2d-sincos':    # 2D sin-cos fixed positional embedding
            h, w = self.patch_embed.grid_size
            grid_w = torch.arange(w, dtype=torch.float32)
            grid_h = torch.arange(h, dtype=torch.float32)
            grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
            assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
            pos_dim = self.embed_dim // 4
            omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
            omega = 1. / (10000**omega)
            out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
            out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
            pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

            # Handle extra tokens such as the [cls] and register tokens
            if self.num_extra_tokens > 0:
                pe_token = torch.zeros([1, self.num_extra_tokens, self.embed_dim], dtype=torch.float32)     # define the PE for the [cls] and register tokens to be concatenated at the beginning of the PE for the patches
                pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))    # [1, (num_patches+num_extra_tokens), embed_dim]; the PE will be added to the input embeddings of the ViT in the forward pass of the ViT backbone (VisionTransformer)

            pos_emb = nn.Parameter(pos_emb)
            pos_emb.requires_grad = False    # NOTE: set to False for the PE to not be learned; TODO: so the PE of the extra tokens is fixed to zero?

        return pos_embed

    def init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if hasattr(self, 'reg_tokens'):
            nn.init.trunc_normal_(self.reg_tokens, std=0.02)
        if hasattr(self, 'cls_token'):
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward_encode(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) # [B, num_patches, embed_dim]

        # Concatenate the [cls] token and register tokens (if any) to the input embeddings
        # TODO: depending on if we want to use PE for the extra tokens, we should add first the PE and then concatenate or concatenate first and then add the PE
        if hasattr(self, 'reg_tokens'):
            reg_tokens = self.reg_tokens.expand(B, -1, -1)   # [B, num_register_tokens, embed_dim]; use expand to create and add the tokens to each sample in the batch instead of just one sample
            x = torch.cat((reg_tokens, x), dim=1) 
        if hasattr(self, 'cls_token'):
            cls_tokens = self.cls_token.expand(B, -1, -1)   # [B, 1, embed_dim]; use expand to create and add the token for each sample in the batch instead of just one sample
            x = torch.cat((cls_tokens, x), dim=1)   # [B, num_pos_embeds, embed_dim]
        
        # Add the positional encoding
        x = x + self.pos_embed  # [B, num_pos_embeds, embed_dim], where num_pos_embeds depends on the number of patches and extra tokens (if any) (e.g., cls ro register tokens)
        x = self.pos_drop(x)    # [B, num_pos_embeds, embed_dim]

        # Transformer layers/blocks
        x = self.transformer_layers(x)  # [B, num_pos_embeds, embed_dim]

        # Last layer norm
        x = self.last_layer_norm(x) # [B, num_pos_embeds, embed_dim]

        # TODO FIXME
        # Where and how to remove the extra tokens (e.g., cls or register tokens) from the output? Especially the register tokens, as the cls may still be used (or removed in forward_features)
        
        return x
    
    def forward_features(self, x):
        if self.aggregation_method == 'cls_token':
            # Aggregate features to the cls token
            # TODO: Should we do some sort of pooling, or does the model learn to aggregate the features to the cls token?
            #       What about attention? Should the cls token attend and be attended by the other tokens? See masks.
            x = x[:, 0, :]  # [B, embed_dim]
            # x = x.mean(dim=1)   # [B, embed_dim]
        else:
            # Output the features for each patch/token after having removed the extra tokens (e.g., cls and register tokens) if any
            x = x[:, self.num_extra_tokens:, :]    # [B, num_pos_embeds - num_extra_tokens, embed_dim]

        return x

    def forward(self, x):
        x = self.forward_encode(x)  # [B, num_pos_embeds, embed_dim] <-- [B, C, H, W]

        # Output a single embedding for the input image
        x = self.forward_features(x)    # [B, embed_dim] <-- [B, num_pos_embeds, embed_dim]

        if self.use_as_classifier:
            # Output classification logits
            x = self.head(x)    # [B, num_pos_embeds, num_classes] if no aggregation, otherwise [B, num_classes]; classification head

        return x
    

if __name__ == '__main__':

    # Test the model
    B = 2
    C = 1
    H = 30
    W = 30
    x = torch.randn(B, C, H, W)

    # Instance 1: (essentially CVR approach, good if we want to predict a single class for a sequence of patches)
    model = VisionTransformer(
        img_size=30,
        patch_size=1,   # if 1, tokens are pixels instead of patches
        in_channels=1,
        num_classes=10,
        embed_dim=384,
        num_layers=8,
        num_heads=8,
        mlp_ratio=4.,
        qkv_bias=True,
        pos_embed_dropout_rate=0.,
        attn_dropout_rate=0.,
        proj_dropout_rate=0.,
        ff_dropout_rate=0.,
        num_register_tokens=2,
        aggregation_method='cls_token', 
        use_as_classifier=True
    )

    # Get the predictions
    print("Results using the model as a classifier outputting a single embedding for the input image")
    y = model(x)  # [B, num_classes]; raw logits for each input sequence
    print("Output shape: ", y.shape)
    y = torch.argmax(y, dim=-1) # [B]; predicted classes
    print("Predictions (output after softmax) shape: ", y.shape)
    print("Predictions: ", y)

    # Instance 2 (approach that loses too much information for REARC and BEFOREARC to be useful, no?)
    model = VisionTransformer(
        img_size=30,
        patch_size=1,   # if 1, tokens are pixels instead of patches
        in_channels=1,
        num_classes=10,
        embed_dim=384,
        num_layers=8,
        num_heads=8,
        mlp_ratio=4.,
        qkv_bias=True,
        pos_embed_dropout_rate=0.,
        attn_dropout_rate=0.,
        proj_dropout_rate=0.,
        ff_dropout_rate=0.,
        num_register_tokens=2,
        aggregation_method='cls_token', 
        use_as_classifier=False
    )

    # Get the predictions
    print("Results using the model as an encoder outputting features aggregated to the [cls] token")
    y = model(x)  # [B, embed_dim]; features embedding (embedding of the cls token) representing the full input sequence encoded by the transformer
    print("Output shape: ", y.shape)
    print("Output (feature embedding): ", y)


    # Instance 3 (good approach for REARC and BEFOREARC?)
    model = VisionTransformer(
        img_size=30,
        patch_size=1,   # if 1, tokens are pixels instead of patches
        in_channels=1,
        num_classes=10,
        embed_dim=384,
        num_layers=8,
        num_heads=8,
        mlp_ratio=4.,
        qkv_bias=True,
        pos_embed_dropout_rate=0.,
        attn_dropout_rate=0.,
        proj_dropout_rate=0.,
        ff_dropout_rate=0.,
        num_register_tokens=2,
        aggregation_method=None, 
        use_as_classifier=True
    )

    # Get the predictions
    print("Results using the model as a classifier outputting raw logits for each patch/token in the input sequence")
    y = model(x)    # [B, num_patches, num_classes]; raw logits for each of the token in the sequence; note that the extra tokens (e.g. cls or registers) are not considered in the output
    print("Output shape: ", y.shape)
    y = torch.argmax(y, dim=-1) # [B, num_patches]; predicted classes for each of the token in the sequence
    print("Predictions (output after softmax) shape: ", y.shape)
    print("Predictions (for each patch/token in the sequence): ", y)

    # Instance 4 (good approach for REARC and BEFOREARC if we want to use a Transformer decoder on top of the ViT encoder?)
    model = VisionTransformer(
        img_size=30,
        patch_size=1,   # if 1, tokens are pixels instead of patches
        in_channels=1,
        num_classes=10,
        embed_dim=384,
        num_layers=8,
        num_heads=8,
        mlp_ratio=4.,
        qkv_bias=True,
        pos_embed_dropout_rate=0.,
        attn_dropout_rate=0.,
        proj_dropout_rate=0.,
        ff_dropout_rate=0.,
        num_register_tokens=2,
        aggregation_method=None, 
        use_as_classifier=False
    )

    # Get the predictions
    print("Results using the model as an encoder outputting features for each patch/token in the sequence")
    y = model(x)    # [B, num_patches, embed_dim]; features embeddings for each token in the full input sequence encoded by the transformer
    print("Output shape: ", y.shape)
    print("Output (features embeddings for each patch/token): ", y)    # features embeddings for each token in the full input sequence encoded by the transformer; we do not consider the extra tokens (e.g. cls or registers) in the output


import torch
import torch.nn as nn

# Personal codebase dependencies
from utility.logging import logger


__all__ = ['get_transformer_decoder']


def create_absolute_positional_encoding(ape_type: str, tgt_seq_len: int, embed_dim: int) -> nn.Parameter:
    """ 
    Create and return the desired APE (absolute positional embedding).
    TODO: What PE to consider for decoding?    
    """
    
    if ape_type == 'learn':    # learned positional encoding
        pos_embed = nn.Parameter(torch.randn(1, tgt_seq_len, embed_dim))  # [1, seq_len, embed_dim]
        pos_embed.requires_grad = True    # NOTE: set to True for the PE to be learned

    elif ape_type == '2dsincos':    # 2D sin-cos fixed positional embedding
        h, w = tgt_seq_len**0.5, tgt_seq_len**0.5
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

        pos_embed = nn.Parameter(pos_embed)
        pos_embed.requires_grad = False    # NOTE: set to False for the PE to not be learned; TODO: so the PE of the extra tokens is fixed to zero?

    else:
        raise ValueError(f"Invalid positional encoding type: {ape_type}. Choose from ['learn', '2dsincos']")


    return pos_embed


class TransformerDecoder(nn.TransformerDecoder):
    def __init__(self, model_config, network_config, decoder_layer, max_seq_len):
        super().__init__(decoder_layer, network_config.num_layers)

        # TODO: How to get the max sequence length of the target sequence?
        self.max_seq_len = max_seq_len

        # Get the positional encoding
        self.ape = create_absolute_positional_encoding(model_config.ape_type, self.max_seq_len, network_config.embed_dim)

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

        # Weights init: Transformer Encoder
        self.apply(self._init_weights)

    # NOTE: Masks for the forward() method
    # 1) Causal (Look-ahead) (Self-Attention) (additive) mask. Use tgt_mask in forward() of nn.TransformerDecoder. Prevents the decoder from looking at the future tokens/positions. Use an upper triangular matrix where future positions are -inf and otherwise 0.
    # 2) [Not needed] Padding mask. Use tgt_key_padding_mask in forward() of nn.TransformerDecoder. Similar to the encoder. It is used to mask the padding tokens in the target sequence.
    # 3) [Not needed] Cross-Attention masks. Use memory_mask in forward() of nn.TransformerDecoder. It is used for custom attention masking of the memory sequence. Use value 0 for allowed and -inf for masked positions.
    # 4) [Not needed] Use memory_key_padding_mask in forward() of nn.TransformerDecoder. It is essentially the same as src_key_padding_mask for the forward() of nn.TransformerEncoder. It is used to prevent the decoder from attending to the padding tokens in the memory sequence.

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # NOTE: tgt_mask is used as causal mask for tgt. Shape: [tgt_seq_len, tgt_seq_len]
        B, tgt_seq_len, D = tgt.shape

        # In training mode, full teacher forcing is used. Hence the decoding is done in parallel for all tokens in the sequence using a causal mask. Thus, tgt is simply the ground-truth y.
        # In eval mode, the decoding is done in an autoregressive manner (i.e., the decoding is done token by token with all the previously predicted tokens used as tgt). Hence, tgt is the predicted sequence so far.
        
        # NOTE: The torch nn.TransformerDecoder implementation always returns as many token embeddings as there were tokens in the target sequence given.
        #       That is why using a causal mask even for inference AR decoding is important in order to make sure we do not use future tokens, in case we would for some reason use the outputted token embeddings of previous steps.

        # TODO:
        # Correct to use Positional Encoding on tgt by simply truncating the PE (of length max_tgt_seq_len) to the length of the tgt sequence?
        tgt += self.ape[:, :tgt_seq_len, :].to(tgt.device)  # [B, tgt_seq_len, embed_dim]

        # Forward pass through the Transformer Decoder Layers
        tgt_decoded = super().forward(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)    # [B, tgt_seq_len, embed_dim]
        
        # NOTE: We make the choice to output the embeddings and not output the logits ([B, tgt_seq_len, num_classes]) by using an additional Linear layer here
        #       That is what TransformerDecoder does. Hence, we should not forget to add a Linear layer after calling the decoder if we want to get the logits.
        
        return tgt_decoded

def get_transformer_decoder(model_config, network_config, max_seq_len, device):
    """Returns a Transformer decoder instance"""

    decoder_layer = nn.TransformerDecoderLayer(
        d_model=network_config.embed_dim,   # the dimension of the embeddings for each input and output token
        nhead=network_config.num_heads,     # the dimension of the embeddings for each head is: d_model // num_heads
        dim_feedforward=network_config.embed_dim * network_config.mlp_ratio,  # ff_dim; the hidden dimension of the feedforward network model
        batch_first=True, 
        device=device   # 'cuda'
    )

    decoder = TransformerDecoder(model_config, network_config, decoder_layer, max_seq_len)
    return decoder
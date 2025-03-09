import torch
import torch.nn as nn

# Personal codebase dependencies
from utility.logging import logger


__all__ = ['get_transformer_encoder']


# TODO: see how to build and use the 2D sin-cos positional encoding
def get_positional_encoding():
    pass

class TransformerEncoder(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers):
        super().__init__(encoder_layer, num_layers)

        # Weights initialization
        self._init_weights()

        # Get the positional encoding
        self.pos_enc = get_positional_encoding()

    # TODO: check if override?
    def _init_weights(self,):
        pass

    # TODO: implement masks for the forward() method
    # 1) [Not needed] (Self-)Attention mask. Use mask in forward() of nn.TransformerEncoder. It is used for custom attention masking. Use value 0 for allowed and -inf for masked positions.
    # 2) [Needed?] Padding mask. Use src_key_padding_mask in forward() of nn.TransformerEncoder. It is used to mask the padding (or other) tokens in the input sequence. Use value True for padding tokens and False for actual tokens.

    def forward(self, src, mask=None, src_key_padding_mask=None):
        return super().forward(src, mask, src_key_padding_mask)


def get_transformer_encoder(network_config, device):
    """Returns a Transformer encoder instance"""

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=network_config.embed_dim,   # the dimension of the embeddings for each input and output token
        nhead=network_config.num_heads,     # the dimension of the embeddings for each head is: d_model // num_heads
        dim_feedforward=network_config.ff_dim,  # the hidden dimension of the feedforward network model
        batch_first=True, 
        device=device   # 'cuda'
    )

    encoder = TransformerEncoder(encoder_layer, num_layers=network_config.num_layers)
    return encoder


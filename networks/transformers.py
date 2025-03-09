import torch
import torch.nn as nn

# Personal codebase dependencies
from utility.logging import logger


__all__ = ['get_transformer']


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

class TransformerDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers):
        super().__init__(decoder_layer, num_layers)

        # Weights initialization
        self._init_weights()

        # Get the positional encoding
        self.pos_enc = get_positional_encoding()

    # TODO: check if override?
    def _init_weights(self,):
        pass

    # TODO: implement masks for the forward() method
    # 1) Causal (Look-ahead) mask. Use tgt_mask in forward() of nn.TransformerDecoder. Prevents the decoder from looking at the future tokens/positions. Use an upper triangular matrix where future positions are -inf and diagonal is 1.
    # 2) [Needed?] Padding mask. Use tgt_key_padding_mask in forward() of nn.TransformerDecoder. Similar to the encoder. It is used to mask the padding tokens in the target sequence.
    # 3) [Not needed] Cross-Attention masks. Use memory_mask in forward() of nn.TransformerDecoder. It is used for custom attention masking of the memory sequence. Use value 0 for allowed and -inf for masked positions.
    # 4) [Needed?] Use memory_key_padding_mask in forward() of nn.TransformerDecoder. It is essentially the same as src_key_padding_mask for the forward() of nn.TransformerEncoder. It is used to prevent the decoder from attending to the padding tokens in the memory sequence.

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        return super().forward(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)

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

def get_transformer_decoder(network_config, device):
    """Returns a Transformer decoder instance"""

    decoder_layer = nn.TransformerDecoderLayer(
        d_model=network_config.embed_dim,   # the dimension of the embeddings for each input and output token
        nhead=network_config.num_heads,     # the dimension of the embeddings for each head is: d_model // num_heads
        dim_feedforward=network_config.ff_dim,  # the hidden dimension of the feedforward network model
        batch_first=True, 
        device=device   # 'cuda'
    )

    decoder = TransformerDecoder(decoder_layer, num_layers=network_config.num_layers)
    return decoder

def get_transformer(model_config, backbone_network_config, head_network_config):
    """Returns a vanilla Transformer encoder and decoder"""

    if torch.cuda.is_available():
        device = torch.device("cuda")
        for i in range(torch.cuda.device_count()):
            logger.info(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA is not available. Running on CPU.")

    assert backbone_network_config.embed_dim == head_network_config.embed_dim, "The embedding dimensions of the backbone and head networks must be the same. Otherwise, implement a Linear projection layer before the decoder input to match the decoder embedding dimension."

    encoder = get_transformer_encoder(backbone_network_config, device)
    decoder = get_transformer_decoder(head_network_config, device)
    return encoder, decoder

import torch
import torch.nn as nn
from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import top_k, top_p
from torch.amp import autocast

# Personal codebase dependencies
from utility.custom_logging import logger
from utility.utils import timer_decorator

__all__ = ['get_xtransformer_decoder']


class XTransformerDecoder(nn.Module):
    def __init__(self, model_config, network_config, num_classes, seq_len):
        """
        Use x-transformers for a decoder with cross-attention and KV caching.
        """
        super().__init__()

        embed_dim = network_config.embed_dim
        num_heads = network_config.num_heads
        num_layers = network_config.num_layers

        self.model_config = model_config
        self.network_config = network_config
        self.num_classes = num_classes
        self.seq_len = seq_len

        self.use_amp_inference = True

        self.PAD_TOKEN = 10
        self.BOS_TOKEN = 15
        self.vocab_size = self.num_classes + 1
        log_message = f"Decoder: BOS={self.BOS_TOKEN}, PAD={self.PAD_TOKEN}"

        self.use_EOS_for_decoding = False
        if self.use_EOS_for_decoding:
            self.EOS_TOKEN = 16
            log_message += f", EOS={self.EOS_TOKEN}"
            self.vocab_size += 1
        else:
            self.EOS_TOKEN = None

        log_message += f"\nTotal vocab size: {self.vocab_size}"
        logger.info(log_message)

        # x-transformers
        self.decoder_core = TransformerWrapper(
            num_tokens=self.vocab_size, # NOTE: token embeddings are handled internally, so we will input token ids
            max_seq_len=self.seq_len,
            # use_abs_pos_emb = True,
            # scaled_sinu_pos_emb = True,
            # l2norm_embed = True,
            attn_layers=Decoder(
                dim=embed_dim,
                depth=num_layers,
                heads=num_heads,
                ff_mult=network_config.mlp_ratio,
                # adaptive_condition_mlp = True,
                # adaptive_condition_mlp_expansion=4,
                cross_attend=True,  # using memory/context
                attn_flash=True,
                # use_adaptive_rmsnorm = True,
                # use_rmsnorm = True,
                # use_simple_rmsnorm = True,
                # alibi_pos_bias = True,
                rotary_pos_emb = True,  # more stable KV caching ?
                rotary_emb_dim = embed_dim // num_heads, # dimensionality of rotary embeddings
                # rotary_xpos = True,
                # weight_tie_layers = True,
            ),
            # NOTE: Do NOT set logits_dim here. We want embeddings out.
        )

        # self.decoder_core = torch.compile(self.decoder_core)

        # To predict the next token based on the current step's embedding.
        # To go from an embedding of dim=embed_dim to dim=vocab_size (to predict any token in the vocab)
        self.decoder_output_layer = nn.Linear(embed_dim, self.vocab_size)

        # To go from an embedding of dim=embed_dim to dim=num_classes (logits)
        self.final_output_layer = nn.Linear(embed_dim, num_classes)

    # @timer_decorator
    def training_decode(self, y, x_encoded):
        """
        Perform parallel decoding with full teacher forcing.
        """
        device = x_encoded.device
        y = y.to(device)
        B, S_target = y.shape
        B_mem, S_mem, D_mem = x_encoded.shape

        # Prepend the start token to the target sequence
        start_token = torch.full((B, 1), self.BOS_TOKEN, dtype=torch.long, device=device)  # [B, 1]
        decoder_input_tokens = torch.cat([start_token, y], dim=1) # [B, S_target + 1]

        # We do NOT want to ignore any token (padding tokens too), so mask is all False
        tgt_mask = torch.zeros_like(decoder_input_tokens, dtype=torch.bool, device=device)

        # Parallel Decoding (with teacher forcing)
        output_embeddings = self.decoder_core(
            x=decoder_input_tokens,     # token IDs [B, S_target + 1]
            context=x_encoded,          # encoder memory [B, S_mem, D]
            mask=tgt_mask,              # target padding mask [B, S_target + 1]; causal masking is handled internally by the Decoder layers
            # context_mask=None,        # add if x_encoded can have padding [B, S_mem]
            return_embeddings=True,     # to return embeddings
            return_intermediates=False  # no cache needed for parallel training
        )   # [B, S_target + 1, E]

        # Only keep the output embeddings that predict y_1, y_2, ..., y_S_target
        relevant_embeddings = output_embeddings[:, :-1, :]  # [B, S_target, E]

        return relevant_embeddings

    # @timer_decorator
    def inference_decode(self, x_encoded):
        """
        Performs autoregressive inference using KV caching.
        Output a sequence of embeddings of the predicted tokens.
        """

        B, S_mem, D = x_encoded.shape[0]
        device = x_encoded.device

        # Initialize KV cache
        cache = None

        # To store embeddings for each token generated
        all_embeddings = []

        # Initialize sequence with BOS token
        current_tokens = torch.full((B, 1), self.BOS_TOKEN, dtype=torch.long, device=device) # [B, 1]

        self.eval()
        with torch.no_grad():
            # AR Decoding
            for t in range(self.seq_len):
                # Get only the last token for input (KV cache handles Key-Value for the previous tokens)
                input_token = current_tokens[:, -1:] # [B, 1]

                # Needs to be a tensor indicating the start position (step index 't') for each batch element.
                start_pos_tensor = torch.full((B,), t, dtype=torch.long, device=device) # [B]

                # Run core decoder for a single step with caching
                # Causal masking is handled internally by the Decoder layers.
                # Pass cache, ask for cache back, provide step index tensor
                with autocast(device_type=device.type, enabled=self.use_amp_inference):
                    step_output, cache = self.decoder_core(
                        x=input_token,              # only the newest token [B, 1]
                        context=x_encoded,          # encoder memory [B, S_mem, D]
                        cache=cache,                # current cache state
                        seq_start_pos=start_pos_tensor, # pass tensor, not int
                        return_intermediates=True,  # get updated cache back
                        return_embeddings=True      # to return embeddings
                        )   # step_output: [B, 1, E], cache: []

                # Extract the embedding for the current step
                current_embedding = step_output[:, -1, :] # [B, E]
                all_embeddings.append(current_embedding)

                # Predict the next token
                step_logits = self.decoder_output_layer(current_embedding) # [B, vocab_size]
                next_token = torch.argmax(step_logits, dim=-1, keepdim=True) # [B, 1]; greedy sampling; TODO: use top_k, top_p, etc. for better sampling

                # Store the predicted token
                current_tokens = torch.cat([current_tokens, next_token], dim=1) # [B, current_len + 1]

        # Stack the embeddings from all generated steps
        output_embeddings = torch.stack(all_embeddings, dim=1) # [B, seq_len, E]

        return output_embeddings
    
    # @timer_decorator
    def decode_sequence(self, y, x_encoded):
        if self.training:   # use PTL LightningModule's self.training attribute to check if the model is in training mode; could also use self.trainer.training, self.trainer.validating, self.trainer.testing
            # Parallel decoding with full teacher forcing. All positions are predicted at the same time.
            output_target_seq = self.training_decode(y, x_encoded)
        else:
            # AR decoding. The model predicts one token at a time in an auto-regressive manner.
            output_target_seq = self.inference_decode(x_encoded)

        return output_target_seq

    def forward(self, tgt, memory):
        """
        NOTE:
        We make the choice to output the logits ([B, S, num_classes]) instead of the embeddings. That is why we use a Linear final output layer here.
        """

        # Decode the target sequence using the encoder output as memory
        output_target_seq = self.decode_sequence(tgt, memory)   # [B, S, E]
        
        # Apply a final linear output layer to get the logits from the predicted target sequence embeddings
        logits = self.final_output_layer(output_target_seq)  # [B, S, num_classes]

        return logits

def get_xtransformer_decoder(model_config, network_config, num_classes, seq_len):
    """Returns a Transformer decoder instance"""

    decoder = XTransformerDecoder(model_config, network_config, num_classes, seq_len)
    return decoder
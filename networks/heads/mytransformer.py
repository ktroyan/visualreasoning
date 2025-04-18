import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Personal codebase dependencies
from utility.logging import logger
from utility.utils import timer_decorator


__all__ = ['get_mytransformer_decoder']

"""
TODO: Check if pre/post norm correct. Check with original paper
"""


def create_causal_mask(size):
    """ 1/True elements are masked out """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask

def create_absolute_positional_encoding(ape_type: str, ape_length: int, embed_dim: int) -> nn.Parameter:
    """ 
    Create and return an APE (absolute positional embedding) to be added to the network input in the forward pass.
    TODO: What PE to consider for decoding?
    NOTE: The BOS token is considered for the APE, and the current token to predict is not, hence the APE length is seq_len.
    """

    if ape_type == 'learn':    # learned positional encoding
        # pos_embed_layer = nn.Embedding(ape_length, embed_dim)
        # torch.nn.init.normal_(pos_embed_layer.weight, std=0.02)
        # pos_embed = pos_embed_layer(torch.arange(ape_length)).unsqueeze(0)
        # pos_embed.requires_grad = True

        pos_embed = nn.Parameter(torch.randn(1, ape_length, embed_dim))  # [1, seq_len, embed_dim]
        pos_embed.requires_grad = True    # NOTE: set to True for the PE to be learned

    elif ape_type == '2dsincos':    # 2D sin-cos fixed positional embedding
        h, w = (ape_length-1)**0.5, (ape_length-1)**0.5 # TODO: Correct?
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

        # Concatenate PE for BOS token
        pos_embed = torch.cat([torch.zeros(1, 1, embed_dim), pos_embed], dim=1)  # [1, seq_len+1, embed_dim]; first token is the BOS token, so we set its PE to zero

        pos_embed = nn.Parameter(pos_embed)
        pos_embed.requires_grad = False    # NOTE: set to False for the PE to not be learned

    else:
        raise ValueError(f"Invalid positional encoding type: {ape_type}. Choose from ['learn', '2dsincos']")


    return pos_embed

class DropResNorm(nn.Module):
    def __init__(self, embed_dim, dropout=0.0, use_pre_norm=True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_pre_norm = use_pre_norm

    def forward(self, x, residual):
        """ x is the target sequence, residual is what is added to the target sequence flowing """
        if self.use_pre_norm:   # pre-norm
            x = self.layer_norm(x)
            x = x + self.dropout(residual)
        else:                   # post-norm
            x = x + self.dropout(residual)
            x = self.layer_norm(x)
        
        return x

class MHSA(nn.Module):
    """ 
    Multi-Head Self-Attention for Decoder with KV Caching.
    Use F.scaled_dot_product_attention for efficiency. --> See PyTorch documentation
    """
    def __init__(self, embed_dim, num_heads, qkv_bias, mha_dropout=0.0, is_causal=True):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.mha_dropout = mha_dropout
        self.is_causal = is_causal

        # Query, Key, Value (QKV) projection with a single matrix/layer
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias) # W_qkv; similar to having three separate linear layers for Q, K, V as W_q, W_k, W_v
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim) # W_o; output projection after having merged the attention heads


    def forward(self, tgt, attn_mask=None, past_kv=None):

        B, S, E = tgt.shape

        is_inference = past_kv is not None

        # Project Q, K, V
        qkv = self.qkv_proj(tgt)    # [B, S, 3*E]

        # Reshape Queries, Keys, Values for Multi-Head: [B, num_heads, S, head_dim] <-- [B, S, 3*E]
        qkv = qkv.reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # [3, B, num_heads, S, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]    # ([B, num_heads, S, head_dim], [B, num_heads, S, head_dim], [B, num_heads, S, head_dim])

        # KV Caching
        if is_inference:
            # Input tgt is only the current token, so S=1
            # past_kv = (past_key, past_value)
            # past_key and past_value shape: [B, num_heads, S_past, head_dim]
            
            if past_kv[0] is not None:
                # Concatenate current kv with past kv along the sequence dimension
                # kv have S=1 here from the current token
                past_key, past_value = past_kv
                k = torch.cat([past_key, k], dim=2)     # [B, num_heads, S_past + 1, head_dim]
                v = torch.cat([past_value, v], dim=2)   # [B, num_heads, S_past + 1, head_dim]

            # Inference mode: store the updated cache (for the next iteration)
            updated_kv = (k, v)
            
            # Handle causality
            effective_is_causal = False # during inference, causality is handled implicitly by processing one token at a time
            effective_attn_mask = None  # causal mask not needed here
       
        else:
            # Training mode: no cache update needed, so we return None
            updated_kv = None
        
            # Handle causality
            effective_is_causal = self.is_causal # Use causal mask if specified
            effective_attn_mask = attn_mask # Pass the provided causal mask

        # Compute Attention (with efficient PyTorch primitive for SDPA)
        attn_output = F.scaled_dot_product_attention(
            q,  # query
            k,  # key
            v,  # value
            attn_mask=effective_attn_mask,  # causal mask for training
            dropout_p=self.mha_dropout if self.training else 0.0,   # apply dropout only during training
            is_causal=effective_is_causal,   # use built-in causal masking if True
            # scale=self.embed_dim**-0.5,   # scaling factor for the attention scores
        )   # [B, num_heads, S_q, head_dim]; S_q = 1 for inference, S for training

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, self.embed_dim)  # [B, S_q, E] <-- [B, S_q, num_heads * head_dim] <-- [B, num_heads, S_q, head_dim]

        # Output Projection (W_o)
        attn_output = self.out_proj(attn_output)    # [B, S_q, E]

        # No need for dropout as it is done in the SDPA computation directly (?)

        return attn_output, updated_kv

class MHCA(nn.Module):
    """ Multi-Head Cross-Attention for Decoder """
    
    def __init__(self, embed_dim, num_heads, qkv_bias, mha_dropout=0.0):
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, bias=qkv_bias, dropout=mha_dropout, batch_first=True)

    def forward(self, tgt, memory):
        
        cross_attn_output, _ = self.cross_attn(query=tgt,           # [B, S_tgt, E]; query from decoder
                                               key=memory,          # [B, S_mem, E]; key from encoder
                                               value=memory,        # [B, S_mem, E]; value from encoder
                                               attn_mask=None,      # no causal mask needed for cross-attention
                                               need_weights=False   # if need to return the attention weights computed
                                               )
        
        return cross_attn_output

class FeedForward(nn.Module):
    """ FeedForward layer """
    
    def __init__(self, embed_dim, ff_dim, ff_dropout=0.0, activation="gelu"):
        super().__init__()
        self.ff_linear1 = nn.Linear(embed_dim, ff_dim)
        self.ff_linear2 = nn.Linear(ff_dim, embed_dim)
        self.ff_dropout = nn.Dropout(ff_dropout)
        
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu

    def forward(self, x):
        x = self.ff_linear1(x)
        x = self.activation(x)
        x = self.ff_dropout(x)
        x = self.ff_linear2(x)
        
        return x

class MyTransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias, ff_dim, mhsa_dropout=0.0, mhca_dropout=0.0, ff_dropout=0.0, drn_dropout=0.0):
        super().__init__()
        
        # Multi-Head Self-Attention
        self.mhsa = MHSA(embed_dim, num_heads, qkv_bias, mha_dropout=mhsa_dropout, is_causal=True)
        self.mhsa_dropresnorm = DropResNorm(embed_dim, dropout=drn_dropout)

        # Multi-Head Cross-Attention
        self.mhca = MHCA(embed_dim, num_heads, qkv_bias, mha_dropout=mhca_dropout)
        self.mhca_dropresnorm = DropResNorm(embed_dim, dropout=drn_dropout)

        # FeedForward Network
        self.feed_forward = FeedForward(embed_dim, ff_dim, ff_dropout)
        self.ff_dropresnorm = DropResNorm(embed_dim, dropout=drn_dropout)


    def forward(self,
                tgt,            # [B, S_tgt, E]; target sequence input
                memory,         # [B, S_mem, E]; encoder output
                tgt_mask=None,  # [S_tgt, S_tgt]; causal mask for MHSA
                kv_cache=None   # Tuple[(K, V)]; KV Cache for MHSA (inference)
                ):

        # Multi-Head Self-Attention (with KV caching for inference)
        mhsa_output, updated_kv_cache = self.mhsa(tgt,
                                                  attn_mask=tgt_mask,   # used only in training as inference handles causality by construction
                                                  past_kv=kv_cache
                                                  )
        
        # LayerNorm, Residual, Dropout
        tgt = self.mhsa_dropresnorm(tgt, mhsa_output)

        # Multi-Head Cross-Attention
        mhca_output = self.mhca(tgt, memory)
        
        # LayerNorm, Residual, Dropout
        tgt = self.mhca_dropresnorm(tgt, mhca_output)

        # FeedForward
        ff_output = self.feed_forward(tgt)
        
        # LayerNorm, Residual, Dropout
        tgt = self.ff_dropresnorm(tgt, ff_output)

        return tgt, updated_kv_cache
    

class MyTransformerDecoder(nn.Module):
    def __init__(self, model_config, network_config, num_classes, seq_len):
        super().__init__()

        self.model_config = model_config
        self.network_config = network_config
        
        self.num_classes = num_classes
        self.seq_len = seq_len

        embed_dim = network_config.embed_dim
        num_heads = network_config.num_heads
        num_layers = network_config.num_layers

        
        ## Recall data special tokens that are relevant to consider in this decoder
        self.PAD_TOKEN = 10  # padding token

        
        ## Define decoder's special control tokens
        log_message = "Special decoding tokens for Transformer decoder: "
        self.BOS_TOKEN = 15  # beginning of sequence token
        self.vocab_size = num_classes + 1  # number of different tokens that we consider; +1 for the BOS token
        self.pos_len = seq_len + 1  # length of the sequence for the APE; +1 for the BOS token
        log_message += f"BOS={self.BOS_TOKEN}"
        
        self.use_EOS_for_decoding = False   # TODO: Not sure how to handle when True (see below in inference_decode() function)
        if self.use_EOS_for_decoding:
            self.EOS_TOKEN = 16  # end of sequence token
            log_message += f", EOS={self.EOS_TOKEN}"
            self.vocab_size = self.vocab_size + 1  # number of different tokens that we consider; +1 for EOS token

        logger.info(log_message)


        ## Embed tgt tokens
        # Create a target projection layer to map the ground truth target tokens/sequence (i.e., y) to the decoder embedding dimension as a Transformer Decoder needs to receive the target sequence in an embedding space
        self.embed_tgt  = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=network_config.embed_dim)
        self.scale_emb = math.sqrt(embed_dim)   # more stable training?

        ## Absolute positional encoding
        # TODO: This assumes that we use the same APE as for the encoder since it comes from the model config. OK?
        if model_config.ape.enabled:
            self.ape = create_absolute_positional_encoding(model_config.ape.ape_type, self.pos_len, network_config.embed_dim)
            # self.register_parameter("ape", self.ape) # no need to register as ape is already a nn.Parameter instance

        ## Decoder layers
        self.layers = nn.ModuleList([
            MyTransformerDecoderLayer(embed_dim=embed_dim,
                                      num_heads=num_heads,
                                      qkv_bias=network_config.qkv_bias,
                                      ff_dim=network_config.embed_dim * network_config.mlp_ratio, 
                                      mhsa_dropout=network_config.attn_dropout_rate,
                                      mhca_dropout=network_config.attn_dropout_rate,
                                      ff_dropout=network_config.ff_dropout_rate,
                                      drn_dropout=0.0   # not considered in the config for now
                                      )
            for _ in range(num_layers)
        ])

        # Final LayerNorm (after the last decoder layer)
        self.last_layernorm = nn.LayerNorm(embed_dim)

        ## Output layer to go from a decoder output embedding to logits; we use self.vocab_size as the EOS token could be predicted
        self.decoder_output_layer = nn.Linear(network_config.embed_dim, self.vocab_size) 

        ## Output layer to go from the final decoded sequence (of token embeddings) to output logits; we use num_classes as the EOS cannot be predicted, only the data tokens and special tokens
        self.final_output_layer = nn.Linear(network_config.embed_dim, num_classes)   

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

        # Absolute Positional Embedding
        if hasattr(self, 'ape') and self.ape.requires_grad:  # check if the APE is learnable
            nn.init.trunc_normal_(self.ape, std=0.02) 


    # @timer_decorator
    def training_decode(self, y, x_encoded):
        """
        Parallel decoding with full teacher forcing.
        """
        
        device = x_encoded.device
        y = y.to(device)
        B, S = y.shape
        B, S_mem, D_mem = x_encoded.shape

        # Add BOS token for start position
        start_token = torch.full((B, 1), self.BOS_TOKEN, dtype=torch.long, device=device)  # [B, 1]
        y_shifted_right = y[:, :-1]
        decoder_input_tokens = torch.cat([start_token, y_shifted_right], dim=1)  # [B, S]

        # Embed tgt tokens
        input_embeddings = self.embed_tgt(decoder_input_tokens) * self.scale_emb  # [B, S, E]

        # Add positional encoding
        if self.model_config.ape.enabled:
            input_embeddings = input_embeddings + self.ape[:, :S]  # [B, S, E]

        # Causal mask
        causal_mask = create_causal_mask(S).to(device)  # [S, S]

        x = input_embeddings  # [B, S, E]
        
        for i, layer in enumerate(self.layers):
            x, _ = layer(tgt=x,                 # current target embeddings
                         memory=x_encoded,      # memory (from encoder output)
                         tgt_mask=causal_mask,  # causal mask for training
                         kv_cache=None          # no KV cache during training
                         )  # x is [B, S, E], kv is not considered for training

        x = self.last_layernorm(x)  # [B, S, E]

        return x


    # @timer_decorator
    def inference_decode(self, x_encoded):
        """
        Autoregressive decoding with key-value caching.
        """

        B, S_mem, D = x_encoded.shape
        device = x_encoded.device
        S = self.seq_len
        
        # Initialize KV cache for each layer
        past_kvs = [(None, None)] * len(self.layers)    # list of tuples (K, V), where K is [B, num_heads, S_cache, head_dim] and V is [B, num_heads, S_cache, head_dim]

        # Start sequence generation with BOS token
        current_token = torch.full((B, 1), self.BOS_TOKEN, dtype=torch.long, device=device)    # [B, 1]

        # Store the predicted embeddings that are used to predict the next token
        # TODO: See if correct how I store and use the embeddings. Correct output at the end conceptually?
        generated_embeddings = []

        # Keep track of whether sequences have finished (i.e., EOS is predicted)
        if self.use_EOS_for_decoding:
            finished_sequences = torch.zeros(B, dtype=torch.bool, device=device)

        self.eval()
        with torch.no_grad():
            for t in range(S):
                # Embed current tokens
                outputs = self.embed_tgt(current_token) * self.scale_emb # [B, 1, E]

                # Add APE
                if self.model_config.ape.enabled:
                    # Get APE for position t
                    outputs = outputs + self.ape[:, t:t+1]    # [1, 1, E] + [1, pos_len, E]

                # Input to the first decoder layer
                x = outputs     # [B, 1, E]
                
                # To store updated KV cache for the current step
                new_kvs = []

                # Decoder layers
                for i, layer in enumerate(self.layers):
                    x, kv_cache = layer(tgt=x,              # [B, 1, E]; input should only be the current token embedding
                                        memory=x_encoded,   # [B, S_mem, E]; memory (encoder output)
                                        kv_cache=past_kvs[i] # cache from the previous step
                                        ) # x is [B, 1, E], kv_cache is [B, num_heads, S_cache, head_dim]
                    
                    new_kvs.append(kv_cache)    # store updated cache (K, V)

                # Final Layer Normalization (on the single token output)
                x = self.last_layernorm(x)  # [B, 1, E]

                # Update the KV cache for the next iteration step
                past_kvs = new_kvs  # .copy() ?

                # Store predicted embeddings that allow to predict the token IDs
                generated_embeddings.append(x)

                # Vocab logits
                vocab_logits = self.decoder_output_layer(x) # [B, 1, vocab_size]

                # Predict/generate next token
                next_token_logits = vocab_logits[:, -1, :]  # logits for the last (here only) token: [B, vocab_size]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [B, 1]; greedy decoding

                # Update current token (for the next iteration step)
                current_token = next_token

                # Check for EOS token and update finished sequences
                if self.use_EOS_for_decoding:
                    just_finished = (next_token.squeeze(1) == self.EOS_TOKEN)   # stop generation for sequences that have produced EOS
                    finished_sequences = finished_sequences | just_finished     # do not stop sequences that were already finished

                    # Stop decoding if all sequences have finished
                    if finished_sequences.all():
                        break

        
        # Concatenate predicted embeddings along the sequence dimension
        output_embeddings = torch.cat(generated_embeddings, dim=1) # [B, generated_seq_len]

        # TODO: 
        # Complete the sequences with the special visual tokens: border tokens (3), padding token (1), newline token (1) ?
        # However in this class we do not have the direct information about what visual tokens are used.
        # Maybe it is ok to infer them from vocab_size since the number of "data tokens" is fixed to 10 and the padding is always considered.
        # But do we have to complete the sequences this way? It seems better to just not use EOS.

        return output_embeddings


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
        Decoding entry point.
        """
        output_target_seq = self.decode_sequence(tgt, memory)   # [B, S, E]

        # Apply the final linear output layer to get class logits
        logits = self.final_output_layer(output_target_seq)  # [B, S, num_classes]; project from embedding dim to num_classes (i.e., excluding special control tokens BOS/EOS)

        return logits
    

def get_mytransformer_decoder(model_config, network_config, num_classes, seq_len):
    """ Return Transformer decoder instance """

    decoder = MyTransformerDecoder(model_config, network_config, num_classes, seq_len)

    return decoder

import torch
import torch.nn as nn

# Personal codebase dependencies
from utility.logging import logger
from utility.utils import timer_decorator


__all__ = ['get_transformer_decoder']


def create_absolute_positional_encoding(ape_type: str, ape_length: int, embed_dim: int) -> nn.Parameter:
    """ 
    Create and return an APE (absolute positional embedding) to be added to the network input in the forward pass.
    TODO: What PE to consider for the decoder?
    TODO: Check if the PE length is correct with a BOS token used. 
          How to handle the special control token BOS, since we use PE for it?
          In 2dsincos, prepend a zero PE (for BOS) and thus we need h as (ape_length-1)**0.5 and w as (ape_length-1)**0.5 ?
          Or ape_length is simply seq_len as y is shifted right ?
    """

    if ape_type == 'learn':    # learned positional encoding
        pos_embed = nn.Parameter(torch.randn(1, ape_length, embed_dim))  # [1, seq_len, embed_dim]
        pos_embed.requires_grad = True    # NOTE: set to True for the PE to be learned

    elif ape_type == '2dsincos':    # 2D sin-cos fixed positional embedding
        h, w = ape_length**0.5, ape_length**0.5 # assuming a square grid image
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
        pos_embed.requires_grad = False    # NOTE: set to False for the PE to not be learned

    else:
        raise ValueError(f"Invalid positional encoding type: {ape_type}. Choose from ['learn', '2dsincos']")


    return pos_embed


class TransformerDecoder(nn.TransformerDecoder):
    def __init__(self, model_config, network_config, decoder_layer, num_classes, seq_len):
        """ 
        Transformer Decoder class. Inherits from nn.TransformerDecoder.

        TODO: 
        See how we want to separate the model_config interactions with this decoder class as in theory
        we want to make as few changes as possible to the decoder since we want to compare the encoder networks principally.
        So for example the APE in this class for the decoder should be decided by the decoder/head network config instead.
        However, it depends for example on how important it is that the decoder uses the same APE type as the encoder.

        """
        
        super().__init__(decoder_layer=decoder_layer, num_layers=network_config.num_layers)

        self.model_config = model_config
        self.network_config = network_config
        self.num_classes = num_classes
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seq_len = seq_len


        ## Recall data special tokens
        self.PAD_TOKEN = 10  # padding token


        ## Define decoder's special tokens
        log_message = "Special decoding tokens for Transformer decoder: "
        self.BOS_TOKEN = 15  # beginning of sequence token
        self.vocab_size = self.num_classes + 1  # number of different tokens that we consider; +1 for the BOS token
        self.pos_length = self.seq_len  # the actual number of positions in the sequence when using the BOS token
        log_message += f"BOS={self.BOS_TOKEN}"
        
        self.use_EOS_for_decoding = False  # TODO: See why error when True; whether to use the EOS token for decoding or not
        if self.use_EOS_for_decoding:
            self.EOS_TOKEN = 16  # end of sequence token
            log_message += f", EOS={self.EOS_TOKEN}"
            self.vocab_size = self.vocab_size + 1  # number of different tokens that we consider; +1 for EOS token

        logger.info(log_message)


        ## Causal mask
        # Create an additive mask to prevent the Transformer Decoder from looking at the future tokens/positions in self-attention module
        # The mask is a square Tensor of size [seq_len, seq_len] with -inf where we want to mask and 0 where we want to keep
        self.register_buffer('tgt_mask', torch.triu(torch.full((seq_len, seq_len), float("-inf"), dtype=torch.float32), diagonal=1))  # [seq_len, seq_len]; register as buffer so that it moves device with the model


        ## Embed tgt tokens
        # Create a target projection layer to map the ground truth target tokens/sequence (i.e., y) to the decoder embedding dimension as a Transformer Decoder needs to receive the target sequence in an embedding space
        self.tgt_projection = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=network_config.embed_dim)


        ## Absolute positional encoding
        if model_config.ape.enabled:
            self.ape = create_absolute_positional_encoding(model_config.ape.ape_type, self.pos_length, network_config.embed_dim)


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
        NOTE:
        In training mode, full teacher forcing is used. Hence the decoding is done in parallel for all tokens in the sequence using a causal mask. 
        Thus, tgt is simply the ground-truth y.
        
        NOTE: 
        The torch nn.TransformerDecoder implementation always returns as many token embeddings as there were tokens in the target sequence given.
        That is why using a causal mask even for inference AR decoding is important in order to make sure we do not use future tokens, in case we would for some reason use the outputted token embeddings of previous steps.
        """
        
        B, S_mem, D_mem = x_encoded.shape
        B, S = y.shape
        device = y.device   # TODO: See how to use self.device instead of y.device, self.device is cpu at __init__ time of Model module

        # Prepare the decoder input: [BOS, y1, y2, ..., y_{S-1}]; the last token y_S is not used as part of the input to the decoder (as there is no EOS token to predict)
        start_token = torch.full((B, 1), self.BOS_TOKEN, dtype=torch.long, device=device)  # [B, 1]; we use a start token as the first token
        y_shifted_right = y[:, :-1] # [B, S-1]
        decoder_input_tokens = torch.cat([start_token, y_shifted_right], dim=1) # [B, S]

        # Embed the decoder input tokens
        tgt = self.tgt_projection(decoder_input_tokens) # [B, S, E]

        # We use APE on the tgt by simply truncating the PE (of length max_tgt_seq_len) to the length of the current tgt sequence
        if self.model_config.ape.enabled:
            tgt += self.ape.to(device)  # [B, S, embed_dim]

        # AR decoding with full teacher forcing. All positions are predicted at the same time.
        # The outputs at positions 0 to S-1 correspond to embedding predictions for y_1 to y_S
        output_target_seq = super().forward(tgt=tgt,   # [B, S, E]
                                            memory=x_encoded, # [B, S_mem, E]
                                            tgt_mask=self.tgt_mask,   # [S, S]
                                            memory_mask=None,
                                            tgt_key_padding_mask=None,
                                            memory_key_padding_mask=None
                                            )    # [B, S, E]; forward pass through the nn.TransformerDecoder Layers

        return output_target_seq

    # @timer_decorator
    def inference_decode(self, x_encoded):
        """
        NOTE:
        In eval mode, the decoding is done in an autoregressive manner (i.e., the decoding is done token by token with all the previously predicted tokens used as tgt).
        Hence, tgt is the predicted sequence so far.
        """

        B, S_mem, D_mem = x_encoded.shape
        device = x_encoded.device   # TODO: See how to use self.device instead of x_encoded.device, self.device is cpu at __init__ time of Model module

        # Make sure the decoder network is in evaluation mode
        self.eval()
        with torch.no_grad():

            # Output embeddings from each step
            output_embeddings = []

            # Initialize the decoded sequence with the start token BOS
            decoded_sequence = torch.full((B, 1), self.BOS_TOKEN, dtype=torch.long, device=device) # [B, 1]

            if self.use_EOS_for_decoding:
                # Keep track of the finished sequences (i.e., those that have already generated the EOS token)
                finished_sequences_flag = torch.zeros(B, dtype=torch.bool, device=device)  # [B]

            # AR decoding loop (no teacher forcing, one token at a time)
            for t in range(self.seq_len):   # we have seq_len tokens to get

                # Get the current length of the decoded sequence
                current_len = decoded_sequence.shape[1] # t+1

                # Embed the tokens part of the current decoded sequence
                decoder_input_embedded = self.tgt_projection(decoded_sequence) # [B, current_len, E]

                # We use APE on the tgt by simply truncating the PE (of length seq_len) to the length of the current tgt sequence
                if self.model_config.ape.enabled:
                    decoder_input_embedded += self.ape[:, :current_len, :].to(device)  # [B, current_len, embed_dim]

                # Get the relevant tgt/causal mask by slicing the general tgt mask ([seq_len, seq_len]) created during initialization
                tgt_mask = self.tgt_mask[:current_len, :current_len] # [current_len, current_len]

                # Compute decoder output embeddings for positions up to t+1
                # The outputs at positions 0 to S-1 correspond to predicted embeddings for tokens y_1 to y_S
                outputs = super().forward(tgt=decoder_input_embedded,   # [B, current_len, E]
                                          memory=x_encoded, # [B, S_mem, E]
                                          tgt_mask=tgt_mask,   # [current_len, current_len]
                                          memory_mask=None,
                                          tgt_key_padding_mask=None,
                                          memory_key_padding_mask=None
                                          )    # [B, current_len, E]; forward pass through the nn.TransformerDecoder Layers
                
                # Get the embedding for the current prediction step (i.e., the last output embedding of the decoder)
                output_embeddings_t = outputs[:, -1, :]   # [B, E]; embedding that allows to predict the next token

                # Store the output embedding for the current step
                output_embeddings.append(output_embeddings_t)

                # Predict the next token
                predicted_token_logits = self.decoder_output_layer(output_embeddings_t) # [B, vocab_size] 
                predicted_token = torch.argmax(predicted_token_logits, dim=-1)   # [B]; TODO: Implement better sampling than argmax (greedy) ?

                if self.use_EOS_for_decoding:
                    # TODO: 
                    # See if/how to use the EOS token for inference decoding.
                    # It seems that we never want to only predict PAD tokens since we introduced the newline tokens
                    # that should figure at the end of each row of the grid.
                    # Thus, if we don't use the EOS token, we need to add +1 to self.num_classes instead of +2.
                    # The issue is that currently the program blocks after some epochs when using use_EOS_for_decoding=True.
                    
                    # Modify the predicted token if needed. This is to handle appending a PAD_TOKEN if a sequence had already finished (i.e., it had predicted the EOS token)
                    newly_finished = (~finished_sequences_flag) & (predicted_token == self.EOS_TOKEN)   # [B]; identify the sequences that have finished generating tokens at this step, so it's True only if the sequence was not already finished (~finished_sequences_flag) AND its current predicted_token is the EOS_TOKEN
                    finished_sequences_flag |= newly_finished   # [B]; use element-wise OR where 1/True means finished and 0/False means not finished
                    finished_previously = finished_sequences_flag & (~newly_finished)   # [B]; identify the sequences that were already finished before the current step, as they will need to be appended a PAD_TOKEN instead of the predicted token
                    token_to_append = predicted_token.masked_fill(finished_previously, self.PAD_TOKEN)   # [B]; for previously finished sequences, masked_fill replaces their predicted_token with self.PAD_TOKEN, otherwise sequences still running or finishing now keep their predicted_token (which could be a regular token or the self.EOS_TOKEN)

                else:
                    token_to_append = predicted_token   # [B]

                # Append the correct newly predicted token. The current decoded sequence is then used for the next iteration's decoder input
                decoded_sequence = torch.cat([decoded_sequence, token_to_append.unsqueeze(1)], dim=1)  # [B, current_len+1]

                if self.use_EOS_for_decoding:
                    # Check if all sequences in the batch have finished generating (i.e., an EOS token has been generated for all sequences)
                    if finished_sequences_flag.all():
                        break
            
            # Prepare the sequence of output embeddings corresponding to the tokens generated during AR decoding. We stack embeddings along the sequence dimension
            output_target_seq = torch.stack(output_embeddings, dim=1)   # [B, seq_len, E]

        return output_target_seq
    
    # @timer_decorator
    def decode_sequence(self, y, x_encoded):
        if self.training:   # use PTL LightningModule's self.training attribute to check if the model is in training mode; could also use self.trainer.training, self.trainer.validating, self.trainer.testing
            # AR decoding with full teacher forcing. All positions are predicted at the same time.
            output_target_seq = self.training_decode(y, x_encoded)
        else:
            # AR decoding. The model predicts one token at a time in an auto-regressive manner.
            output_target_seq = self.inference_decode(x_encoded)

        return output_target_seq

    def forward(self, tgt, memory):
        """
        NOTE:
        We make the choice to output the logits ([B, S, num_classes]) instead of the embeddings. That is why we use a Linear final output layer here.
        Since nn.TransformerDecoder outputs the embeddings, we should not forget to add a Linear layer after calling the nn.TransformerDecoder forward if we want to get the logits.
        """
        # NOTE: Masks for the forward() method
        # 1) Causal (Look-ahead) (Self-Attention) (additive) mask. Use tgt_mask in forward() of nn.TransformerDecoder. Prevents the decoder from looking at the future tokens/positions. Use an upper triangular matrix where future positions are -inf and otherwise 0.
        # 2) [Not needed] Padding mask. Use tgt_key_padding_mask in forward() of nn.TransformerDecoder. Similar to the encoder. It is used to mask the padding tokens in the target sequence.
        # 3) [Not needed] Cross-Attention masks. Use memory_mask in forward() of nn.TransformerDecoder. It is used for custom attention masking of the memory sequence. Use value 0 for allowed and -inf for masked positions.
        # 4) [Not needed] Use memory_key_padding_mask in forward() of nn.TransformerDecoder. It is essentially the same as src_key_padding_mask for the forward() of nn.TransformerEncoder. It is used to prevent the decoder from attending to the padding tokens in the memory sequence.

        # Decode the target sequence using the encoder output as memory
        output_target_seq = self.decode_sequence(tgt, memory)   # [B, S, E]
        
        # Apply a final linear output layer to get the logits from the predicted target sequence embeddings
        logits = self.final_output_layer(output_target_seq)  # [B, S, num_classes]

        return logits

def get_transformer_decoder(model_config, network_config, num_classes, seq_len):
    """Returns a Transformer decoder instance"""

    decoder_layer = nn.TransformerDecoderLayer(
        d_model=network_config.embed_dim,   # the dimension of the embeddings for each input and output token
        nhead=network_config.num_heads,     # the dimension of the embeddings for each head is: d_model // num_heads
        dim_feedforward=network_config.embed_dim * network_config.mlp_ratio,  # ff_dim; the hidden dimension of the feedforward network model
        batch_first=True, 
        )

    decoder = TransformerDecoder(model_config, network_config, decoder_layer, num_classes, seq_len)
    return decoder
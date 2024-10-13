import torch
import torch.nn as nn
from torch import Tensor
from models.encoder import Encoder
from models.decoder import Decoder
from features.embedding import Embeddings
from features.pos_encoding import PositionalEncoding


###########################################################
###########################################################
def make_model(device, src_vocab, trg_vocab, n_layers: int = 3, d_model: int = 512, 
               d_ffn: int = 2048, n_heads: int = 8, dropout: float = 0.1, 
               max_length: int = 5000, src_pad_idx: int = 0, trg_pad_idx: int = 0):
    """
    Construct a model when provided parameters.

    Args:
        src_vocab:    source vocabulary
        trg_vocab:    target vocabulary
        n_layers:     Number of Encoder and Decoders 
        d_model:      dimension of embeddings
        d_ffn:        dimension of feed-forward network
        n_heads:      number of heads
        dropout:      probability of dropout occurring
        max_length:   maximum sequence length for positional encodings

    Returns:
        Transformer model based on hyperparameters
    """

    # create the encoder
    encoder = Encoder(d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ffn=d_ffn, dropout=dropout)
    # create the decoder
    decoder = Decoder(vocab_size=len(trg_vocab), d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ffn=d_ffn, dropout=dropout)

    # create source embedding matrix
    src_embed = Embeddings(vocab_size=len(src_vocab), d_model=d_model)
    # create target embedding matrix
    trg_embed = Embeddings(vocab_size=len(trg_vocab), d_model=d_model)

    # create a positional encoding matrix
    pos_enc = PositionalEncoding(d_model=d_model, dropout=dropout, max_length=max_length)

    # create the Transformer model
    model = Transformer(encoder=encoder, decoder=decoder, src_embed=nn.Sequential(src_embed, pos_enc), 
                        trg_embed=nn.Sequential(trg_embed, pos_enc),
                        src_pad_idx=src_pad_idx, 
                        trg_pad_idx=trg_pad_idx,
                        device=device)

    # initialize parameters with Xavier/Glorot
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

###########################################################
###########################################################
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder,
                src_embed: Embeddings, trg_embed: Embeddings,
                src_pad_idx: int, trg_pad_idx: int, device):
        """
        Args:
            encoder:      encoder stack                    
            decoder:      decoder stack
            src_embed:    source embeddings and encodings
            trg_embed:    target embeddings and encodings
            src_pad_idx:  padding index          
            trg_pad_idx:  padding index
            device:       cuda or cpu

        Returns:
            output:       sequences after decoder           (batch_size, trg_seq_length, vocab_size)
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.device = device
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src: Tensor):
        """
        Args:
            src:          raw sequences with padding        (batch_size, seq_length)              

        Returns:
            src_mask:     mask for each sequence            (batch_size, 1, 1, seq_length)
        """
        # assign 1 to tokens that need attended to and 0 to padding tokens, then add 2 dimensions
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask

    def make_trg_mask(self, trg: Tensor):
        """
        Args:
            trg:          raw sequences with padding        (batch_size, seq_length)              

        Returns:
            trg_mask:     mask for each sequence            (batch_size, 1, seq_length, seq_length)
        """

        seq_length = trg.shape[1]

        # assign True to tokens that need attended to and False to padding tokens, then add 2 dimensions
        trg_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_length)

        # generate subsequent mask
        trg_sub_mask = torch.tril(torch.ones((seq_length, seq_length), device=self.device)).bool() # (batch_size, 1, seq_length, seq_length)

        # bitwise "and" operator | 0 & 0 = 0, 1 & 1 = 1, 1 & 0 = 0
        trg_mask = trg_mask & trg_sub_mask

        return trg_mask

    def forward(self, src: Tensor, trg: Tensor):
        """
        Args:
            trg:          raw target sequences              (batch_size, trg_seq_length)
            src:          raw src sequences                 (batch_size, src_seq_length)

        Returns:
            output:       sequences after decoder           (batch_size, trg_seq_length, vocab_size)
        """

        # create source and target masks     
        src_mask = self.make_src_mask(src=src) # (batch_size, 1, 1, src_seq_length)
        trg_mask = self.make_trg_mask(trg=trg) # (batch_size, 1, trg_seq_length, trg_seq_length)

        # push the src through the encoder layers
        src = self.encoder(src=self.src_embed(src), src_mask=src_mask)  # (batch_size, src_seq_length, d_model)

        # decoder output and attention probabilities
        output = self.decoder(trg=self.trg_embed(trg), src=src, trg_mask=trg_mask, src_mask=src_mask)

        return output
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import pad

###########################################################
###########################################################
def pad_seq(seq: Tensor, max_length: int = 10, pad_idx: int = 0):
    """
    Pads a sequence to the specified maximum length.

    Args:
        seq:                The input sequence to pad.
        max_length          The desired length after padding. Defaults to 10.
        pad_idx:            The value used for padding. Defaults to 0.

    Returns:                The padded sequence.
    """
    pad_to_add = max_length - len(seq) # amount of padding to add
    return pad(seq,(0, pad_to_add), value=pad_idx)

###########################################################
###########################################################
def make_src_mask(src: Tensor, pad_idx: int = 0):
    """
    Args:
        src:          raw sequences with padding        (batch_size, seq_length)              

    Returns:
        src_mask:     mask for each sequence            (batch_size, 1, 1, seq_length)
    """
    # assign 1 to tokens that need attended to, and 0 to padding tokens, then add 2 dimensions
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

    return src_mask

###########################################################
###########################################################
def make_trg_mask(trg: Tensor, pad_idx: int = 0):
    """
    Args:
        trg:          raw sequences with padding        (batch_size, seq_length)              
        
    Returns:
        trg_mask:     mask for each target sequence            (batch_size, 1, seq_length, seq_length)
    """

    seq_length = trg.shape[1]

    # assign True to tokens that need attended to and False to padding tokens, then add 2 dimensions
    trg_mask = (trg != pad_idx).unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_length)

    # generate look-ahead mask
    trg_sub_mask = torch.tril(torch.ones((seq_length, seq_length))).bool() # (batch_size, 1, seq_length, seq_length)

    # bitwise "and" operator | 0 & 0 = 0, 1 & 1 = 1, 1 & 0 = 0
    trg_mask = trg_mask & trg_sub_mask

    return trg_mask

###########################################################
###########################################################
def tokenize(sequence, special_toks=True):
    """
    Tokenizes the input sequence of text.

    Args:
        sequence:           The input string of text to be tokenized.
        special_toks:       Boolean of adding <bos>/<eos> tokens.
    
    Returns:
        list:               A list of lowercase tokens (words).
    """
    # remove punctuation
    for punc in ["!", ".", "?", ","]:
        sequence = sequence.replace(punc, "")

    # split the sequence on spaces and lowercase each token
    sequence = [token.lower() for token in sequence.split(" ")]

    # add beginning and end tokens
    if special_toks:
        sequence = ['<bos>'] + sequence + ['<eos>']

    return sequence

###########################################################
###########################################################
def build_vocab(data: str):
    """
    Builds a vocabulary dictionary from the input data.

    Args:
        data:               The input string of text from which to build the vocabulary.

    Returns:
        dict:               A dictionary where keys are unique tokens and values are their corresponding integer indices.
    
    """
    # tokenize the data and remove duplicates
    vocab = list(set(tokenize(data, special_toks=False)))
    # sort the vocabulary
    vocab.sort()

    # add special tokens
    vocab = ['<pad>', '<bos>', '<eos>'] + vocab
    # assign an integer to each word
    vocab = {word:i for i, word in enumerate(vocab)}

    return vocab

###########################################################
###########################################################
class Embeddings(nn.Module):
    """
    This class creates an embedding look-up table and scales the embeddings by the 
    square root of the embedding dimension.
    """

    def __init__(self, vocab_size: int, d_model: int):
        """
        Args:
          vocab_size:       size of vocabulary
          d_model:          dimension of embeddings
        """
        # inherit from nn.Module
        super().__init__()   

        # embedding look-up table (lut)                          
        self.lut = nn.Embedding(vocab_size, d_model)

        # dimension of embeddings 
        self.d_model = d_model                          

    def forward(self, x: Tensor):
        """
        Args:
          x:                input Tensor (batch_size, seq_length)

        Returns:            embedding vector
        """
        # scaling embeddings 
        return self.lut(x) * math.sqrt(self.d_model)
    
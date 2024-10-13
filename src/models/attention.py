import math
import torch
import torch.nn as nn
from torch import Tensor

seed = 0 

###########################################################
###########################################################
class MultiHeadAttention(nn.Module):
    """
    This class has an option to include bias in the query (Wq), key (Wk), value (Wv), 
    and output (Wo) linear layers by passing the bias argument to nn.Linear. The def-
    ault setting excludes bias (bias=False).
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        """
        Args:
            d_model:      dimension of embeddings
            n_heads:      number of self attention heads
            dropout:      probability of dropout occurring
        """
        super().__init__()
        torch.manual_seed(seed)
        assert d_model % n_heads == 0            # ensure an even num of heads
        self.d_model = d_model                   # dim (e.g. 512)
        self.n_heads = n_heads                   # 8 heads
        self.d_key = d_model // n_heads          # assume d_value equals d_key (e.g. 512/8=64)
        self.bias = bias                         # bias

        self.Wq = nn.Linear(d_model, d_model, bias)    # query weights
        self.Wk = nn.Linear(d_model, d_model, bias)    # key weights
        self.Wv = nn.Linear(d_model, d_model, bias)    # value weights
        self.Wo = nn.Linear(d_model, d_model, bias)    # output weights

        self.dropout = nn.Dropout(p=dropout)     # initialize dropout layer

    def scaled_dot_product_attention(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask=None
    ):
        # (32, 8, 10, 64) x (32, 8, 64, 10) -> (32, 8, 10, 10) = (batch_size, n_heads, q_length, k_length)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_key)

        # fill those positions of product as (-1e10) where mask positions are 0
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # apply softmax
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # multiply by values to obtain the final output
        # (batch_size, n_heads, q_length, k_length) x (batch_size, n_heads, v_length, d_key)
        # (32, 8, 10, 10) x (32, 8, 10, 64) -> (32, 8, 10, 64)
        output = torch.matmul(self.dropout(attn_probs), V)

        # (batch_size, n_heads, q_length, d_key) = (32, 8, 10, 64)
        return output, attn_probs

    def split_heads(
        self, x: Tensor
    ):
        # reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()

        # (batch_size, seq_length, d_model) -> (batch_size, seq_length, n_heads, d_key)
        # (32, 10, 512) -> (32, 10, 8, 64)
        x = x.view(batch_size, seq_length, self.n_heads, self.d_key)

        # (32, 10, 8, 64) -> (32, 8, 10, 64) = (batch_size, n_heads, seq_length, d_key)
        return x.transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor = None
    ):
        """
        Args:
           query:         query vector         (batch_size, q_length, d_model)
           key:           key vector           (batch_size, k_length, d_model)
           value:         value vector         (batch_size, s_length, d_model)
           mask:          mask for decoder

        Returns:
           output:        attention values     (batch_size, q_length, d_model)
           attn_probs:    softmax scores       (batch_size, n_heads, q_length, k_length)
        """
        batch_size = key.size(0)

        # calculate query, key, and value tensors
        Q = self.Wq(query)                       # (32, 10, 512) x (512, 512) = (32, 10, 512)
        K = self.Wk(key)                         # (32, 10, 512) x (512, 512) = (32, 10, 512)
        V = self.Wv(value)                       # (32, 10, 512) x (512, 512) = (32, 10, 512)

        # split each tensor into n-heads to compute attention
        # (32, 10, 512) -> (32, 10, 8, 64) -> (32, 8, 10, 64) = (batch_size, n_heads, seq_length, d_key)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # computes attention
        # scaled dot product -> QK^{T} / sqrt(d_key)
        # (32, 8, 10, 64) x (32, 8, 64, 10) x (32, 8, 10, 64) -> (32, 8, 10, 64) = (batch_size, n_heads, v_length, d_key)
        attn_output, attn_probs = self.scaled_dot_product_attention(Q, K, V, mask)

        # combine the multiple heads
        # (32, 8, 10, 64) -> (32, 10, 8, 64)
        attn_output = attn_output.transpose(1, 2).contiguous()
        # reshape attention back to (32, 10, 512)
        # (32, 10, 8, 64) -> (32, 10, 512) = (batch_size, q_length, d_model)
        attn_output = attn_output.view(batch_size, -1, self.n_heads * self.d_key)

        # push through the final weight layer
        # (32, 10, 512) x (512, 512) = (32, 10, 512)
        output = self.Wo(attn_output)

        return output, attn_probs 
    
###########################################################
###########################################################
class MultiHeadAttention2(nn.Module):
    """
    This class use a bias term in the linear transformations. All linear layers (Wq, 
    Wk, Wv, Wo) are initialized without a bias option.
    """
    
    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.0):
        """
        Args:
            d_model:      dimension of embeddings
            n_heads:      number of self attention heads
            dropout:      probability of dropout occurring
        """
        super().__init__()
        torch.manual_seed(seed)
        assert d_model % n_heads == 0            # ensure an even num of heads
        self.d_model = d_model                   # 512 dim
        self.n_heads = n_heads                   # 8 heads
        self.d_key = d_model // n_heads          # assume d_value equals d_key | 512/8=64

        self.Wq = nn.Linear(d_model, d_model)    # query weights
        self.Wk = nn.Linear(d_model, d_model)    # key weights
        self.Wv = nn.Linear(d_model, d_model)    # value weights
        self.Wo = nn.Linear(d_model, d_model)    # output weights

        self.dropout = nn.Dropout(p=dropout)     # initialize dropout layer  

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None):
        """
        Args:
           query:         query vector         (batch_size, q_length, d_model)
           key:           key vector           (batch_size, k_length, d_model)
           value:         value vector         (batch_size, s_length, d_model)
           mask:          mask for decoder     

        Returns:
           output:        attention values     (batch_size, q_length, d_model)
           attn_probs:    softmax scores       (batch_size, n_heads, q_length, k_length)
        """
        batch_size = key.size(0)                  

        # calculate query, key, and value tensors
        Q = self.Wq(query)                       # (32, 10, 512) x (512, 512) = (32, 10, 512)
        K = self.Wk(key)                         # (32, 10, 512) x (512, 512) = (32, 10, 512)
        V = self.Wv(value)                       # (32, 10, 512) x (512, 512) = (32, 10, 512)

        # split each tensor into n-heads to compute attention

        # query tensor
        Q = Q.view(batch_size,                   # (32, 10, 512) -> (32, 10, 8, 64) 
                   -1,                           # -1 = q_length
                   self.n_heads,              
                   self.d_key
                   ).permute(0, 2, 1, 3)         # (32, 10, 8, 64) -> (32, 8, 10, 64) = (batch_size, n_heads, q_length, d_key)
        # key tensor
        K = K.view(batch_size,                   # (32, 10, 512) -> (32, 10, 8, 64) 
                   -1,                           # -1 = k_length
                   self.n_heads,              
                   self.d_key
                   ).permute(0, 2, 1, 3)         # (32, 10, 8, 64) -> (32, 8, 10, 64) = (batch_size, n_heads, k_length, d_key)
        # value tensor
        V = V.view(batch_size,                   # (32, 10, 512) -> (32, 10, 8, 64) 
                   -1,                           # -1 = v_length
                   self.n_heads, 
                   self.d_key
                   ).permute(0, 2, 1, 3)         # (32, 10, 8, 64) -> (32, 8, 10, 64) = (batch_size, n_heads, v_length, d_key)

        # computes attention
        # scaled dot product -> QK^{T}
        # (32, 8, 10, 64) x (32, 8, 64, 10) -> (32, 8, 10, 10) = (batch_size, n_heads, q_length, k_length)
        scaled_dot_prod = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.d_key)      # sqrt(64)

        # fill those positions of product as (-1e10) where mask positions are 0
        if mask is not None:
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask == 0, -1e10)

        # apply softmax 
        attn_probs = torch.softmax(scaled_dot_prod, dim=-1)

        # multiply by values to get attention
        A = torch.matmul(self.dropout(attn_probs), V)       # (32, 8, 10, 10) x (32, 8, 10, 64) -> (32, 8, 10, 64)
                                                            # (batch_size, n_heads, q_length, k_length) x (batch_size, n_heads, v_length, d_key) -> (batch_size, n_heads, q_length, d_key)

        # reshape attention back to (32, 10, 512)
        A = A.permute(0, 2, 1, 3).contiguous()              # (32, 8, 10, 64) -> (32, 10, 8, 64)
        A = A.view(batch_size, -1, self.n_heads * self.d_key) # (32, 10, 8, 64) -> (32, 10, 8*64) -> (32, 10, 512) = (batch_size, q_length, d_model)

        # push through the final weight layer
        output = self.Wo(A)                                 # (32, 10, 512) x (512, 512) = (32, 10, 512) 

        return output, attn_probs                           # return attn_probs for visualization of the scores
    

from typing import Self
import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.module) :
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


# What is positiona; Encoding ?
# Original Sentence : "Your cat is a lovely cat"
# Embedding (Vector of size 512)
# Position Embedding (Vector of size 512): Only computed once and reused for every sentence during training and inference.
# seq_len is a maximum length of the sentence because we need to create one vector to each position
# dropout to make model less overfit
# d_model represents the total feature dimensionality of the Transformer model, which dictates the size of the embedding vectors passed through its layers.
class PositionalEncoding(nn.module) :
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.dropout(dropout)
        
    
    # create a atrix of shape (seq_len, d_model) for each position
    # compute the formula for positional; encoding: pe(pos, 2i) = sin(pos / 10000^(2i / d_model)) and pe(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))
    # save the positional encoding in a buffer and return it

    # torch.zeros is a PyTorch function that returns a tensor filled entirely with the scalar value 0.  
    pe = torch.zeros(seq_len, d_model)

    # torch.arange is a PyTorch function that returns a tensor filled with evenly spaced values within a given interval.
    # create a vector of shape (seq_len, 1)
    # unsqueeze is a tensor operation that inserts a new dimension of size 1 at a specified position (axis). 
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

    # torch.exp is a PyTorch function that returns a tensor with the exponential of each element in the input tensor.
    # torch.arange is a PyTorch function that returns a tensor filled with evenly spaced values within a given interval.
    # torch.div is a PyTorch function that returns the element-wise division of the two input tensors.
    # torch.mul is a PyTorch function that returns the element-wise multiplication of the two input tensors.
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    # Apply the sin to even positions
    pe[:, 0::2] = torch.sin(position * div_term)

    # Apply the cos to odd positions
    pe[:, 1::2] = torch.cos(position * div_term)

    # Add a batch dimension
    pe = pe.unsqueeze(0)
    
    # Register the positional encoding as a buffer
    self.register_buffer('pe', pe)

    # Forward pass - add positional encoding to input embeddings
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(false)
        return self.dropout(x)
        
        


# We also introduce two parameters, usually called gamma(multiplicative) and beta(additive) that
# introduce some fluctuations int he data, because maybe having all values between 0 and 1 may be
# too restrictive for the network will learn to tune these two parameters to introduce fluctuations
# when necessary.
class LayerNormalization(nn.module) :
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # Multiplicative
        self.bias = nn.Parameter(torch.zeros(1))  # Additive
        
    # Create a learnable parameter for affine transformation (y = gamma * x + beta)
    def forward(self, x):
        # Normalize over the feature dimension (last dimension)
        mean = x.mean(dim = -1, keepdim = True) 
        std = x.std(dim = -1, keepdim = True)   

        # eps is added for numerical stability (to avoid division by zero)
        # Apply Layer Normalization formula: (x - mean) / (std + eps)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias  


class FeedForwardBlock(nn.module) :
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

        
    # in the forward pass we apply the following function: FNN(x) = max(0, x W1 + B1) W2 + B2
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.module) :
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # wq
        self.w_k = nn.Linear(d_model, d_model) # wk
        self.w_v = nn.Linear(d_model, d_model) # wv

        self.w_o = nn.Linear(d_model, d_model) # wo
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask) :
        query = self.w_q(q) # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        key = self.w_k(k) # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        value = self.w_v(v) # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)

        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, h, d_k) --> (Batch, h, Seq_Len, d_k)
        query = query.view(query.shape[1], query.shape[1], self.h, self.d_k).transpose(1, 2)
        
        

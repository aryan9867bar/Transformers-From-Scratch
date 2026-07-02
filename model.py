from typing import Self
import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.module) {
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
class PositionalEncoding(nn.module) {
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
        
        
}
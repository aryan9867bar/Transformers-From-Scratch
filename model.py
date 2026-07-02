from typing import Self
import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module) :
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
class PositionalEncoding(nn.Module) :
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
class LayerNormalization(nn.Module) :
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


class FeedForwardBlock(nn.Module) :
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

        
    # in the forward pass we apply the following function: FNN(x) = max(0, x W1 + B1) W2 + B2
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module) :
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

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout) :
        d_k = query.shape[-1]
        # (Batch, h, Seq_Len, d_k) * (Batch, h, d_k, Seq_Len) --> (Batch, h, Seq_Len, Seq_Len)
        # @ means matrix multiplication
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h, seq_len, seq_len)

        if dropout is not None :
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores # for visualization

    def forward(self, q, k, v, mask) :
        query = self.w_q(q) # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        key = self.w_k(k) # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        value = self.w_v(v) # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)

        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, h, d_k) --> (Batch, h, Seq_Len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        # (Batch, h, Seq_Len, d_k) --> (Batch, Seq_Len, h, d_k) --> (Batch, Seq_Len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module) :
    def __init__(self, dropout: float) -> None :
        super().__init__()
        self.dropout = nn.dropout(dropout)
        self.norm = LayerNormalization()
    
    def forward(self, x, sublayer) :
        # (x + dropout(sublayer(x))) 
        return x + self.dropout(sublayer(self.norm(x))) 
        
class EncoderBlock(nn.Module) : 
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None :
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    # src_mask is the mask that we apply to the input of the encoder
    def forward(self, x, src_mask) -> torch.Tensor:
        # Apply self-attention block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        
        # Apply feed-forward block
        x = self.residual_connections[1](x, self.feed_forward_block)
        
        return x


class Encoder(nn.Module) :
    def __init__(self, layers: nn.ModuleList) -> None :
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)   
        

# In MHA -> Query from decoder and key and value from encoder called cross attention.nn





class DecoderBlock(nn.Module) :
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None :
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.Module([ResidualConnection(dropout) for _ in range(3)])

    # src_mask is the mask that we apply to the input of the encoder
    # There are two mask: one come from encoder(src_mask) and another come from decoder(tgt_mask)
    def forward(self, x, encoder_output, src_mask, tgt_mask) :
        # Apply self-attention block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        
        # Apply cross-attention block
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        
        # Apply feed-forward block
        x = self.residual_connections[2](x, self.feed_forward_block)
        
        return x



class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x) :
        # (Batch, Seq_len, d_model) --> (Batch, Seq_len, vocab_size)
        return self.projection(x)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask) :
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask) :
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x) :
        return self.projection_layer(x)




def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer



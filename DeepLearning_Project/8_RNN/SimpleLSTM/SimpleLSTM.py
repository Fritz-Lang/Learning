import torch
from torch import nn

class SimpleGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
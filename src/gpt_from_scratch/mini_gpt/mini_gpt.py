import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
import math,copy,re
import pandas as pd
import numpy as np
import seaborn as sns


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention module for decoder-only transformer
    """
    def __init__(self, embedding_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        
        assert embedding_dim % n_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        
        self.register_buffer("mask", None)
    
    
    def forward(self, x):
        B, T, C = x.size()  

        if self.mask is None or self.mask.size(0) != T:
            mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
            self.mask = mask
        
        q = self.query(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  
        k = self.key(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
       
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim) 
        scores = scores.masked_fill(self.mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)  
        attn_weights = self.dropout(attn_weights)
    
        out = attn_weights @ v  
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)  
        out = self.proj(out)
        
        return out


class FeedForward(nn.Module):
    """
    Simple feed-forward network with GELU activation
    """

    def __init__(self, embedding_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * embedding_dim

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x



class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, n_layers=4, n_heads=4, dropout=0.1):
        super().__init__()
        pass
        
    
    def get_embeddings(self, tokens):
        pass

    def forward(self):
        pass



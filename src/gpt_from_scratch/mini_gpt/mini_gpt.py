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



class TransformerBlock(nn.Module):
    """
    Transformer block with pre-layer normalization
    """

    def __init__(self, embedding_dim, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attn = CausalSelfAttention(embedding_dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ffn = FeedForward(embedding_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x



class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, n_layers=4, n_heads=4, dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(embedding_dim)
        self.output_proj = nn.Linear(embedding_dim, vocab_size, bias=False)
    

    def get_embeddings(self, tokens):
        """Get token and positional embeddings"""
        B, T = tokens.size()
        assert T <= self.context_size, f"Sequence length {T} exceeds context size {self.context_size}"
        
        # Token embeddings
        token_embeds = self.token_embedding(tokens)
        
        # Positional embeddings
        positions = torch.arange(0, T, dtype=torch.long, device=tokens.device).unsqueeze(0).expand(B, T)
        pos_embeds = self.position_embedding(positions)
        
        # Combine and apply dropout
        x = token_embeds + pos_embeds
        x = self.dropout(x)
        return x


    def forward(self, tokens):
        """Forward pass through the model"""
        # Get embeddings
        x = self.get_embeddings(tokens)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.output_proj(x)
        
        return logits

    @torch.no_grad()
    def generate(self, prefix_tokens, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text using the model
        Args:
            prefix_tokens: torch.LongTensor of shape (batch_size, prefix_len)
            max_new_tokens: int, maximum number of new tokens to generate
            temperature: float, controls randomness (lower = more deterministic)
            top_k: int or None, if specified, only sample from top k most likely tokens
        Returns:
            tokens: torch.LongTensor of shape (batch_size, prefix_len + max_new_tokens)
        """
        tokens = prefix_tokens.clone()
        
        for _ in range(max_new_tokens):
            if tokens.size(1) > self.context_size:
                tokens_cond = tokens[:, -self.context_size:]
            else:
                tokens_cond = tokens
            
            logits = self(tokens_cond)
            logits = logits[:, -1, :] / temperature  
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
                
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            tokens = torch.cat([tokens, next_token], dim=1)
            
        return tokens


    def calculate_perplexity(self, tokens, labels=None):
        """
        Calculate perplexity on a batch of text
        Args:
            tokens: torch.LongTensor of shape (batch_size, seq_len)
            labels: Optional target labels (if None, shifts tokens to right)
        Returns:
            perplexity: torch.Tensor scalar
        """
        if labels is None:
            labels = tokens[:, 1:].contiguous()
            tokens = tokens[:, :-1].contiguous()
            
        logits = self(tokens)
        
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))
        
        perplexity = torch.exp(loss)
        
        return perplexity



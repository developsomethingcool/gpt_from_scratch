"""
Neural n-gram Model (PyTorch)
-----------------------------
This module defines a simple neural n-gram model using PyTorch.
The model predicts the next token given a context of (n-1) tokens.

Architecture:
    - Embedding layer: maps token IDs to dense vectors
    - Flatten: concatenates embeddings for all context tokens
    - MLP: one or more linear layers with GELU activation
    - Output: logits for next-token prediction (softmax handled by loss)
"""

import torch
import torch.nn as nn

class NeuralNGramModel(nn.Module):
    def __init__(self, vocab_size, context_size, emb_dim=128, hidden_dim=256):
        """
        Args:
            vocab_size (int): Number of tokens in the vocabulary.
            context_size (int): Number of context tokens (n-1 for n-gram).
            emb_dim (int): Size of each embedding vector.
            hidden_dim (int): Size of the hidden layer in the MLP.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(context_size * emb_dim, hidden_dim),
            nn.GELU(),  # GELU is smoother than ReLU, often used in transformers
            nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): shape (batch_size, context_size), token IDs for context

        Returns:
            logits (Tensor): shape (batch_size, vocab_size), unnormalized scores for next token
        """
        # Embed each token in the context
        emb = self.embedding(x)  # (batch_size, context_size, emb_dim)
        # Flatten context embeddings into a single vector per example
        emb = emb.view(emb.size(0), -1)  # (batch_size, context_size * emb_dim)
        # Pass through MLP to get logits for next token
        logits = self.mlp(emb)  # (batch_size, vocab_size)
        return logits

if __name__ == "__main__":
    # Example usage and sanity check
    vocab_size = 1000
    context_size = 3  # For trigram model
    model = NeuralNGramModel(vocab_size, context_size)
    x = torch.randint(0, vocab_size, (4, context_size))  # batch of 4
    logits = model(x)
    print("Logits shape:", logits.shape)  # Should be (4, vocab_size)
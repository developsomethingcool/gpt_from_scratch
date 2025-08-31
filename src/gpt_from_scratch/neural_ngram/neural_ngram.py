import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNGramModel(nn.Module):
    def __init__(self, vocab_size, context_size, embedding_dim=128, hidden_dim=256, dropout=0.1):
        """
        Args:
            vocab_size (int): Number of tokens in the vocabulary.
            context_size (int): Number of context tokens (n-1 for n-gram).
            embedding_dim (int): Size of each embedding vector.
            hidden_dim (int): Size of the hidden layer in the MLP.
            dropout (float): Dropout probability for regularization.
        """
        super().__init__()
        # required for the trainer's loss calculation
        self.vocab_size = vocab_size  
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(context_size * embedding_dim, hidden_dim),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
    
    def forward(self, x):
        """
        Args:
            x (Tensor): shape (batch_size, context_size), token IDs for context

        Returns:
            logits (Tensor): shape (batch_size, vocab_size), unnormalized scores for next token
        """
        # embed each token in the context
        emb = self.embedding(x)  # (batch_size, context_size, emb_dim)
        # flatten context embeddings into a single vector per example
        emb = emb.view(emb.size(0), -1)  # (batch_size, context_size * emb_dim)
        emb = self.dropout(emb)
        # pass through MLP to get logits for next token
        logits = self.mlp(emb)  # (batch_size, vocab_size)
        return logits
    
    def generate(self, input_ids, max_new_tokens=30, temperature=0.8, top_k=50):
        """Generate text from the model.
        
        Args:
            input_ids (Tensor): Starting token IDs with shape (1, seq_len)
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Temperature for sampling (higher = more random)
            top_k (int): Number of highest probability tokens to consider in sampling
            
        Returns:
            Tensor: Generated token IDs
        """
        self.eval()
        context_size = input_ids.size(1)
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # take the last context_size tokens as input
                inputs = generated[:, -context_size:] 
                
                # ff we don't have enough context yet, pad with zeros
                if inputs.size(1) < context_size:
                    pad_size = context_size - inputs.size(1)
                    padding = torch.zeros((1, pad_size), dtype=torch.long, device=inputs.device)
                    inputs = torch.cat([padding, inputs], dim=1)
                
                # get logits and apply temperature
                logits = self(inputs)
                logits = logits / temperature
                
                # apply top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # sample from the distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
        return generated
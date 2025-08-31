"""
Training loop for Neural n-gram Model
-------------------------------------
- Uses teacher forcing (always feeds gold context).
- Implements early stopping: stops if validation loss doesn't improve for N epochs.
- Saves the best model checkpoint to models/neural_ngram/checkpoint.pt.
Optimized with multiprocessing for parallel tokenization and efficient file reading.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from multiprocessing import Pool
from models.neural_ngram.model import NeuralNGramModel
from models.neural_ngram.utils import make_ngram_batches
from models.neural_ngram.tokenizer.tokenizer_bpe import BPETokenizer

# Load tokenizer globally for multiprocessing
tokenizer = BPETokenizer.load("models/neural_ngram/tokenizer", "bpe_0500")

def tokenize_line(line):
    """
    Tokenizes a single line using the BPE tokenizer.
    """
    return tokenizer.encode(line.strip(), add_bos=False, add_eos=True).ids

def get_dataloader(split, block_size=128, batch_size=32):
    """
    Optimized dataloader with multiprocessing for tokenization.

    Args:
        split (str): 'train', 'valid', or 'test'.
        block_size (int): Sequence length for each batch.
        batch_size (int): Number of sequences per batch.

    Yields:
        Tensor: Batch of tokenized sequences (batch_size, block_size).
    """
    file_path = f"models/neural_ngram/shakespeare_dirty_{split}.txt"
    print(f"Reading data from: {file_path}")  # Debugging print

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        return None

    # Tokenize lines in parallel using multiprocessing
    print("Tokenizing lines in parallel...")
    with Pool() as pool:
        tokenized_lines = pool.map(tokenize_line, lines)

    # Create batches
    batches = []
    for i in range(0, len(tokenized_lines), batch_size):
        batch = tokenized_lines[i:i + batch_size]
        # Pad sequences to block_size
        padded_batch = [seq[:block_size] + [0] * (block_size - len(seq)) for seq in batch]
        batches.append(torch.tensor(padded_batch))

    print(f"Generated {len(batches)} batches for {split}")  # Debugging print
    for batch in batches:
        yield batch

def train_ngram(
    vocab_size, context_size, train_loader, val_loader,
    emb_dim=128, hidden_dim=256, lr=1e-3, epochs=50, patience=3,
    checkpoint_path="models/neural_ngram/checkpoint.pt"
):
    """
    Trains the neural n-gram model with early stopping and checkpointing.

    Args:
        vocab_size (int): Size of the vocabulary.
        context_size (int): Number of context tokens (n-1).
        train_loader: DataLoader for training data (generator).
        val_loader: DataLoader for validation data (generator).
        emb_dim (int): Embedding dimension.
        hidden_dim (int): Hidden layer size.
        lr (float): Learning rate.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        checkpoint_path (str): Path to save the best model.
    """
    model = NeuralNGramModel(vocab_size, context_size, emb_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        for batch in train_loader:
            contexts, targets = make_ngram_batches(batch, context_size + 1)
            optimizer.zero_grad()
            logits = model(contexts)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_steps += 1
        train_loss = train_loss / train_steps if train_steps > 0 else 0

        # Validation phase
        model.eval()
        val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                contexts, targets = make_ngram_batches(batch, context_size + 1)
                logits = model(contexts)
                loss = criterion(logits, targets)
                val_loss += loss.item()
                val_steps += 1
        val_loss = val_loss / val_steps if val_steps > 0 else 0

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Best validation loss: {best_val_loss:.4f}")
    return model

if __name__ == "__main__":
    vocab_size = 5000  # Set to your actual vocab size if known
    context_size = 3   # For trigram (n=3)
    train_loader = get_dataloader('train', block_size=128, batch_size=32)
    val_loader = get_dataloader('valid', block_size=128, batch_size=32)
    train_ngram(vocab_size, context_size, train_loader, val_loader)
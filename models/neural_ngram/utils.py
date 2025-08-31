"""
Dataloader utilities for Neural n-gram Model
--------------------------------------------
This module provides a wrapper to convert a stream of BPE tokens into
(context, next_token) pairs for n-gram modeling.

Uses a trained BPE tokenizer to convert text into token IDs.
"""

import torch
from models.neural_ngram.tokenizer.tokenizer_bpe import BPETokenizer

# Load your trained BPE tokenizer
tokenizer = BPETokenizer.load("models/neural_ngram/tokenizer", "bpe_0500")

def get_dataloader(split, block_size=128, batch_size=32):
    """
    Reads data from the specified split and generates batches.

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

    # Tokenize lines using the BPE tokenizer
    tokenized_lines = []
    for line in lines:
        encoded = tokenizer.encode(line.strip(), add_bos=False, add_eos=True)  # Tokenize the line into token IDs
        tokenized_lines.append(encoded)

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

def make_ngram_batches(token_batch, n):
    """
    Converts a batch of token sequences into (context, next_token) pairs.

    Args:
        token_batch (Tensor): shape (batch_size, seq_len)
        n (int): n-gram size (context size = n-1)

    Returns:
        contexts (Tensor): shape (num_samples, n-1)
        targets (Tensor): shape (num_samples,)
    """
    contexts = []
    targets = []
    for seq in token_batch:
        for i in range(len(seq) - n + 1):
            contexts.append(seq[i:i + n - 1])  # Extract context (n-1 tokens)
            targets.append(seq[i + n - 1])    # Extract next token (target)

    contexts = torch.stack(contexts)
    targets = torch.tensor(targets)
    print("Contexts:\n", contexts)
    print("Targets:\n", targets)
    print("Contexts shape:", contexts.shape)
    print("Targets shape:", targets.shape)
    return contexts, targets
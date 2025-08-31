"""
Neural n-gram Training Loop
---------------------------
- Teacher forcing (feeds gold context).
- Early stopping with patience.
- Saves best checkpoint at models/neural_ngram/checkpoint.pt
- Multiprocessing for tokenization efficiency.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from multiprocessing import Pool
from models.neural_ngram.model import NeuralNGramModel
from models.neural_ngram.tokenizer.tokenizer_bpe import BPETokenizer

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer globally
tokenizer = BPETokenizer.load("models/neural_ngram/tokenizer", "bpe_0500")

def tokenize_line(line):
    """Tokenizes a single line using the BPE tokenizer."""
    return tokenizer.encode(line.strip(), add_bos=False, add_eos=True)

def get_dataloader(split, block_size=128, batch_size=32):
    """Yield batches of tokenized sequences."""
    file_path = f"models/neural_ngram/shakespeare_dirty_{split}.txt"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist!")
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Tokenize in parallel
    print(f"[{split}] Tokenizing {len(lines)} lines...")
    with Pool() as pool:
        tokenized_lines = pool.map(tokenize_line, lines)

    # Create padded batches
    batches = []
    for i in range(0, len(tokenized_lines), batch_size):
        batch = tokenized_lines[i:i + batch_size]
        padded_batch = [seq[:block_size] + [0]*(block_size - len(seq)) for seq in batch]
        batches.append(torch.tensor(padded_batch, dtype=torch.long))
    print(f"[{split}] Generated {len(batches)} batches.")
    for batch in batches:
        yield batch.to(device)

def make_ngram_batches(batch, n):
    """
    Converts a batch of sequences into contexts and targets for n-grams.
    Args:
        batch: (batch_size, seq_len)
        n: n-gram size
    Returns:
        contexts: (num_ngrams, n-1)
        targets: (num_ngrams,)
    """
    contexts, targets = [], []
    for seq in batch:
        seq = seq.tolist() if torch.is_tensor(seq) else seq
        for i in range(len(seq) - n + 1):
            contexts.append(seq[i:i+n-1])
            targets.append(seq[i+n-1])
    if len(targets) == 0:
        return torch.empty(0, n-1, dtype=torch.long), torch.empty(0, dtype=torch.long)
    return torch.tensor(contexts, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

def train_ngram(
    vocab_size, context_size, train_loader, val_loader,
    emb_dim=128, hidden_dim=256, lr=1e-3, epochs=50, patience=3,
    checkpoint_path="models/neural_ngram/checkpoint.pt"
):
    """Train the Neural N-gram model with early stopping."""
    model = NeuralNGramModel(vocab_size, context_size, emb_dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # ----- Training -----
        model.train()
        total_train_loss = 0
        train_batches = 0

        for batch in train_loader:
            contexts, targets = make_ngram_batches(batch, context_size + 1)
            if targets.numel() == 0:
                continue  # skip empty batches

            contexts = contexts.to(device)
            targets = targets.to(device).long()

            optimizer.zero_grad()
            logits = model(contexts)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1

        avg_train_loss = total_train_loss / max(train_batches, 1)

        # ----- Validation -----
        model.eval()
        total_val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                contexts, targets = make_ngram_batches(batch, context_size + 1)
                if targets.numel() == 0:
                    continue

                contexts = contexts.to(device)
                targets = targets.to(device).long()

                logits = model(contexts)
                loss = criterion(logits, targets)
                total_val_loss += loss.item()
                val_batches += 1

        avg_val_loss = total_val_loss / max(val_batches, 1)

        print(f"Epoch {epoch:02d}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        # ----- Early Stopping -----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  New best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    return model

if __name__ == "__main__":
    vocab_size = 5000
    context_size = 3  # trigram
    train_loader = get_dataloader('train', block_size=128, batch_size=32)
    val_loader = get_dataloader('valid', block_size=128, batch_size=32)

    model = train_ngram(vocab_size, context_size, train_loader, val_loader)



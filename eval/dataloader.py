import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from pathlib import Path

class LM_Dataset(Dataset):
    """Dataset for language modeling using BPE tokenizer."""
    def __init__(self, split, tokenizer_dir="tokenizer/bpe_5000", block_size=128):
        
        self.file_path = Path(f"data/processed/{split}.txt")
        if not self.file_path.exists():
            raise FileNotFoundError(f"{self.file_path} not found! Run build_splits.py first.")

        self.block_size = block_size

        # Load frozen BPE tokenizer
        self.tokenizer = Tokenizer.from_file(f"{tokenizer_dir}/tokenizer.json")

        # Read and tokenize text
        text = self.file_path.read_text(encoding="utf-8")
        self.tokens = self.tokenizer.encode(text).ids
        print(f" Loaded {split} split, {len(self.tokens)} tokens")

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx+self.block_size], dtype=torch.long)
        y = torch.tensor(self.tokens[idx+1:idx+self.block_size+1], dtype=torch.long)
        return x, y

def get_dataloader(split, block_size=128, batch_size=32, tokenizer_dir="tokenizer/bpe_5000", device="cpu", seed=42):
    """Return a PyTorch DataLoader for a given split."""
    torch.manual_seed(seed)
    dataset = LM_Dataset(split, tokenizer_dir=tokenizer_dir, block_size=block_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=(device in ["cuda", "mps"]))

#!/usr/bin/env python3
#!/usr/bin/env python3
import re, unicodedata, hashlib
from pathlib import Path
import numpy as np

RAW_PATH = Path("data/raw/shakespeare.txt")
OUT_DIR = Path("data/processed")

def normalize(text: str) -> str:
    """Normalize text: lowercase + NFKC unicode normalization."""
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    # Keep only basic chars (letters, numbers, punctuation, spaces, newlines)
    text = re.sub(r"[^a-z0-9\s\.,;:!?'\-\"\(\)\n]", " ", text)
    text = re.sub(r"\s+", " ", text)  # collapse spaces
    return text.strip()

def main():
    print("Starting build_splits.py...")

    # Check if raw file exists
    if not RAW_PATH.exists():
        print(f"Error: Raw dataset not found at {RAW_PATH}")
        print(" Make sure your file is inside data/raw/ and update RAW_PATH if needed.")
        return

    # Read raw file
    raw_text = RAW_PATH.read_text(encoding="utf-8")
    print(f"📖 Read {len(raw_text)} characters from {RAW_PATH}")

    # Normalize
    norm_text = normalize(raw_text)
    print(f"✨ Normalized text length: {len(norm_text)} characters")

    # Shuffle deterministically
    np.random.seed(42)
    chars = np.array(list(norm_text))
    n = len(chars)
    idx = np.arange(n)
    np.random.shuffle(idx)

    # Create splits
    train_idx = idx[:int(0.8*n)]
    val_idx   = idx[int(0.8*n):int(0.9*n)]
    test_idx  = idx[int(0.9*n):]

    splits = {
        "train": "".join(chars[train_idx]),
        "val":   "".join(chars[val_idx]),
        "test":  "".join(chars[test_idx]),
    }

    # Save splits & hashes
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, text in splits.items():
        out_file = OUT_DIR / f"{name}.txt"
        out_file.write_text(text, encoding="utf-8")
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        hashes[name] = sha
        print(f"💾 Saved {name}.txt ({len(text)} chars) | SHA256: {sha[:12]}...")

    # Write SPLIT.md
    with open("data/SPLIT.md", "w") as f:
        f.write("# Data Splits\n\n")
        f.write("Deterministic split with seed=42. Hashes:\n\n")
        for k, v in hashes.items():
            f.write(f"- {k}: `{v}`\n")

    print("Done! Splits are in data/processed/ and documented in data/SPLIT.md")

if __name__ == "__main__":
    main()


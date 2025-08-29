import csv
import math
import os
from pathlib import Path
from typing import List, Dict, Tuple

from tokenizer_bpe import BPETokenizer

DATA_DIR = Path("data/processed")
TRAIN_PATH = DATA_DIR / "train.txt"
VAL_PATH   = DATA_DIR / "val.txt"

TOKENIZER_DIR = Path("tokenizer/tokenizers")
LOGS_DIR      = Path("tokenizer/logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Config ----------
MERGES_GRID = [200, 500, 1000, 2000, 5000]  # sweep; pick top-3 with lowest val PPL
NGRAM_N     = 3                             # trigram LM
DELTA       = 1.0                           # Laplace smoothing
SEED        = 42

# ---------- Helpers ----------
def read_lines(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    # Preserve newlines as natural boundaries; strip trailing whitespace to keep things clean
    return [ln.rstrip("\n") for ln in text.splitlines()]

def build_stream(tokenizer: BPETokenizer, lines: List[str]) -> List[int]:
    """Encode each line, append EOS, and concatenate to a single token stream."""
    ids: List[int] = []
    for ln in lines:
        ids += tokenizer.encode(ln, add_bos=False, add_eos=True)
    return ids
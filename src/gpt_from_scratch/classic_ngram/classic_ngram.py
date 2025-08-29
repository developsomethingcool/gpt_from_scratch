import math
import os
from pathlib import Path
from typing import List, Dict, Tuple
import csv
import sys

project_root = Path(__file__).parent.parent.parent.parent  # Navigate to gpt_from_scratch/
sys.path.insert(0, str(project_root / "src"))

from gpt_from_scratch.tokenizer.tokenizer_bpe import BPETokenizer

# ---------- Paths ----------
DATA_DIR = Path("data/processed")
TRAIN_PATH = DATA_DIR / "train.txt"
VAL_PATH   = DATA_DIR / "val.txt"

TOKENIZER_DIR = Path("tokenizers")
LOGS_DIR      = Path("logs")
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

class NGramLM:
    """Classic n-gram with Laplace (add-delta) smoothing, token-level evaluation."""
 
    def __init__(self, n: int, delta: float, vocab_size: int):
        assert n >= 1
        self.n = n
        self.delta = delta
        self.vocab_size = vocab_size
        # context -> next_token -> count
        self.counts: Dict[Tuple[int, ...], Dict[int, int]] = {}
        # context -> total count
        self.totals: Dict[Tuple[int, ...], int] = {}

    def update(self, stream: List[int]) -> None:
        n = self.n
        if n == 1:
            # Unigram counts
            ctx = ()
            self.counts.setdefault(ctx, {})
            for t in stream:
                self.counts[ctx][t] = self.counts[ctx].get(t, 0) + 1
            self.totals[ctx] = sum(self.counts[ctx].values())
            return

        # N-gram counts (sliding window)
        for i in range(n - 1, len(stream)):
            ctx = tuple(stream[i - (n - 1):i])   # prev n-1 tokens
            nxt = stream[i]
            bucket = self.counts.setdefault(ctx, {})
            bucket[nxt] = bucket.get(nxt, 0) + 1
            self.totals[ctx] = self.totals.get(ctx, 0) + 1

    def log_prob(self, ctx: Tuple[int, ...], nxt: int) -> float:
        """Laplace smoothing: (c+delta)/(total + delta*V).
           If unseen context, behaves like uniform prior (delta/(delta*V))."""
        V = self.vocab_size
        c_ctx = self.counts.get(ctx)
        if c_ctx is None:
            # unseen context -> uniform after smoothing
            return math.log((self.delta) / (self.delta * V))
        c = c_ctx.get(nxt, 0)
        total = self.totals.get(ctx, 0)
        return math.log((c + self.delta) / (total + self.delta * V))
    
    def perplexity(self, stream: List[int]) -> float:
        n = self.n
        if len(stream) < n:
            return float("inf")
        nll = 0.0
        T = 0
        for i in range(n - 1, len(stream)):
            ctx = tuple(stream[i - (n - 1):i])
            nxt = stream[i]
            nll -= self.log_prob(ctx, nxt)
            T += 1
        return math.exp(nll / max(T, 1))
    
def eval_ngram_ppl(tokenizer: BPETokenizer, n: int, delta: float) -> Tuple[float, int]:
    """Train n-gram on train stream, evaluate PPL on val stream."""
    train_lines = read_lines(TRAIN_PATH)
    val_lines   = read_lines(VAL_PATH)

    train_stream = build_stream(tokenizer, train_lines)
    val_stream   = build_stream(tokenizer, val_lines)

    lm = NGramLM(n=n, delta=delta, vocab_size=len(tokenizer.vocab))
    lm.update(train_stream)
    ppl = lm.perplexity(val_stream)
    return ppl, len(tokenizer.vocab)
    
def sweep_and_select_topK(merges_grid: List[int], top_k: int = 3) -> None:
    results = []

    for merges in merges_grid:
        print(f"\n=== Training tokenizer with merges={merges} ===")
        tok = BPETokenizer(
            seed=SEED,
            bos_token="<BOS>",
            eos_token="<EOS>",
            pad_token=None,
            unk_token=None,
            lowercase=True,
            unicode_normalization="NFKC",
            collapse_whitespace=True,
            keep_newlines=True,
        )

        # Read TRAIN text (train split ONLY) and train tokenizer
        train_text = TRAIN_PATH.read_text(encoding="utf-8")
        tok.train(
            train_text,
            merges=merges,
            guard_train_only=True,
            source_tag="train",
            min_pair_freq=1,
        )

        # Save artifacts per merge
        tag = f"bpe_{merges:04d}"
        TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
        tok.save(str(TOKENIZER_DIR), tag)

        # Quick diagnostics (optional, could log too)
        # diag = tok.diagnostics(read_lines(TRAIN_PATH)[:500])

        # Evaluate token-level val PPL with classic n-gram
        ppl, vocab_size = eval_ngram_ppl(tok, n=NGRAM_N, delta=DELTA)
        print(f"→ merges={merges} | vocab={vocab_size} | val PPL={ppl:.3f}")

        results.append({
            "merges": merges,
            "vocab_size": vocab_size,
            "ngram_n": NGRAM_N,
            "delta": DELTA,
            "val_token_ppl": ppl,
            "tokenizer_tag": tag,
        })

    # Write full sweep log
    ppl_csv = LOGS_DIR / "ngram_ppl.csv"
    with ppl_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for row in results:
            w.writerow(row)
    print(f"\n📄 Wrote sweep results: {ppl_csv}")

    # Select top-K by lowest validation PPL
    results_sorted = sorted(results, key=lambda r: r["val_token_ppl"])
    top = results_sorted[:top_k]
    top_csv = LOGS_DIR / "top_merges_for_gpt.csv"
    with top_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(top[0].keys()))
        w.writeheader()
        for row in top:
            w.writerow(row)
    print(f"🏁 Wrote top-{top_k} merges for GPT: {top_csv}")
    print("Selected:")
    for r in top:
        print(f"  - merges={r['merges']} | vocab={r['vocab_size']} | val PPL={r['val_token_ppl']:.3f} | tag={r['tokenizer_tag']}")

if __name__ == "__main__":
    if not TRAIN_PATH.exists() or not VAL_PATH.exists():
        raise FileNotFoundError("Expected data/processed/train.txt and val.txt. Build your splits first.")
    sweep_and_select_topK(MERGES_GRID, top_k=3)

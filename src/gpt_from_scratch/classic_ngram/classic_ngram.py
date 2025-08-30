import math
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import csv
import sys
import random

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
NGRAM_N     = [1, 2, 3, 4]                             # trigram LM
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
 
    def __init__(self, n: int, delta: float, vocab_size: int, 
                 smoothing_method: str = "laplace",
                 backoff_alpha: float = 0.4,
                 interpolation_lambdas: Optional[List[float]] = None):
        assert n >= 1
        self.n = n
        self.delta = delta
        self.vocab_size = vocab_size
        self.smoothing_method = smoothing_method  # "laplace", "backoff", or "interpolation"
        self.backoff_alpha = backoff_alpha  # For backoff

        # For interpolation: weights for different n-gram orders (default: equal weights)
        if interpolation_lambdas is None:
            self.lambdas = [1.0 / n] * n  # Equal weight to each n-gram order
        else:
            assert len(interpolation_lambdas) == n
            assert abs(sum(interpolation_lambdas) - 1.0) < 1e-10  # Must sum to 1
            self.lambdas = interpolation_lambdas

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
        """Calculate log probability based on selected smoothing method."""
        if self.smoothing_method == "laplace":
            return self._log_prob_laplace(ctx, nxt)
        elif self.smoothing_method == "backoff":
            return self._log_prob_backoff(ctx, nxt)
        elif self.smoothing_method == "interpolation":
            return self._log_prob_interpolation(ctx, nxt)
        else:
            raise ValueError(f"Unknown smoothing method: {self.smoothing_method}")
    
    def _log_prob_laplace(self, ctx: Tuple[int, ...], nxt: int) -> float:
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

    def _log_prob_backoff(self, ctx: Tuple[int, ...], nxt: int) -> float:
        """Simple 'stupid' backoff: if context exists, use it; otherwise back off to
           a lower order with a penalty factor alpha."""
        c_ctx = self.counts.get(ctx)
        if c_ctx is not None and nxt in c_ctx:
            # We've seen this exact context and next token
            return math.log(c_ctx[nxt] / self.totals[ctx])
        
        # Back off to lower order if available
        if len(ctx) > 0:
            # Apply backoff penalty and use shorter context
            return math.log(self.backoff_alpha) + self._log_prob_backoff(ctx[1:], nxt)
        
        # Base case: unigram (or uniform if even that's unseen)
        empty_ctx = ()
        c_ctx = self.counts.get(empty_ctx)
        if c_ctx is not None and nxt in c_ctx:
            return math.log(c_ctx[nxt] / self.totals[empty_ctx])
        
        # Last resort: uniform distribution over vocabulary
        return math.log(1.0 / self.vocab_size)

    def _log_prob_interpolation(self, ctx: Tuple[int, ...], nxt: int) -> float:
        """Jelinek-Mercer interpolation: linearly combine probabilities from all n-gram orders."""
        n = self.n
        prob = 0.0
        
        # Iterate through all context lengths (n-gram orders)
        for i in range(n):
            # Get the appropriate context length for this n-gram order
            order_ctx = ctx[i:] if i < len(ctx) else ()
            
            # Get counts for this context
            c_ctx = self.counts.get(order_ctx)
            if c_ctx is None:
                p = 1.0 / self.vocab_size  # Uniform if context unseen
            else:
                c = c_ctx.get(nxt, 0)
                total = self.totals.get(order_ctx, 0)
                if total == 0:
                    p = 1.0 / self.vocab_size
                else:
                    # Add small smoothing to avoid zeros
                    p = (c + 0.01) / (total + 0.01 * self.vocab_size)
            
            # Add weighted probability
            prob += self.lambdas[i] * p
        
        return math.log(prob)

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
    
    def generate_text(self, 
                      prefix: List[int], 
                      max_length: int = 100, 
                      temperature: float = 1.0,
                      use_argmax: bool = False,
                      eos_token: Optional[int] = None) -> List[int]:
        """Generate text starting from a prefix.
        
        Args:
            prefix: Initial token sequence to start generation from
            max_length: Maximum number of tokens to generate (including prefix)
            temperature: Controls randomness (lower = more deterministic)
            use_argmax: If True, always pick most likely token; otherwise sample
            eos_token: Token ID that signals end of sequence
            
        Returns:
            List of token IDs (including the initial prefix)
        """
        generated = list(prefix)
        context_size = self.n - 1  # Context size for n-gram model
        
        while len(generated) < max_length:
            # Get current context
            if len(generated) < context_size:
                # Not enough context yet, use what we have
                ctx = tuple(generated)
            else:
                # Use the last (n-1) tokens as context
                ctx = tuple(generated[-context_size:])
            
            # Get next token based on context
            next_token = self._predict_next_token(ctx, temperature, use_argmax)
            generated.append(next_token)
            
            # Check for EOS token
            if eos_token is not None and next_token == eos_token:
                break
                
        return generated
    
    def _predict_next_token(self, ctx: Tuple[int, ...], temperature: float, use_argmax: bool) -> int:
        """Predict next token based on context using either argmax or sampling."""
        # Collect probabilities for all possible next tokens
        next_token_logprobs = {}
        
        # For efficiency, only check tokens we've seen after some context
        candidates = set()
        
        # Try contexts of decreasing specificity to find candidate tokens
        curr_ctx = ctx
        while len(curr_ctx) >= 0:
            c_ctx = self.counts.get(curr_ctx, {})
            candidates.update(c_ctx.keys())
            if not curr_ctx:  # Empty context
                break
            curr_ctx = curr_ctx[1:]  # Shorten context
            
        # If no candidates found, use all tokens
        if not candidates:
            candidates = range(self.vocab_size)
            
        # Calculate log probability for each candidate
        for token in candidates:
            next_token_logprobs[token] = self.log_prob(ctx, token)
            
        # Convert to probabilities and apply temperature
        probs = {t: math.exp(lp / temperature) for t, lp in next_token_logprobs.items()}
        
        if use_argmax:
            # Select token with highest probability (greedy)
            return max(probs.items(), key=lambda x: x[1])[0]
        else:
            # Sample from the distribution
            total = sum(probs.values())
            if total <= 0:
                # Fallback if all probabilities are 0
                return random.randint(0, self.vocab_size - 1)
                
            # Normalize
            probs = {t: p/total for t, p in probs.items()}
            
            # Sample
            r = random.random()
            cumsum = 0
            for token, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                cumsum += prob
                if cumsum >= r:
                    return token
                    
            # Fallback (should rarely happen)
            return list(probs.keys())[0]
    
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
    
def sweep_and_select_topK(merges_grid: List[int], n_values_grid: List[int], top_k: int = 3) -> None:
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

        # Evaluate with different n-gram sizes
        for n in n_values_grid:
            print(f"  Evaluating with n={n}...")
            # Evaluate token-level val PPL with classic n-gram
            ppl, vocab_size = eval_ngram_ppl(tok, n=n, delta=DELTA)
            print(f"  → merges={merges} | n={n} | vocab={vocab_size} | val PPL={ppl:.3f}")

            results.append({
                "merges": merges,
                "ngram_n": n,
                "vocab_size": vocab_size, 
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
    print(f"Wrote top-{top_k} merges for GPT: {top_csv}")
    print("Selected:")
    for r in top:
        print(f"  - merges={r['merges']} | vocab={r['vocab_size']} | val PPL={r['val_token_ppl']:.3f} | tag={r['tokenizer_tag']}")


def generate_and_decode(lm: NGramLM, tokenizer: BPETokenizer, seed_text: str, 
                       max_length: int = 100, temperature: float = 0.8,
                       use_argmax: bool = False) -> str:
    """Generate text from a seed string and decode it."""
    # Encode seed text
    prefix = tokenizer.encode(seed_text, add_bos=False, add_eos=False)
    
    # Generate token sequence
    generated_ids = lm.generate_text(
        prefix=prefix,
        max_length=max_length,
        temperature=temperature,
        use_argmax=use_argmax,
        eos_token=tokenizer.eos_id
    )
    
    # Decode to text
    return tokenizer.decode(generated_ids)


def evaluate_generation(merges: int, n: int, smoothing_method: str, seed_texts: List[str]) -> None:
    """Evaluate text generation with different settings."""
    # Load tokenizer
    tokenizer_path = TOKENIZER_DIR / f"bpe_{merges:04d}"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    tok = BPETokenizer.load(str(tokenizer_path))
    
    # Load training data
    train_lines = read_lines(TRAIN_PATH)
    train_stream = build_stream(tok, train_lines)
    
    # Train the model
    if smoothing_method == "interpolation":
        lambdas = [0.1, 0.2, 0.3, 0.4][:n]  # Adjust for different n values
        lambdas = [l/sum(lambdas) for l in lambdas]  # Normalize
        lm = NGramLM(n=n, delta=DELTA, vocab_size=len(tok.vocab), 
                     smoothing_method=smoothing_method,
                     interpolation_lambdas=lambdas)
    else:
        lm = NGramLM(n=n, delta=DELTA, vocab_size=len(tok.vocab), 
                     smoothing_method=smoothing_method)
    
    lm.update(train_stream)
    
    # Generate text
    print(f"\n=== Text Generation (merges={merges}, n={n}, method={smoothing_method}) ===")
    for seed in seed_texts:
        print(f"\nSeed: \"{seed}\"")
        
        # Generate with argmax (deterministic)
        argmax_text = generate_and_decode(lm, tok, seed, use_argmax=True)
        print(f"Argmax: \"{argmax_text}\"")
        
        # Generate with sampling (more creative)
        for temp in [0.5, 1.0]:
            sampled_text = generate_and_decode(lm, tok, seed, temperature=temp)
            print(f"Sampled (temp={temp}): \"{sampled_text}\"")

if __name__ == "__main__":
    if not TRAIN_PATH.exists() or not VAL_PATH.exists():
        raise FileNotFoundError("Expected data/processed/train.txt and val.txt. Build your splits first.")
    
    # First sweep to find best tokenizer and n-gram settings
    sweep_and_select_topK(MERGES_GRID, NGRAM_N, top_k=3)
    
    # Then evaluate text generation with the best settings
    print("\n\n=== Text Generation Evaluation ===")

    # Use best settings from sweep (assuming we've run the sweep)
    # You might want to read the top_merges_for_gpt.csv file to get these dynamically
    best_merges = 2000  # Example - replace with your best value
    best_n = 3  # Example - replace with your best value
    
    # Sample seed texts for generation
    seed_texts = [
        "To be or not to be",
        "All the world's a stage",
        "Friends, Romans, countrymen",
        "Now is the winter of"
    ]
    
    # Test different smoothing methods
    for method in ["laplace", "backoff", "interpolation"]:
        try:
            evaluate_generation(best_merges, best_n, method, seed_texts)
        except Exception as e:
            print(f"Generation with {method} failed: {e}")

from typing import Dict, List, Tuple, Optional
from collections import Counter
from dataclasses import dataclass
import random
import re
import hashlib
import sys
import os
import io
import json

def normalize_text(
    s: str,
    lowercase: bool = True,
    unicode_normalization: str = "NFKC",
    collapse_whitespace: bool = True,
    keep_newlines: bool = True,
) -> str:
    """
    Apply consistent normalization. This MUST be used in both train() and encode().
    """
    try:
        import unicodedata
    except Exception:
        unicodedata = None

    if unicode_normalization and unicodedata is not None:
        s = unicodedata.normalize(unicode_normalization, s)

    if not keep_newlines:
        s = s.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    else:
        # normalize CRLF/CR to LF
        s = s.replace("\r\n", "\n").replace("\r", "\n")

    if lowercase:
        s = s.lower()

    if collapse_whitespace:
        # collapse spaces and tabs (but do NOT touch newlines)
        s = re.sub(r"[ \t]+", " ", s)

    return s


def byte_to_safe_unicode(b: int) -> str:
    """
    Map a byte (0..255) to a private-range unicode char. This is a simple,
    fully reversible mapping that keeps tokens printable.
    """
    return chr(0x0100 + b) 


def unicode_to_byte(ch: str) -> int:
    v = ord(ch)
    if 0x0100 <= v <= 0x01FF:
        return v - 0x0100
    raise ValueError(f"Unexpected symbol '{ch}' outside the byte-mapped range.")


def text_to_byte_symbols(s: str) -> List[str]:
    """
    Encode text to UTF-8 bytes, then map each byte to a unique unicode "symbol".
    """
    byts = s.encode("utf-8")
    return [byte_to_safe_unicode(b) for b in byts]

def byte_symbols_to_text(symbols: List[str]) -> str:
    """
    Inverse of text_to_byte_symbols.
    """
    b = bytes(unicode_to_byte(ch) for ch in "".join(symbols))
    return b.decode("utf-8", errors="strict")

@dataclass
class BPETokenizerConfigSnapshot:
    version: int
    seed: int
    merges: int
    unicode_normalization: str
    lowercase: bool
    collapse_whitespace: bool
    keep_newlines: bool
    tie_break: str
    vocab_size: int
    corpus_sha1: Optional[str]



class BPETokenizer:
    """
    Deterministic BPE tokenizer with:
      - byte-level base alphabet,
      - merge-count training control,
      - BOS/EOS hooks for integration with dataloaders,
      - artifact versioning & diagnostics.

    Public API:
      - train(text: str, merges: int, *, guard_train_only: bool=False, source_tag: str="train")
      - encode(text: str, add_bos: bool=False, add_eos: bool=False) -> List[int]
      - decode(token_ids: List[int]) -> str
      - save(out_dir: str, tag: str, snapshot: Optional[BPETokenizerConfigSnapshot]) -> None
      - load(out_dir: str, tag: str) -> "BPETokenizer"
      - diagnostics(lines: List[str]) -> dict
    """
    def __init__(
        self,
        seed: int = 42,
        bos_token: Optional[str] = "<BOS>",
        eos_token: Optional[str] = "<EOS>",
        pad_token: Optional[str] = None,   # fixed-block training
        unk_token: Optional[str] = None,   # byte-level makes UNK unnecessary
        lowercase: bool = True,
        unicode_normalization: str = "NFKC",
        collapse_whitespace: bool = True,
        keep_newlines: bool = True,
        tie_break: str = "lexicographic",  # deterministic pair tie-break strategy
    ):
         
        self.seed = seed
        random.seed(seed)

        self.special_tokens = []
        for tok in [bos_token, eos_token, pad_token, unk_token]:
            if tok is not None:
                self.special_tokens.append(tok)

        # normalization options
        self.lowercase = lowercase
        self.unicode_normalization = unicode_normalization
        self.collapse_whitespace = collapse_whitespace
        self.keep_newlines = keep_newlines
        
        self.tie_break = tie_break

        # artifacts
        self.vocab: Dict[int, str] = {}         # id -> symbol (symbol is a unicode string of byte-mapped chars)
        self.inverse_vocab: Dict[str, int] = {} # symbol -> id
        self.merges_list: List[Tuple[str, str]] = []  # ordered list of merges (a,b)
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}  # (a,b) -> rank

        # remember last training settings for snapshot
        self._last_merges = 0
        self._last_corpus_sha1: Optional[str] = None

        # assign special token IDs first
        cur_id = 0
        for tok in self.special_tokens:
            if tok in self.inverse_vocab:
                continue
            self.vocab[cur_id] = tok
            self.inverse_vocab[tok] = cur_id
            cur_id += 1

        # add all 256 byte symbols to vocab (no OOV ever)
        for b in range(256):
            sym = byte_to_safe_unicode(b)
            self.vocab[cur_id] = sym
            self.inverse_vocab[sym] = cur_id
            cur_id += 1
        
     
    def train(
        self,
        text: str,
        merges: int,
        *,
        guard_train_only: bool = False,
        source_tag: str = "train",
        min_pair_freq: int = 1,
    ) -> None:
        """
        Train the BPE merges for EXACTLY 'merges' iterations on the PROVIDED TEXT (train split).
        Set guard_train_only=True and source_tag="train" to help avoid accidental leakage.
        """
        if guard_train_only and source_tag.lower() != "train":
            raise RuntimeError("Refusing to train tokenizer: source_tag != 'train' while guard_train_only=True.")

        text_norm = normalize_text(
            text,
            lowercase=self.lowercase,
            unicode_normalization=self.unicode_normalization,
            collapse_whitespace=self.collapse_whitespace,
            keep_newlines=self.keep_newlines,
        )

        self._last_corpus_sha1 = hashlib.sha1(text_norm.encode("utf-8")).hexdigest()

        # represent the corpus as a flat list of base symbols (byte-mapped chars)
        symbols: List[str] = text_to_byte_symbols(text_norm)
     
        # run merges
        for _ in range(merges):
            pair = self._most_frequent_pair(symbols)
            if pair is None:
                break
            (a, b), count = pair
            if count < min_pair_freq:
                break

            merged = a + b
            self._apply_merge_in_place(symbols, a, b, merged)

            # Record the merge and update vocab
            self.merges_list.append((a, b))
            new_id = max(self.vocab.keys()) + 1
            self.vocab[new_id] = merged
            self.inverse_vocab[merged] = new_id

        # build rank table for encode()
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges_list)}
        self._last_merges = len(self.merges_list)

    def _most_frequent_pair(self, symbols: List[str]) -> Optional[Tuple[Tuple[str, str], int]]:
        """
        Count adjacent pairs and return the (pair, count) with deterministic tie-break.
        """
        if len(symbols) < 2:
            return None
        ctr: Dict[Tuple[str, str], int] = Counter(zip(symbols, symbols[1:]))
        if not ctr:
            return None

        # deterministic tie-break: (count DESC, merged_string LEX, a LEX, b LEX)
        def key(item):
            (a, b), cnt = item
            return (cnt, a + b, a, b)

        (best_pair, best_count) = max(ctr.items(), key=key)
        return best_pair, best_count

    def _apply_merge_in_place(self, symbols: List[str], a: str, b: str, merged: str) -> None:
        """
        In-place pass merging all occurrences of (a, b) into 'merged'.
        """
        i = 0
        out: List[str] = []
        N = len(symbols)
        while i < N:
            if i < N - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        symbols[:] = out

    #-----------------Encoder / Decoder part----------------------

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Encode text -> token ids using learned merges (greedy GPT-2-style pair merging).
        """
        text_norm = normalize_text(
            text,
            lowercase=self.lowercase,
            unicode_normalization=self.unicode_normalization,
            collapse_whitespace=self.collapse_whitespace,
            keep_newlines=self.keep_newlines,
        )

        # start with base symbols (one per byte)
        symbols = text_to_byte_symbols(text_norm)

        # iteratively merge lowest-rank pairs until no more found
        if self.bpe_ranks:
            while True:
                pairs = set(zip(symbols, symbols[1:]))
                if not pairs:
                    break
                # choose the pair with the BEST (lowest) rank among known merges
                bigram = None
                best_rank = sys.maxsize
                for p in pairs:
                    r = self.bpe_ranks.get(p, sys.maxsize)
                    if r < best_rank:
                        best_rank = r
                        bigram = p
                if bigram is None or best_rank == sys.maxsize:
                    break
                a, b = bigram
                merged = a + b

                # merge all occurrences of (a,b)
                i = 0
                out: List[str] = []
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(symbols[i])
                        i += 1
                symbols = out

                if len(symbols) == 1:
                    break

        # map symbols to ids
        ids = []
        if add_bos and "<BOS>" in self.inverse_vocab:
            ids.append(self.inverse_vocab["<BOS>"])
        for sym in symbols:
            tid = self.inverse_vocab.get(sym)
            if tid is None:
                raise KeyError(f"Symbol not in vocab, can't encode: {repr(sym)}")
            ids.append(tid)
        if add_eos and "<EOS>" in self.inverse_vocab:
            ids.append(self.inverse_vocab["<EOS>"])
        return ids

     
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token ids -> text (UTF-8). Special tokens are omitted.
        """
        symbols: List[str] = []
        skip = set(tok for tok in ["<BOS>", "<EOS>", "<PAD>", "<UNK>"] if tok in self.inverse_vocab)
        skip_ids = {self.inverse_vocab[tok] for tok in skip}

        for tid in token_ids:
            if tid not in self.vocab:
                raise KeyError(f"Token id {tid} not found in vocab.")
            if tid in skip_ids:
                continue
            symbols.append(self.vocab[tid])

        return byte_symbols_to_text(symbols)

    #-----------Artifacts---------

    def save(self, out_dir: str, tag: str, snapshot: Optional[BPETokenizerConfigSnapshot] = None) -> None:
        """
        Write:
          - vocab.json  (id -> symbol)
          - merges.txt  (each line: "<a> <b>")
          - config.yaml (snapshot of settings for reproducibility)
        """
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, tag)

        # vocab.json
        with io.open(base + "_vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        # merges.txt
        with io.open(base + "_merges.txt", "w", encoding="utf-8") as f:
            for a, b in self.merges_list:
                f.write(f"{a} {b}\n")

        # config.yaml snapshot 
        if snapshot is None:
            snapshot = BPETokenizerConfigSnapshot(
                version=1,
                seed=self.seed,
                merges=self._last_merges,
                unicode_normalization=self.unicode_normalization,
                lowercase=self.lowercase,
                collapse_whitespace=self.collapse_whitespace,
                keep_newlines=self.keep_newlines,
                tie_break=self.tie_break,
                vocab_size=len(self.vocab),
                corpus_sha1=self._last_corpus_sha1,
            )

        yaml = [
            f"version: {snapshot.version}",
            f"seed: {snapshot.seed}",
            f"merges: {snapshot.merges}",
            f"unicode_normalization: {snapshot.unicode_normalization}",
            f"lowercase: {str(snapshot.lowercase).lower()}",
            f"collapse_whitespace: {str(snapshot.collapse_whitespace).lower()}",
            f"keep_newlines: {str(snapshot.keep_newlines).lower()}",
            f"tie_break: {snapshot.tie_break}",
            f"vocab_size: {snapshot.vocab_size}",
            f"corpus_sha1: {snapshot.corpus_sha1 if snapshot.corpus_sha1 else 'null'}",
        ]
        with io.open(base + "_config.yaml", "w", encoding="utf-8") as f:
            f.write("\n".join(yaml) + "\n")

    @classmethod
    def load(cls, out_dir: str, tag: str) -> "BPETokenizer":
        """
        Load tokenizer artifacts saved by save().
        """
        base = os.path.join(out_dir, tag)

        # load vocab
        with io.open(base + "_vocab.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)
        # instantiate a bare tokenizer and inject artifacts
        tok = cls()
        tok.vocab = {int(k): v for k, v in vocab.items()}
        tok.inverse_vocab = {v: int(k) for k, v in tok.vocab.items()}

        # load merges
        merges_path = base + "_merges.txt"
        tok.merges_list = []
        tok.bpe_ranks = {}
        if os.path.exists(merges_path):
            with io.open(merges_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    a, b = line.split(" ", 1)
                    tok.merges_list.append((a, b))
                    tok.bpe_ranks[(a, b)] = i
        tok._last_merges = len(tok.merges_list)
        return tok
    
    # -------------------- Diagnostics ----------------------

    def diagnostics(self, lines: List[str]) -> dict:
        """
        Quick health metrics (use on train/val to guide merge-count selection).
        """
        # tokenize line-by-line with EOS 
        lengths = []
        total_tokens = 0
        for ln in lines:
            ids = self.encode(ln, add_bos=False, add_eos=True)
            L = len(ids)
            lengths.append(L)
            total_tokens += L

        avg_tokens_per_line = (sum(lengths) / max(len(lengths), 1)) if lengths else 0.0
        # unique token types / total tokens (rough heuristic)
        unique_types = len(self.vocab)
        ttr = unique_types / max(total_tokens, 1)

        metrics = dict(
            avg_tokens_per_line=avg_tokens_per_line,
            type_token_ratio=ttr,
            oov_rate=0.0,
            vocab_size=len(self.vocab),
        )
        return metrics


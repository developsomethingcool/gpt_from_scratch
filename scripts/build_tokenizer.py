import argparse
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from tokenizers.processors import TemplateProcessing

TRAIN_PATH = Path("data/processed/train.txt")
TOKENIZER_DIR = Path("tokenizer")

def train_bpe(merges: int):
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f" Train split not found at {TRAIN_PATH}. Run build_splits.py first.")

    print(f"Reading train data from {TRAIN_PATH}")
    train_text = TRAIN_PATH.read_text(encoding="utf-8").splitlines()

    # Normalizer: lowercase + NFKC
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.Lowercase()
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    # Trainer: learns merges + vocab
    trainer = trainers.BpeTrainer(
        vocab_size=merges,
        min_frequency=2,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
    )

    print(f"🔧 Training BPE tokenizer with vocab size = {merges}")
    tokenizer.train_from_iterator(train_text, trainer=trainer)

    # Post-processing: add BOS/EOS handling
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B:1 [EOS]:1",
        special_tokens=[("[BOS]", tokenizer.token_to_id("[BOS]")),
                        ("[EOS]", tokenizer.token_to_id("[EOS]"))]
    )

    # Save files
    out_dir = TOKENIZER_DIR / f"bpe_{merges}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.model.save(str(out_dir))

    # Also save tokenizer.json (full config)
    tokenizer.save(str(out_dir / "tokenizer.json"))

    print(f"Tokenizer saved in {out_dir}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merges", type=int, default=5000,
                        help="Number of merges (vocab size) for BPE")
    args = parser.parse_args()
    train_bpe(args.merges)

if __name__ == "__main__":
    main()

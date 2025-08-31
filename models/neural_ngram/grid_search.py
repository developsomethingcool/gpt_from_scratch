"""
Grid search script for hyperparameter tuning.
---------------------------------------------
Sweeps over learning rates and merge counts.
Logs results to reports/neural_grid.csv.
"""

import argparse
import pandas as pd
from models.neural_ngram.train import train_ngram
from models.neural_ngram.utils import get_dataloader
import os


def grid_search(vocab_sizes, learning_rates, context_size, block_size, batch_size, epochs, patience):
    results = []
    os.makedirs("reports", exist_ok=True)
    for vocab_size in vocab_sizes:
        for lr in learning_rates:
            print(f"Running grid search with vocab_size={vocab_size}, lr={lr}")
            train_loader = get_dataloader('train', block_size=block_size, batch_size=batch_size)
            val_loader = get_dataloader('valid', block_size=block_size, batch_size=batch_size)
            model = train_ngram(vocab_size, context_size, train_loader, val_loader, lr=lr, epochs=epochs, patience=patience)
            results.append({"vocab_size": vocab_size, "learning_rate": lr, "best_val_loss": model.best_val_loss})
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_sizes", nargs="+", type=int, default=[1000, 2000, 3000])
    parser.add_argument("--learning_rates", nargs="+", type=float, default=[1e-3, 1e-4])
    parser.add_argument("--context_size", type=int, default=3)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    results = grid_search(
        vocab_sizes=args.vocab_sizes,
        learning_rates=args.learning_rates,
        context_size=args.context_size,
        block_size=args.block_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("reports/neural_grid.csv", index=False)
    print("Grid search results saved to reports/neural_grid.csv")
# GPT From Scratch

A complete implementation of the Generative Pre-trained Transformer (GPT) architecture built for GPT from Scratch Course

## Overview

This repository contains implementations of multiple language modeling approaches, including classical n-gram models, neural n-gram models, and a mini-GPT transformer. The project includes BPE tokenization, comprehensive training pipelines, and text generation capabilities.

## Features

- Pure PyTorch implementation built without high-level abstractions
- Multiple model types: Classical N-gram, Neural N-gram, and Mini-GPT
- BPE tokenizer implementation with configurable merge operations
- Universal training pipeline supporting all model types
- Device-optimized training (CPU, CUDA, MPS)
- Text generation capabilities with sampling strategies
- Comprehensive model evaluation and comparison tools

## Requirements

```
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.64.0
```

## Installation

Clone the repository:
```bash
git clone https://github.com/developsomethingcool/gpt_from_scratch.git
cd gpt_from_scratch
```

Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Classical N-gram Model

```python
from gpt_from_scratch.classic_ngram.classic_ngram import NGramLM
from gpt_from_scratch.tokenizer.tokenizer_bpe import BPETokenizer

# Initialize tokenizer
tokenizer = BPETokenizer(seed=42)
tokenizer.train(text_data, merges=300)

# Create and train n-gram model
model = NGramLM(n=3, delta=1.0, vocab_size=tokenizer.vocab_size)
model.update(token_stream)

# Generate text
generated_tokens = model.generate_text(prefix_tokens, max_length=100)
```

### Neural N-gram Model

```python
from gpt_from_scratch.neural_ngram.neural_ngram import NeuralNGramModel
from gpt_from_scratch.trainer.trainer import ModelTrainer

# Initialize model
model = NeuralNGramModel(
    vocab_size=5000,
    context_size=4,
    embedding_dim=128,
    hidden_dim=256,
    dropout=0.1
)

# Train model
config = {
    'epochs': 10,
    'batch_size': 64,
    'learning_rate': 3e-4
}
trainer = ModelTrainer('neural_ngram', config)
trained_model, metrics = trainer.train(model, train_data, val_data)

# Generate text
generated = model.generate(input_ids, max_new_tokens=50, temperature=0.8)
```

### Mini-GPT Transformer

```python
from gpt_from_scratch.mini_gpt.mini_gpt import MiniGPT
from gpt_from_scratch.trainer.trainer import ModelTrainer

# Initialize Mini-GPT
model = MiniGPT(
    vocab_size=10000,
    embedding_dim=512,
    context_size=256,
    n_layers=6,
    n_heads=8,
    dropout=0.1
)

# Train model
config = {
    'epochs': 20,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'use_mixed_precision': True
}
trainer = ModelTrainer('mini_gpt', config)
trained_model, metrics = trainer.train(model, train_data, val_data)

# Generate text
generated = model.generate(prefix_tokens, max_new_tokens=100, temperature=0.8, top_k=50)
```

### BPE Tokenizer

```python
from gpt_from_scratch.tokenizer.tokenizer_bpe import BPETokenizer

# Initialize and train tokenizer
tokenizer = BPETokenizer(
    seed=42,
    bos_token="<BOS>",
    eos_token="<EOS>",
    lowercase=True
)

# Train on text data
tokenizer.train(text_data, merges=500, guard_train_only=True)

# Encode and decode
token_ids = tokenizer.encode("Hello world!", add_bos=True, add_eos=True)
text = tokenizer.decode(token_ids)

# Save and load
tokenizer.save("tokenizers", "my_tokenizer")
loaded_tokenizer = BPETokenizer.load("tokenizers", "my_tokenizer")
```

## Project Structure

```
gpt_from_scratch/
├── README.md
├── requirements.txt
├── src/
│   └── gpt_from_scratch/
│       ├── classic_ngram/
│       │   ├── __init__.py
│       │   └── classic_ngram.py      # Classical n-gram language model
│       ├── neural_ngram/
│       │   ├── __init__.py
│       │   └── neural_ngram.py       # Neural n-gram model
│       ├── mini_gpt/
│       │   ├── __init__.py
│       │   └── mini_gpt.py          # Mini-GPT transformer implementation
│       ├── tokenizer/
│       │   ├── __init__.py
│       │   └── tokenizer_bpe.py     # BPE tokenizer implementation
│       └── trainer/
│           ├── __init__.py
│           └── trainer.py           # Universal model trainer
├── data/
│   └── processed/                   # Training and validation splits
├── tokenizers/                      # Saved tokenizer artifacts
├── logs/                           # Training logs and metrics
└── experiments/                    # Experiment outputs
```

## Model Architecture

### Classical N-gram
- Frequency-based statistical model with smoothing techniques
- Supports Laplace smoothing, backoff, and interpolation
- Configurable n-gram order (unigram to 4-gram)

### Neural N-gram
- Embedding layer for token representations
- Multi-layer perceptron for next-token prediction
- GELU activation and dropout regularization

### Mini-GPT Transformer
- **Multi-head Self-attention**: Causal attention mechanism
- **Feed-forward Networks**: GELU activation with residual connections
- **Layer Normalization**: Applied before sublayers (Pre-LN)
- **Positional Embeddings**: Learned position encodings
- **Causal Masking**: Ensures autoregressive generation

## Training Features

- **Universal Trainer**: Supports all model types with unified interface
- **Device Optimization**: Automatic detection and optimization for CPU/CUDA/MPS
- **Mixed Precision**: Available for CUDA devices to improve performance
- **Gradient Clipping**: Prevents gradient explosion during training
- **Model Compilation**: Automatic PyTorch 2.0 compilation when available

## Text Generation

All models support text generation with various sampling strategies:
- **Temperature Scaling**: Controls randomness in generation
- **Top-k Sampling**: Limits sampling to k most likely tokens
- **Deterministic Generation**: Argmax selection for reproducible output


"""
Generate samples using the trained model.
-----------------------------------------
Generates 2–3 short samples and saves them to reports/neural_ablation.md.
"""

import torch
from models.neural_ngram.model import NeuralNGramModel

def generate_samples(model_path, vocab_size, context_size, num_samples=3):
    model = NeuralNGramModel(vocab_size, context_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    samples = []
    for _ in range(num_samples):
        context = torch.randint(0, vocab_size, (1, context_size))
        generated = context.tolist()[0]
        for _ in range(20):  # Generate up to 20 tokens
            logits = model(context)
            next_token = torch.argmax(logits, dim=-1).item()
            generated.append(next_token)
            context = torch.tensor(generated[-context_size:]).unsqueeze(0)
        samples.append(generated)
    return samples

if __name__ == "__main__":
    model_path = "models/neural_ngram/checkpoint.pt"
    vocab_size = 5000
    context_size = 3
    samples = generate_samples(model_path, vocab_size, context_size)
    with open("reports/neural_ablation.md", "w") as f:
        for i, sample in enumerate(samples):
            f.write(f"Sample {i+1}: {sample}\n")
    print("Samples saved to reports/neural_ablation.md")
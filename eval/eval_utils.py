import torch
import torch.nn.functional as F
from eval.dataloader import get_dataloader

def evaluate_perplexity(model, split="val", block_size=128, batch_size=32, tokenizer_dir="tokenizer/bpe_5000", device="cpu"):
    """
    Compute perplexity on a given split using the provided model.
    """
    model.eval()
    model.to(device)
    
    dl = get_dataloader(split, block_size=block_size, batch_size=batch_size, tokenizer_dir=tokenizer_dir, device=device)
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            
            logits = model(x)  # shape: [batch_size, block_size, vocab_size]
            
            # reshape for cross-entropy: [batch_size*block_size, vocab_size]
            logits = logits.view(-1, logits.size(-1))
            y = y.view(-1)
            
            loss = F.cross_entropy(logits, y, reduction="sum")
            
            total_loss += loss.item()
            total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss))
    return ppl.item()

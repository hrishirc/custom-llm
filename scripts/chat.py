#!/usr/bin/env python3
"""Chat with the trained model."""
import sys
import torch
import warnings
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.llm import LLM
from src.tokenizer.tokenizer import TokenizerWrapper
from src.model.config import ModelConfig

# Suppress warnings
warnings.filterwarnings("ignore")

def load_latest_checkpoint(checkpoint_dir: Path):
    """Find the latest checkpoint."""
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        print("No checkpoints found!")
        return None
    
    # Sort by modification time
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"Loading checkpoint: {latest.name}")
    return latest

def generate(
    model: LLM, 
    tokenizer: TokenizerWrapper, 
    prompt: str, 
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    device: str = "cpu"
):
    """Generate text from the model."""
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    print(f"\nPrompt: {prompt}")
    print("-" * 40)
    print(prompt, end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to context window if needed
            ctx = input_ids[:, -model.config.max_seq_len:]
            
            # Forward pass
            logits = model(ctx)["logits"]
            next_token_logits = logits[:, -1, :]
            
            # Temperature sampling
            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                
                # Top-k sampling
                if top_k > 0:
                    v, _ = torch.topk(probs, min(top_k, probs.size(-1)))
                    probs[probs < v[:, [-1]]] = 0
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append and print
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Decode just the new token locally if possible, or batched
            # For simplicity, we'll just decode the new token
            text = tokenizer.decode(next_token[0].tolist())
            print(text, end="", flush=True)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    print("\n" + "-" * 40)

def main():
    root_dir = Path(__file__).parent.parent
    checkpoint_dir = root_dir / "checkpoints"
    tokenizer_path = root_dir / "data" / "tokenizer.json"
    
    # Setup device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = TokenizerWrapper(str(tokenizer_path))
    
    # Load model
    checkpoint_path = load_latest_checkpoint(checkpoint_dir)
    if not checkpoint_path:
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Use default ModelConfig (checkpoint stores TrainingConfig, not model architecture)
    config = ModelConfig()
    
    model = LLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    print("Model loaded! Type 'quit' to exit.")
    
    while True:
        try:
            prompt = input("\nYou: ")
            if prompt.lower() in ["quit", "exit"]:
                break
            
            generate(model, tokenizer, prompt, device=device)
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()

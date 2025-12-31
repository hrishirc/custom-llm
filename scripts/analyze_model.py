#!/usr/bin/env python3
"""Analyze model generation quality with fixed prompts."""
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.llm import LLM
from src.tokenizer.tokenizer import TokenizerWrapper
from src.model.config import ModelConfig

def load_model_and_tokenizer():
    root_dir = Path(__file__).parent.parent
    checkpoint_dir = root_dir / "checkpoints"
    tokenizer_path = root_dir / "data" / "tokenizer.json"
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = TokenizerWrapper(str(tokenizer_path))
    
    # Find latest checkpoint
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        print("No checkpoints found!")
        return None, None, None, None
        
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"Loading checkpoint: {latest.name}")
    
    # Load checkpoint
    checkpoint = torch.load(latest, map_location=device, weights_only=False)
    
    # Init model with default config
    config = ModelConfig() 
    model = LLM(config)
    
    # Load state
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, tokenizer, device, latest

def main():
    model, tokenizer, device, checkpoint_path = load_model_and_tokenizer()
    if not model:
        return

    prompts = [
        "The universe is",
        "Mathematics can be defined as",
    ]
    
    settings = [
        {"temp": 0.1, "top_k": 50, "desc": "Conservative (Temp=0.1)"},
        {"temp": 0.7, "top_k": 50, "desc": "Balanced (Temp=0.7)"},
        {"temp": 1.2, "top_k": 50, "desc": "Creative/Random (Temp=1.2)"},
        {"temp": 0.7, "top_k": 10, "desc": "Focused (Top-K=10)"},
    ]

    print(f"\n{'='*80}")
    print(f"ANALYSIS REPORT - {checkpoint_path.name}")
    print(f"{'='*80}")

    for setting in settings:
        print(f"\n>>> SETTING: {setting['desc']} (T={setting['temp']}, K={setting['top_k']})")
        for prompt in prompts:
            # Encode
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            
            # Generate
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=40,
                    temperature=setting["temp"],
                    top_k=setting["top_k"]
                )
            
            # Decode
            generated = output_ids[0].tolist()
            generated_text = ""
            for token_id in generated:
                word = tokenizer.tokenizer.id_to_token(token_id)
                if word:
                    word = word.replace("\u2581", " ")
                    if word in ["<BOS>", "<EOS>", "<PAD>", "<UNK>"]:
                        continue
                    generated_text += word
            
            # Print formatted
            print(f"  PROMPT: '{prompt}'")
            print(f"  OUTPUT: {generated_text.strip()}")
            print("-" * 40)

if __name__ == "__main__":
    main()

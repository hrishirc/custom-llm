import sys
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(".")

from src.model.llm import create_model
from src.model.config import ModelConfig

def visualize_gradients():
    print("Initializing model...")
    # Create smaller model for speed if needed, or full model
    # Use full config to match training
    config = ModelConfig() 
    model = create_model(config)
    
    # Check if checkpoint exists and load it
    ckpt_dir = Path("checkpoints/phase1_grammar")
    latest_ckpt = None
    if ckpt_dir.exists():
        ckpts = list(ckpt_dir.glob("*.pt"))
        if ckpts:
            # Sort by modification time
            latest_ckpt = max(ckpts, key=lambda p: p.stat().st_mtime)
            print(f"Loading checkpoint: {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location="cpu")
            # Handle compiled model keys
            state_dict = checkpoint["model_state_dict"]
            final_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("_orig_mod."):
                    final_state_dict[k[10:]] = v
                else:
                    final_state_dict[k] = v
            model.load_state_dict(final_state_dict)
    
    # Create dummy batch
    print("Running forward/backward pass...")
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    # Forward pass
    outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]
    
    # Backward pass
    loss.backward()
    
    # Collect gradients
    print("Collecting gradients...")
    grads = []
    layers = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
             # Identify layer
             if "layers." in name:
                 # Format: layers.0.attention...
                 layer_idx = int(name.split(".")[1])
                 norm = param.grad.data.norm(2).item()
                 grads.append((layer_idx, norm))
    
    # Aggregate per layer (max or avg)
    layer_norms = {}
    for idx, norm in grads:
        if idx not in layer_norms:
            layer_norms[idx] = []
        layer_norms[idx].append(norm)
    
    # Compute average norm per layer
    x = sorted(layer_norms.keys())
    y = [sum(layer_norms[i])/len(layer_norms[i]) for i in x]
    
    # Plot
    print("Generating plot...")
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.title(f"Gradient Norms by Layer (Sequence Length {seq_len})")
    plt.xlabel("Layer Index (0 = Input, 60 = Output)")
    plt.ylabel("Average Gradient Norm")
    plt.grid(True, alpha=0.3)
    
    # Highlight specific trends
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    output_path = "gradient_flow.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    visualize_gradients()

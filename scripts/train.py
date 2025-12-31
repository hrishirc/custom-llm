#!/usr/bin/env python3
"""Main training script for the 60M parameter LLM.

Usage:
    # Full training (all phases)
    python scripts/train.py --data-dir data/tokenized --output-dir checkpoints
    
    # Single phase
    python scripts/train.py --phase 1 --data-dir data/tokenized
    
    # Smoke test (verify everything works)
    python scripts/train.py --smoke-test
"""

import argparse
from pathlib import Path
import sys
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.config import ModelConfig, TrainingConfig, PHASE_CONFIGS
from src.model.llm import create_model, LLM
from src.data.dataloader import CurriculumDataLoader, TokenizedDataset
from src.training.trainer import Trainer


def run_smoke_test(args):
    """Run a quick smoke test to verify the setup."""
    print("\n" + "="*60)
    print("SMOKE TEST: Verifying model and training setup")
    print("="*60 + "\n")
    
    # Create config with smaller model for testing
    if args.tiny:
        config = ModelConfig(
            n_layers=4,
            hidden_size=128,
            n_heads=4,
            head_dim=32,
            intermediate_size=256,
            vocab_size=1000,
            max_seq_len=128,
        )
    else:
        config = ModelConfig()
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    
    # Print parameter count
    n_params = model.get_num_params()
    print(f"Total parameters: {n_params:,}")
    
    # Check device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, labels=labels)
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Loss: {outputs['loss'].item():.4f}")
    
    # Test backward pass
    print("\nTesting backward pass...")
    outputs["loss"].backward()
    
    # Check gradients
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5
    print(f"  Gradient norm: {grad_norm:.4f}")
    
    # Test generation
    print("\nTesting generation...")
    model.eval()
    prompt = torch.tensor([[1]], device=device)  # BOS token
    generated = model.generate(prompt, max_new_tokens=20, temperature=1.0)
    print(f"  Generated tokens: {generated[0].tolist()}")
    
    # Test layer freezing
    print("\nTesting layer freezing...")
    model.freeze_layers(2)
    frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
    total_count = sum(1 for p in model.parameters())
    print(f"  Frozen parameters: {frozen_count}/{total_count}")
    
    # Memory usage
    if device.type == "mps":
        # MPS doesn't have memory_allocated, estimate from parameter count
        param_memory = n_params * 4 / (1024**3)  # FP32 in GB
        print(f"\nEstimated model memory: {param_memory:.2f} GB (FP32)")
    
    print("\n" + "="*60)
    print("SMOKE TEST PASSED!")
    print("="*60 + "\n")
    
    return True


def train_phase(
    phase_name: str,
    model: LLM,
    data_path: Path,
    checkpoint_dir: Path,
    log_dir: Path,
    config: TrainingConfig,
    total_steps: int,
):
    """Train a single phase."""
    print(f"\n{'='*60}")
    print(f"Phase: {phase_name}")
    print(f"Data: {data_path}")
    print(f"Steps: {total_steps}")
    print(f"{'='*60}\n")
    
    # Create data loader
    data_loader = CurriculumDataLoader(
        data_path=data_path,
        batch_size=config.micro_batch_size,
        context_schedule=config.context_schedule,
        num_workers=config.num_workers,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_config=config,
        data_loader=data_loader,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir / phase_name,
    )
    
    # Train
    metrics = trainer.train(
        total_steps=total_steps,
        phase_name=phase_name,
    )
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train 60M parameter LLM")
    
    # Data
    parser.add_argument("--data-dir", type=str, default="data/tokenized",
                        help="Directory with tokenized data")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Checkpoint output directory")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="TensorBoard log directory")
    
    # Training
    parser.add_argument("--phase", type=str, choices=["1", "2", "2b", "all"], default="all",
                        help="Training phase to run")
    parser.add_argument("--steps", type=int, help="Override total steps")
    parser.add_argument("--resume", type=str, help="Checkpoint to resume from")
    
    # Testing
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test only")
    parser.add_argument("--tiny", action="store_true", help="Use tiny model for testing")
    
    args = parser.parse_args()
    
    # Smoke test
    if args.smoke_test:
        run_smoke_test(args)
        return
    
    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    
    # Create model
    config = ModelConfig()
    model = create_model(config)
    
    # Resume if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed from {args.resume}")
    
    # Define phases
    phases = {
        "1": {
            "name": "phase1_grammar",
            "data": data_dir / "phase1.npy",
            "config": PHASE_CONFIGS["phase1_grammar"],
            "steps": args.steps or 50000,  # ~200M tokens
        },
        "2": {
            "name": "phase2_vocabulary",
            "data": data_dir / "phase2.npy",
            "config": PHASE_CONFIGS["phase2_vocabulary"],
            "steps": args.steps or 12500,  # ~50M tokens
        },
        "2b": {
            "name": "phase2b_scientific",
            "data": data_dir / "phase2b.npy",
            "config": PHASE_CONFIGS["phase2b_scientific"],
            "steps": args.steps or 6250,  # ~25M tokens
        },
    }
    
    # Select phases to run
    if args.phase == "all":
        phase_names = ["1", "2", "2b"]
    else:
        phase_names = [args.phase]
    
    # Train each phase
    for phase_name in phase_names:
        phase = phases[phase_name]
        
        if not phase["data"].exists():
            print(f"Warning: Data file not found: {phase['data']}")
            print("Please run data preparation first.")
            continue
        
        train_phase(
            phase_name=phase["name"],
            model=model,
            data_path=phase["data"],
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            config=phase["config"],
            total_steps=phase["steps"],
        )
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

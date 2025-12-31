#!/usr/bin/env python3
"""Modal GPU deployment for LLM training.

This script runs the same training code on Modal's cloud GPUs.
Local training remains unchanged - this is an additional deployment option.

Usage:
    # First time setup:
    pip install modal
    modal token new
    
    # Upload data to Modal volume:
    modal run scripts/modal_train.py::upload_data
    
    # Run training:
    modal run scripts/modal_train.py::train_all
    
    # Or run a specific phase:
    modal run scripts/modal_train.py::train_phase --phase 1
    
    # Download checkpoints:
    modal run scripts/modal_train.py::download_checkpoints
"""

import modal
import os
from pathlib import Path

# Define the Modal app
app = modal.App("custom-llm-training")

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.1.0",
        "datasets>=2.14.0",
        "tokenizers>=0.15.0",
        "huggingface-hub>=0.19.0",
        "transformers>=4.36.0",
        "tqdm>=4.66.0",
        "numpy>=1.24.0",
        "tensorboard>=2.15.0",
        "temporalio>=1.4.0",  # Required by workflows module
    )
    .apt_install("git")
)

# Persistent volume for data and checkpoints
volume = modal.Volume.from_name("llm-training-data", create_if_missing=True)

# Mount paths
DATA_PATH = "/data"
CODE_PATH = "/code"


@app.function(
    image=image,
    volumes={DATA_PATH: volume},
    timeout=3600,  # 1 hour for upload
)
def upload_data():
    """Upload tokenized data and state to Modal volume."""
    import shutil
    
    volume.reload()  # Ensure we see latest state
    
    print("=" * 60)
    print("Uploading data to Modal volume...")
    print("=" * 60)
    
    # List what's in the volume
    data_dir = Path(DATA_PATH)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    existing = list(data_dir.glob("**/*"))
    if existing:
        print(f"Found {len(existing)} existing files in volume")
    
    print("\nTo upload your local data, run these commands locally:")
    print("-" * 60)
    print("""
# Install modal CLI if not already installed
pip install modal

# Upload tokenized data (required)
modal volume put llm-training-data data/tokenized/ tokenized/

# Upload tokenizer (required)
modal volume put llm-training-data data/tokenizer.json tokenizer.json

# Upload state DB (optional, for resume)
modal volume put llm-training-data data/training_state.db training_state.db

# Upload checkpoints (optional, for resume)
modal volume put llm-training-data checkpoints/ checkpoints/
""")
    
    volume.commit()
    print("\nVolume ready!")


@app.function(
    image=image,
    gpu="A10G",  # Options: "T4", "A10G", "A100", "H100"
    volumes={DATA_PATH: volume},
    timeout=86400,  # 24 hours max
    memory=32768,  # 32GB RAM
)
def train_phase(phase: str = "1"):
    """Train a single phase on Modal GPU.
    
    Args:
        phase: Training phase ("1", "2", or "2b")
    """
    import sys
    import torch
    
    volume.reload()
    
    print("=" * 60)
    print(f"Modal GPU Training - Phase {phase}")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch version: {torch.__version__}")
    print("=" * 60)
    
    # Setup paths
    data_dir = Path(DATA_PATH)
    
    # Check for required files
    tokenized_dir = data_dir / "tokenized"
    tokenizer_path = data_dir / "tokenizer.json"
    
    if not tokenized_dir.exists():
        raise RuntimeError(
            f"Tokenized data not found at {tokenized_dir}. "
            "Run 'modal run scripts/modal_train.py::upload_data' first."
        )
    
    if not tokenizer_path.exists():
        raise RuntimeError(
            f"Tokenizer not found at {tokenizer_path}. "
            "Upload data/tokenizer.json to the Modal volume."
        )
    
    print(f"\nTokenized files: {list(tokenized_dir.glob('*.npy'))}")
    
    # Create directories
    checkpoint_dir = data_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Import and setup training (dynamic import after paths are set)
    sys.path.insert(0, "/code")
    
    # We need to copy the code to /code or adjust imports
    # For now, inline the training logic
    from pathlib import Path
    import numpy as np
    
    # Configure for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Import training components
    # Since we can't easily import from local src/, we'll use a simpler approach
    # that copies the key training logic
    
    print("\n⚠️  Direct Modal training requires code bundling.")
    print("For full integration, we recommend using modal.Mount to include src/")
    print("\nAlternatively, run the orchestrator with Modal's local run mode:")
    print("  modal run --detach scripts/modal_train.py::train_phase --phase 1")
    
    volume.commit()


@app.function(
    image=image,
    volumes={DATA_PATH: volume},
    timeout=3600,
)
def download_checkpoints(local_path: str = "checkpoints_modal"):
    """List checkpoints available in Modal volume."""
    volume.reload()
    
    checkpoint_dir = Path(DATA_PATH) / "checkpoints"
    
    if not checkpoint_dir.exists():
        print("No checkpoints found in Modal volume")
        return
    
    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    print(f"\nFound {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        print(f"  - {ckpt.name} ({size_mb:.1f} MB)")
    
    print(f"\nTo download locally, run:")
    print(f"  modal volume get llm-training-data checkpoints/ {local_path}/")


@app.function(
    image=image.add_local_dir("src", remote_path=f"{CODE_PATH}/src")
               .add_local_dir("scripts", remote_path=f"{CODE_PATH}/scripts")
               .add_local_file("requirements.txt", remote_path=f"{CODE_PATH}/requirements.txt"),
    gpu="A10G",
    volumes={DATA_PATH: volume},
    timeout=86400,
    memory=32768,
)
def train_all():
    """Run full training pipeline on Modal GPU with code mounted."""
    import sys
    import os
    import torch
    
    volume.reload()
    
    print("=" * 60)
    print("Modal GPU Training - Full Pipeline")
    print("=" * 60)
    
    # GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"PyTorch version: {torch.__version__}")
    print("=" * 60)
    
    # Add code to path
    sys.path.insert(0, CODE_PATH)
    os.chdir(CODE_PATH)
    
    # Setup symlinks for data paths (Modal volume -> expected paths)
    data_dir = Path(CODE_PATH) / "data"
    data_dir.mkdir(exist_ok=True)
    
    modal_data = Path(DATA_PATH)
    
    # Create symlinks
    symlinks = [
        ("tokenized", modal_data / "tokenized"),
        ("tokenizer.json", modal_data / "tokenizer.json"),
        ("training_state.db", modal_data / "training_state.db"),
    ]
    
    for name, target in symlinks:
        link = data_dir / name
        if not link.exists() and target.exists():
            link.symlink_to(target)
            print(f"Linked: {link} -> {target}")
    
    # Checkpoint dir points to volume for persistence
    ckpt_link = Path(CODE_PATH) / "checkpoints"
    modal_ckpt = modal_data / "checkpoints"
    modal_ckpt.mkdir(exist_ok=True)
    if not ckpt_link.exists():
        ckpt_link.symlink_to(modal_ckpt)
        print(f"Linked: {ckpt_link} -> {modal_ckpt}")
    
    # Log dir
    log_link = Path(CODE_PATH) / "logs"
    modal_log = modal_data / "logs"
    modal_log.mkdir(exist_ok=True)
    if not log_link.exists():
        log_link.symlink_to(modal_log)
    
    # Override device to CUDA
    os.environ["TRAINING_DEVICE"] = "cuda"
    
    # Import and run orchestrator
    from src.workflows.native_orchestrator import NativeOrchestrator
    
    orchestrator = NativeOrchestrator(
        db_path=str(modal_data / "training_state.db"),
        data_dir="data",
        checkpoint_dir="checkpoints",
        log_dir="logs",
        phases=["1", "2", "2b"],
    )
    
    try:
        orchestrator.run()
    finally:
        # Commit volume to persist checkpoints
        volume.commit()
        print("\nVolume committed - checkpoints persisted!")


@app.local_entrypoint()
def main(command: str = "help", phase: str = "1"):
    """Main entry point for Modal CLI.
    
    Commands:
        upload_data: Upload local data to Modal volume
        train: Run training on Modal GPU
        download: Download checkpoints from Modal
    """
    if command == "upload":
        upload_data.remote()
    elif command == "train":
        train_all.remote()
    elif command == "phase":
        train_phase.remote(phase=phase)
    elif command == "download":
        download_checkpoints.remote()
    else:
        print("""
Modal Training Commands:
========================

1. First, upload your data:
   modal run scripts/modal_train.py --command upload

2. Then upload files (run locally):
   modal volume put llm-training-data data/tokenized/ tokenized/
   modal volume put llm-training-data data/tokenizer.json tokenizer.json

3. Run training:
   modal run scripts/modal_train.py --command train

4. Download checkpoints:
   modal run scripts/modal_train.py --command download
   modal volume get llm-training-data checkpoints/ ./checkpoints_modal/
""")


if __name__ == "__main__":
    print("Run with: modal run scripts/modal_train.py")

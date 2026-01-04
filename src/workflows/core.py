"""Core idempotent logic for LLM training pipeline stages.

This module contains the actual implementation of each stage (Download, Clean, 
Tokenize, Train) in a way that is independent of the orchestrator (Temporal or Native).
"""
import os
import time
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Protocol
from datetime import datetime

# Import existing data utilities
from src.data.download import download_dataset as download_dataset_raw
from src.data.clean import clean_file as clean_file_raw
from src.tokenizer.tokenizer import train_tokenizer_from_files, TokenizerWrapper
from src.data.dataloader import CurriculumDataLoader
from src.training.trainer import Trainer
from src.model.config import ModelConfig, PHASE_CONFIGS
from src.model.llm import create_model

class ProgressCallback(Protocol):
    def __call__(self, message: str, progress: Optional[float] = None, details: Optional[Dict[str, Any]] = None) -> None:
        ...

def download_core(
    dataset: str, 
    data_dir: Path, 
    progress_fn: ProgressCallback
) -> Dict[str, Any]:
    """Core download logic with .part file idempotency."""
    raw_dir = data_dir / "raw"
    
    # Map dataset to download function
    if dataset == "wikipedia":
        target_dir = raw_dir / "phase1"
    elif dataset in ["pg19", "bookcorpus"]:
        target_dir = raw_dir / "phase2"
    elif dataset in ["pubmed", "philpapers"]:
        target_dir = raw_dir / "phase2b"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    def heartbeat_adapter(data):
        msg = data.get("message", f"Downloading {dataset}...")
        prog = data.get("progress")
        progress_fn(msg, progress=prog, details=data)

    # download_dataset_raw already handles .part shards and resumption
    dataset_name = "pubmed_abstracts" if dataset == "pubmed" else dataset
    output_dir = download_dataset_raw(
        dataset_name, 
        target_dir, 
        heartbeat_fn=heartbeat_adapter
    )
    
    # Calculate size
    size_bytes = sum(f.stat().st_size for f in output_dir.glob("shard_*.txt"))
    
    # Validate that we actually downloaded something
    if size_bytes == 0:
        return {
            "status": "failed",
            "dataset": dataset,
            "file_path": str(output_dir),
            "size_bytes": 0,
            "error": f"Download resulted in 0 bytes. Dataset '{dataset_name}' may be unavailable or require authentication.",
        }
    
    return {
        "status": "complete",
        "dataset": dataset,
        "file_path": str(output_dir),
        "size_bytes": size_bytes,
    }

def clean_core(
    dataset: str, 
    data_dir: Path, 
    progress_fn: ProgressCallback,
    resume_state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Core cleaning logic with .part file idempotency."""
    raw_paths = {
        "wikipedia": data_dir / "raw/phase1/wikipedia",
        "pg19": data_dir / "raw/phase2/pg19",
        "bookcorpus": data_dir / "raw/phase2/bookcorpus",
        "pubmed": data_dir / "raw/phase2b/pubmed_abstracts",
        "philpapers": data_dir / "raw/phase2b/philpapers",
    }
    
    raw_path = raw_paths.get(dataset)
    if not raw_path or not raw_path.exists():
        return {"status": "skipped", "reason": "no raw file"}
    
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    clean_path = processed_dir / f"{dataset}_clean.txt"
    part_path = clean_path.with_suffix(".txt.part")
    
    # Check if already done
    if clean_path.exists() and not part_path.exists():
         return {"status": "complete", "n_documents": "unknown (cached)"}

    # Resume logic
    skip_shards = 0
    if resume_state:
        skip_shards = resume_state.get("shards_processed", 0)

    def heartbeat_adapter(data):
        msg = data.get("message", f"Cleaning {dataset}...")
        prog = data.get("progress")
        # Ensure we pass shards_processed back to the orchestrator
        progress_fn(msg, progress=prog, details=data)

    # Clean to .part file
    n_docs = clean_file_raw(
        raw_path,
        part_path,
        min_length=100,
        deduplicate=True,
        heartbeat_fn=heartbeat_adapter,
        skip_shards=skip_shards
    )
    
    # Atomic rename
    part_path.rename(clean_path)
    
    return {
        "status": "complete",
        "dataset": dataset,
        "n_documents": n_docs,
        "output_path": str(clean_path),
    }

def train_tokenizer_core(
    data_dir: Path, 
    vocab_size: int, 
    progress_fn: ProgressCallback
) -> Dict[str, Any]:
    """Core tokenizer training logic."""
    processed_dir = data_dir / "processed"
    text_files = list(processed_dir.glob("*_clean.txt"))
    
    if not text_files:
        raise RuntimeError("No cleaned text files found for tokenizer training")
    
    tokenizer_path = data_dir / "tokenizer.json"
    progress_fn(f"Training tokenizer on {len(text_files)} files...")
    
    train_tokenizer_from_files(
        input_files=text_files,
        output_path=tokenizer_path,
        vocab_size=vocab_size,
    )
    
    return {
        "status": "complete",
        "vocab_size": vocab_size,
        "output_path": str(tokenizer_path),
    }

def tokenize_core(
    phase: str, 
    data_dir: Path, 
    progress_fn: ProgressCallback,
    resume_state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Core tokenization logic with .part file and atomic resume."""
    tokenizer_path = data_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise RuntimeError("Tokenizer not found")
        
    tokenizer = TokenizerWrapper(tokenizer_path)
    
    # Phase config (duplicated for core independence or we could import it)
    phase_config = {
        "1": {"files": ["wikipedia_clean.txt"], "target_tokens": 200_000_000},
        "2": {"files": ["pg19_clean.txt", "bookcorpus_clean.txt"], "target_tokens": 50_000_000},
        "2b": {"files": ["pubmed_clean.txt", "philpapers_clean.txt"], "target_tokens": 5_000_000},
    }
    config = phase_config.get(phase)
    if config is None:
        raise ValueError(f"Invalid phase: {phase}. Must be one of {list(phase_config.keys())}")
    
    processed_dir = data_dir / "processed"
    input_files = [processed_dir / f for f in config["files"] if (processed_dir / f).exists()]
    
    if not input_files:
        raise RuntimeError(
            f"No input files found for phase {phase}. "
            f"Expected files: {config['files']}. "
            f"Checked in: {processed_dir}"
        )
    
    tokenized_dir = data_dir / "tokenized"
    tokenized_dir.mkdir(parents=True, exist_ok=True)
    output_path = tokenized_dir / f"phase{phase}.npy"
    part_path = tokenized_dir / f"phase{phase}.part.bin"
    
    # If output already exists and is complete, skip
    if output_path.exists() and not part_path.exists():
        return {
            "status": "complete",
            "phase": phase,
            "n_tokens": "cached",
            "output_path": str(output_path),
        }
    
    # Resume logic
    last_doc_index = 0
    tokens_so_far = 0
    if resume_state:
        last_doc_index = resume_state.get("last_doc_index", 0)
        tokens_so_far = resume_state.get("tokens_so_far", 0)
    
    if last_doc_index > 0 and part_path.exists():
        expected_size = tokens_so_far * 4
        actual_size = part_path.stat().st_size
        if actual_size > expected_size:
            # Use r+b mode to allow truncation
            with open(part_path, "r+b") as f:
                f.truncate(expected_size)
    else:
        last_doc_index = 0
        tokens_so_far = 0
        if part_path.exists(): part_path.unlink()

    target_tokens = config["target_tokens"]
    current_doc_global = 0
    current_tokens = tokens_so_far
    
    from tqdm import tqdm
    
    with open(part_path, "ab") as bin_f:
        for input_file in input_files:
            with open(input_file, "r", encoding="utf-8") as f:
                # Use tqdm for document progress
                pbar = tqdm(f, desc=f"Tokenizing Phase {phase}", unit="docs")
                for i, line in enumerate(pbar):
                    if current_doc_global < last_doc_index:
                        current_doc_global += 1
                        continue
                        
                    line = line.strip()
                    if line:
                        tokens = tokenizer.encode(line, add_special_tokens=True)
                        if current_tokens + len(tokens) > target_tokens:
                            tokens = tokens[:target_tokens - current_tokens]
                        
                        if not tokens: break
                        
                        np.array(tokens, dtype=np.int32).tofile(bin_f)
                        current_tokens += len(tokens)
                        current_doc_global += 1
                        
                        # Update progress bar postfix
                        pbar.set_postfix({
                            "tokens": f"{current_tokens:,}/{target_tokens:,}",
                            "pct": f"{(current_tokens / target_tokens) * 100:.1f}%"
                        })
                        
                        if current_doc_global % 1000 == 0:
                            progress_fn(
                                f"Tokenizing Phase {phase}: {current_tokens:,}/{target_tokens:,} tokens",
                                progress=(current_tokens / target_tokens) * 100,
                                details={"last_doc_index": current_doc_global, "tokens_so_far": current_tokens}
                            )
                        
                        if current_tokens >= target_tokens: break
                pbar.close()
            if current_tokens >= target_tokens: break
    
    # Validate that we wrote some tokens before renaming
    if not part_path.exists():
        raise RuntimeError(f"Tokenization completed but no output file was created at {part_path}")
    
    if part_path.stat().st_size == 0:
        raise RuntimeError(
            f"Tokenization completed but output file is empty. "
            f"This may indicate all input documents were skipped or filtered out."
        )
    
    part_path.rename(output_path)
    
    return {
        "status": "complete",
        "phase": phase,
        "n_tokens": current_tokens,
        "output_path": str(output_path),
    }

def train_phase_core(
    phase: str,
    data_dir: Path,
    checkpoint_dir: Path,
    log_dir: Optional[Path],
    progress_fn: ProgressCallback,
    resume_step: int = 0
) -> Dict[str, Any]:
    """Core training logic for a single phase."""
    phase_map = {"1": "phase1_grammar", "2": "phase2_vocabulary", "2b": "phase2b_scientific"}
    steps_map = {"1": 50000, "2": 12500, "2b": 6250}
    
    phase_name = phase_map[phase]
    total_steps = steps_map[phase]
    config = PHASE_CONFIGS[phase_name]
    
    tokenized_path = data_dir / "tokenized" / f"phase{phase}.npy"
    if not tokenized_path.exists():
        raise RuntimeError(f"Tokenized data not found for phase {phase}")
    
    model_config = ModelConfig()
    model = create_model(model_config)
    
    # Find latest checkpoint for this phase
    import re
    def get_step_number(path):
        match = re.search(r'_step(\d+)\.pt$', str(path))
        return int(match.group(1)) if match else 0
    
    checkpoints = list(checkpoint_dir.glob(f"{phase_name}_step*.pt"))
    latest_ckpt = max(checkpoints, key=get_step_number) if checkpoints else None
    
    # Handle phase linking (load previous phase's final checkpoint for new phases)
    if not latest_ckpt:
        prev_phase_map = {
            "2": "phase1_grammar_final.pt",
            "2b": "phase2_vocabulary_final.pt"
        }
        prev_ckpt_name = prev_phase_map.get(phase)
        
        if prev_ckpt_name:
            prev_ckpt_path = checkpoint_dir / prev_ckpt_name
            if prev_ckpt_path.exists():
                print(f"Initializing Phase {phase} from previous phase checkpoint: {prev_ckpt_name}")
                checkpoint = torch.load(prev_ckpt_path, map_location="cpu", weights_only=False)
                # Load ONLY model weights - we want a fresh optimizer/scheduler for the new phase
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                print(f"Loaded weights from {prev_ckpt_name}")
            else:
                 print(f"Warning: Previous phase checkpoint {prev_ckpt_name} not found. Starting {phase} from scratch.")
    
    data_loader = CurriculumDataLoader(
        data_path=tokenized_path,
        batch_size=config.micro_batch_size,
        context_schedule=config.context_schedule,
        num_workers=config.num_workers,
    )
    
    trainer = Trainer(
        model=model,
        train_config=config,
        data_loader=data_loader,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir / phase_name if log_dir else None,
    )
    
    # Load checkpoint using proper method (restores optimizer + scheduler state)
    if latest_ckpt:
        trainer.load_checkpoint(latest_ckpt)
    elif resume_step > 0:
        # If no checkpoint but resume_step is set (from DB), just set the step counter
        trainer.state.global_step = resume_step
        print(f"Resuming training from step {resume_step} (no checkpoint found)")
    
    # Patch for reporting
    original_opt_step = trainer.optimizer_step
    steps_done = [trainer.state.global_step]
    
    def patched_opt_step():
        res = original_opt_step()
        steps_done[0] += 1
        curr = steps_done[0]
        if curr % 100 == 0:
            loss = trainer.state.best_loss if trainer.state.best_loss != float('inf') else 0.0
            progress_fn(
                f"Phase {phase}: step {curr}/{total_steps}, loss={loss:.4f}",
                progress=(curr / total_steps) * 100,
                details={"step": curr, "loss": loss}
            )
        return res
        
    trainer.optimizer_step = patched_opt_step
    
    metrics = trainer.train(total_steps=total_steps, phase_name=phase_name)
    
    final_ckpt = checkpoint_dir / f"{phase_name}_final.pt"
    
    return {
        "status": "complete",
        "phase": phase,
        "steps": total_steps,
        "loss": metrics.get("final_loss"),
        "checkpoint_path": str(final_ckpt),
    }

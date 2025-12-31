#!/usr/bin/env python3
"""Data preparation script.

Downloads, cleans, tokenizes, and prepares data for training.

Usage:
    # Full preparation (all phases)
    python scripts/prepare_data.py --output-dir data
    
    # Specific phase
    python scripts/prepare_data.py --phase 1 --output-dir data
    
    # Sample for testing
    python scripts/prepare_data.py --max-samples 1000 --output-dir data/test
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.download import (
    download_all_phase1,
    download_all_phase2,
    download_all_phase2b,
    stream_dataset,
)
from src.data.clean import clean_file, sample_to_token_budget
from src.tokenizer.tokenizer import train_tokenizer_from_files, TokenizerWrapper
from src.data.dataloader import save_tokenized_data


def prepare_phase1(output_dir: Path, max_samples: int = None):
    """Download and prepare Phase 1 data (Wikipedia)."""
    print("\n" + "="*60)
    print("Phase 1: Downloading Wikipedia")
    print("="*60)
    
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    
    # Download
    files = download_all_phase1(raw_dir, max_samples)
    
    # Clean
    for name, raw_path in files.items():
        clean_path = processed_dir / f"{name}_clean.txt"
        print(f"Cleaning {name}...")
        n_docs = clean_file(raw_path, clean_path)
        print(f"  Cleaned {n_docs} documents")
    
    return list((processed_dir).glob("*_clean.txt"))


def prepare_phase2(output_dir: Path, max_samples: int = None):
    """Download and prepare Phase 2 data (PG19, BookCorpus)."""
    print("\n" + "="*60)
    print("Phase 2: Downloading PG19 + BookCorpus")
    print("="*60)
    
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    
    # Download
    files = download_all_phase2(raw_dir, max_samples)
    
    # Clean
    for name, raw_path in files.items():
        clean_path = processed_dir / f"{name}_clean.txt"
        print(f"Cleaning {name}...")
        n_docs = clean_file(raw_path, clean_path)
        print(f"  Cleaned {n_docs} documents")
    
    return list((processed_dir).glob("*_clean.txt"))


def prepare_phase2b(output_dir: Path, max_samples: int = None):
    """Download and prepare Phase 2b data (PubMed, PhilPapers)."""
    print("\n" + "="*60)
    print("Phase 2b: Downloading PubMed Abstracts + PhilPapers")
    print("="*60)
    
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    
    # Download
    files = download_all_phase2b(raw_dir, max_samples)
    
    # Clean
    for name, raw_path in files.items():
        clean_path = processed_dir / f"{name}_clean.txt"
        print(f"Cleaning {name}...")
        n_docs = clean_file(raw_path, clean_path)
        print(f"  Cleaned {n_docs} documents")
    
    return list((processed_dir).glob("*_clean.txt"))


def train_tokenizer(text_files: list, output_dir: Path, vocab_size: int = 32000):
    """Train tokenizer on all text files."""
    print("\n" + "="*60)
    print("Training Tokenizer")
    print("="*60)
    
    tokenizer_path = output_dir / "tokenizer.json"
    
    wrapper = train_tokenizer_from_files(
        input_files=text_files,
        output_path=tokenizer_path,
        vocab_size=vocab_size,
    )
    
    return wrapper


def tokenize_data(
    tokenizer: TokenizerWrapper,
    input_files: list,
    output_path: Path,
    target_tokens: int = None,
):
    """Tokenize text files and save as numpy array."""
    print(f"\nTokenizing to {output_path}...")
    
    all_tokens = []
    
    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = tokenizer.encode(line, add_special_tokens=True)
                    all_tokens.extend(tokens)
                    
                    if target_tokens and len(all_tokens) >= target_tokens:
                        break
        
        if target_tokens and len(all_tokens) >= target_tokens:
            break
    
    # Truncate to target if specified
    if target_tokens:
        all_tokens = all_tokens[:target_tokens]
    
    # Save
    save_tokenized_data(all_tokens, output_path)
    
    return len(all_tokens)


def main():
    parser = argparse.ArgumentParser(description="Prepare data for LLM training")
    
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory")
    parser.add_argument("--phase", type=str, choices=["1", "2", "2b", "all"], default="all",
                        help="Phase to prepare")
    parser.add_argument("--max-samples", type=int, help="Max samples (for testing)")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Tokenizer vocab size")
    parser.add_argument("--skip-download", action="store_true", help="Skip download step")
    parser.add_argument("--skip-tokenizer", action="store_true", help="Skip tokenizer training")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_text_files = []
    
    # Download and clean
    if not args.skip_download:
        if args.phase in ["1", "all"]:
            files = prepare_phase1(output_dir, args.max_samples)
            all_text_files.extend(files)
        
        if args.phase in ["2", "all"]:
            files = prepare_phase2(output_dir, args.max_samples)
            all_text_files.extend(files)
        
        if args.phase in ["2b", "all"]:
            files = prepare_phase2b(output_dir, args.max_samples)
            all_text_files.extend(files)
    else:
        # Use existing files
        all_text_files = list((output_dir / "processed").glob("*_clean.txt"))
    
    if not all_text_files:
        print("No text files found. Run without --skip-download first.")
        return
    
    # Train tokenizer
    tokenizer_path = output_dir / "tokenizer.json"
    if not args.skip_tokenizer:
        tokenizer = train_tokenizer(all_text_files, output_dir, args.vocab_size)
    else:
        tokenizer = TokenizerWrapper(tokenizer_path)
    
    # Tokenize each phase
    tokenized_dir = output_dir / "tokenized"
    tokenized_dir.mkdir(exist_ok=True)
    
    phase_files = {
        "phase1": [f for f in all_text_files if "wikipedia" in str(f)],
        "phase2": [f for f in all_text_files if "pg19" in str(f) or "bookcorpus" in str(f)],
        "phase2b": [f for f in all_text_files if "pubmed" in str(f) or "philpapers" in str(f)],
    }
    
    # Token targets (aligned with Training_Steps.md)
    phase_tokens = {
        "phase1": 200_000_000,  # 200M tokens
        "phase2": 50_000_000,   # 50M tokens
        "phase2b": 5_000_000,   # 5M tokens (3M PubMed + 2M PhilPapers)
    }
    
    for phase_name, files in phase_files.items():
        if files:
            output_path = tokenized_dir / f"{phase_name}.npy"
            target = phase_tokens.get(phase_name)
            n_tokens = tokenize_data(tokenizer, files, output_path, target)
            print(f"  {phase_name}: {n_tokens:,} tokens")
    
    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Tokenized data: {tokenized_dir}")


if __name__ == "__main__":
    main()

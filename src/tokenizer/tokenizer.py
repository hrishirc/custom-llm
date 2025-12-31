"""Custom tokenizer training and usage.

Trains a Unigram tokenizer on the combined training corpora.
Vocabulary size: 32k with special tokens.
"""

from pathlib import Path
from typing import Iterator, List, Optional, Union
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFKC


# Special tokens
SPECIAL_TOKENS = {
    "pad": "<PAD>",
    "bos": "<BOS>",
    "eos": "<EOS>",
    "unk": "<UNK>",
}


def create_tokenizer(vocab_size: int = 32000) -> Tokenizer:
    """Create an uninitialized Unigram tokenizer.
    
    Args:
        vocab_size: Target vocabulary size
    
    Returns:
        Tokenizer ready for training
    """
    # Initialize with Unigram model
    tokenizer = Tokenizer(models.Unigram())
    
    # Normalizer: NFKC Unicode normalization
    tokenizer.normalizer = NFKC()
    
    # Pre-tokenizer: Metaspace (preserves whitespace as U+2581)
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    
    # Decoder
    tokenizer.decoder = decoders.Metaspace()
    
    return tokenizer


def train_tokenizer(
    tokenizer: Tokenizer,
    files: List[Path] = None,
    iterator: Iterator[str] = None,
    vocab_size: int = 32000,
) -> Tokenizer:
    """Train the tokenizer on text data.
    
    Args:
        tokenizer: Tokenizer to train
        files: List of text file paths
        iterator: Iterator yielding text strings (alternative to files)
        vocab_size: Target vocabulary size
    
    Returns:
        Trained tokenizer
    """
    # Create trainer
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=list(SPECIAL_TOKENS.values()),
        unk_token=SPECIAL_TOKENS["unk"],
    )
    
    # Train
    if files:
        tokenizer.train(files=[str(f) for f in files], trainer=trainer)
    elif iterator:
        tokenizer.train_from_iterator(iterator, trainer=trainer)
    else:
        raise ValueError("Must provide either files or iterator")
    
    # Add post-processor for BOS/EOS
    bos_id = tokenizer.token_to_id(SPECIAL_TOKENS["bos"])
    eos_id = tokenizer.token_to_id(SPECIAL_TOKENS["eos"])
    
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{SPECIAL_TOKENS['bos']}:0 $A:0 {SPECIAL_TOKENS['eos']}:0",
        special_tokens=[
            (SPECIAL_TOKENS["bos"], bos_id),
            (SPECIAL_TOKENS["eos"], eos_id),
        ],
    )
    
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, output_path: Path):
    """Save tokenizer to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    print(f"Tokenizer saved to {output_path}")


def load_tokenizer(path: Path) -> Tokenizer:
    """Load tokenizer from disk."""
    return Tokenizer.from_file(str(path))


class TokenizerWrapper:
    """Wrapper for convenient tokenizer usage.
    
    Provides a simple interface for encoding and decoding text.
    """
    
    def __init__(self, tokenizer_path: Union[str, Path]):
        self.tokenizer = load_tokenizer(Path(tokenizer_path))
        
        # Get special token IDs (validate they exist)
        self.pad_token_id = self.tokenizer.token_to_id(SPECIAL_TOKENS["pad"])
        self.bos_token_id = self.tokenizer.token_to_id(SPECIAL_TOKENS["bos"])
        self.eos_token_id = self.tokenizer.token_to_id(SPECIAL_TOKENS["eos"])
        self.unk_token_id = self.tokenizer.token_to_id(SPECIAL_TOKENS["unk"])
        
        # Validate all special tokens exist
        missing_tokens = []
        if self.pad_token_id is None:
            missing_tokens.append(SPECIAL_TOKENS["pad"])
        if self.bos_token_id is None:
            missing_tokens.append(SPECIAL_TOKENS["bos"])
        if self.eos_token_id is None:
            missing_tokens.append(SPECIAL_TOKENS["eos"])
        if self.unk_token_id is None:
            missing_tokens.append(SPECIAL_TOKENS["unk"])
        
        if missing_tokens:
            raise ValueError(
                f"Tokenizer is missing required special tokens: {missing_tokens}. "
                f"This tokenizer may be corrupted or incompatible."
            )
        
        self.vocab_size = self.tokenizer.get_vocab_size()
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """Encode text to token IDs."""
        # Temporarily disable post-processor if not adding special tokens
        if not add_special_tokens:
            original_post = self.tokenizer.post_processor
            self.tokenizer.post_processor = None
        
        encoding = self.tokenizer.encode(text)
        
        if not add_special_tokens:
            self.tokenizer.post_processor = original_post
        
        ids = encoding.ids
        
        # Truncate if needed
        if max_length and len(ids) > max_length:
            ids = ids[:max_length]
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    
    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
    ) -> List[List[int]]:
        """Encode multiple texts."""
        return [self.encode(t, add_special_tokens) for t in texts]
    
    def __call__(self, text: str, **kwargs) -> List[int]:
        """Shorthand for encode."""
        return self.encode(text, **kwargs)


def train_tokenizer_from_files(
    input_files: List[Path],
    output_path: Path,
    vocab_size: int = 32000,
    max_samples: int = 5_000_000,  # Sample ~5M lines to avoid memory issues
) -> TokenizerWrapper:
    """Complete tokenizer training workflow with memory-efficient sampling.
    
    Args:
        input_files: List of text files for training
        output_path: Path to save tokenizer
        vocab_size: Target vocabulary size
        max_samples: Maximum number of lines to sample (to avoid memory issues)
    
    Returns:
        TokenizerWrapper for the trained tokenizer
    """
    import random
    
    print(f"Training tokenizer on {len(input_files)} files...")
    print(f"Target vocabulary size: {vocab_size}")
    print(f"Sampling up to {max_samples:,} lines to avoid memory issues...")
    
    # Filter out empty files
    valid_files = [f for f in input_files if f.stat().st_size > 0]
    if not valid_files:
        raise RuntimeError("All input files are empty")
    
    print(f"Using {len(valid_files)} non-empty files")
    
    def sampled_iterator():
        """Memory-efficient iterator that samples lines uniformly from all files."""
        # First pass: count approximate total lines
        total_size = sum(f.stat().st_size for f in valid_files)
        # Estimate ~100 bytes per line on average
        estimated_lines = total_size // 100
        
        # Calculate sampling probability
        sample_prob = min(1.0, max_samples / max(estimated_lines, 1))
        print(f"Estimated {estimated_lines:,} lines, sampling probability: {sample_prob:.4f}")
        
        lines_yielded = 0
        for file_path in valid_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line and len(line) > 50:  # Skip short lines
                            # Reservoir-ish sampling
                            if random.random() < sample_prob:
                                lines_yielded += 1
                                yield line
                                if lines_yielded >= max_samples:
                                    print(f"Reached {max_samples:,} samples, stopping")
                                    return
            except Exception as e:
                print(f"Warning: Error reading {file_path}: {e}")
                continue
        
        print(f"Sampled {lines_yielded:,} lines total")
    
    # Create and train using iterator (memory-efficient)
    tokenizer = create_tokenizer(vocab_size)
    tokenizer = train_tokenizer(tokenizer, iterator=sampled_iterator(), vocab_size=vocab_size)
    
    # Save
    save_tokenizer(tokenizer, output_path)
    
    # Return wrapper
    return TokenizerWrapper(output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train custom tokenizer")
    parser.add_argument("--input", type=str, nargs="+", required=True, help="Input text files")
    parser.add_argument("--output", type=str, required=True, help="Output tokenizer path")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    
    args = parser.parse_args()
    
    wrapper = train_tokenizer_from_files(
        input_files=[Path(f) for f in args.input],
        output_path=Path(args.output),
        vocab_size=args.vocab_size,
    )
    
    # Test
    test_text = "The quick brown fox jumps over the lazy dog."
    ids = wrapper.encode(test_text)
    decoded = wrapper.decode(ids)
    
    print(f"\nTest encoding:")
    print(f"  Input: {test_text}")
    print(f"  Tokens: {ids}")
    print(f"  Decoded: {decoded}")
    print(f"  Vocab size: {wrapper.vocab_size}")

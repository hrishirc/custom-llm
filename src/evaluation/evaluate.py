"""Evaluation and generation utilities for the LLM.

Implements:
- Perplexity calculation
- Validation loss evaluation
- Sample generation with logging
- Behavioral tests
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
from pathlib import Path

from ..model.llm import LLM
from ..tokenizer.tokenizer import TokenizerWrapper


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss.
    
    Perplexity = exp(loss)
    Lower is better. Random guessing = vocab_size.
    """
    return math.exp(loss)


@torch.no_grad()
def evaluate_loss(
    model: LLM,
    dataloader,
    max_batches: Optional[int] = None,
    device: torch.device = None,
) -> Dict[str, float]:
    """Evaluate model on a dataset.
    
    Args:
        model: The LLM to evaluate
        dataloader: DataLoader yielding batches
        max_batches: Maximum batches to evaluate (for speed)
        device: Device to use
    
    Returns:
        Dictionary with loss, perplexity, and token count
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    total_loss = 0.0
    total_tokens = 0
    batch_count = 0
    
    for batch in dataloader:
        # Move to device
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        # Count tokens (excluding padding)
        if attention_mask is not None:
            n_tokens = attention_mask.sum().item()
        else:
            n_tokens = input_ids.numel()
        
        total_loss += outputs["loss"].item() * n_tokens
        total_tokens += n_tokens
        batch_count += 1
        
        if max_batches and batch_count >= max_batches:
            break
    
    avg_loss = total_loss / max(total_tokens, 1)
    
    return {
        "loss": avg_loss,
        "perplexity": compute_perplexity(avg_loss),
        "total_tokens": total_tokens,
        "batches_evaluated": batch_count,
    }


@torch.no_grad()
def generate_samples(
    model: LLM,
    tokenizer: TokenizerWrapper,
    prompts: List[str],
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: torch.device = None,
) -> List[Dict[str, str]]:
    """Generate text samples from prompts.
    
    Args:
        model: The LLM to use
        tokenizer: Tokenizer for encoding/decoding
        prompts: List of prompt strings
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus sampling threshold
        device: Device to use
    
    Returns:
        List of dicts with prompt, generated, and full text
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    results = []
    
    for prompt in prompts:
        # Encode prompt
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], device=device)
        
        # Generate
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        
        # Decode
        generated_ids = output_ids[0].tolist()
        full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_text = tokenizer.decode(generated_ids[len(input_ids):], skip_special_tokens=True)
        
        results.append({
            "prompt": prompt,
            "generated": generated_text,
            "full": full_text,
        })
    
    return results


def log_samples(
    samples: List[Dict[str, str]],
    step: int,
    output_dir: Optional[Path] = None,
    writer=None,
):
    """Log generated samples to console and optional TensorBoard.
    
    Args:
        samples: List of sample dicts from generate_samples
        step: Training step number
        output_dir: Optional directory to save samples
        writer: Optional TensorBoard SummaryWriter
    """
    print(f"\n{'='*60}")
    print(f"Generated Samples (Step {step})")
    print(f"{'='*60}")
    
    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i+1} ---")
        print(f"Prompt: {sample['prompt']}")
        print(f"Generated: {sample['generated']}")
    
    print(f"{'='*60}\n")
    
    # Save to file if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sample_file = output_dir / f"samples_step{step}.txt"
        with open(sample_file, "w", encoding="utf-8") as f:
            for i, sample in enumerate(samples):
                f.write(f"--- Sample {i+1} ---\n")
                f.write(f"Prompt: {sample['prompt']}\n")
                f.write(f"Generated: {sample['generated']}\n")
                f.write(f"Full: {sample['full']}\n\n")
    
    # Log to TensorBoard if writer provided
    if writer:
        text = "\n\n".join([
            f"**Prompt:** {s['prompt']}\n\n**Generated:** {s['generated']}"
            for s in samples
        ])
        writer.add_text("samples", text, step)


# Default prompts for behavioral testing
DEFAULT_PROMPTS = [
    "The quick brown fox",
    "Once upon a time",
    "The most important thing about",
    "In the beginning",
    "Science has shown that",
]


def run_behavioral_tests(
    model: LLM,
    tokenizer: TokenizerWrapper,
    step: int,
    device: torch.device = None,
    output_dir: Optional[Path] = None,
    writer=None,
) -> List[Dict[str, str]]:
    """Run standard behavioral tests and log results.
    
    Tests:
    - Short completion (basic fluency)
    - Longer completion (coherence)
    - Various prompt styles
    
    Args:
        model: The LLM to test
        tokenizer: Tokenizer for encoding/decoding
        step: Training step number
        device: Device to use
        output_dir: Optional output directory
        writer: Optional TensorBoard writer
    
    Returns:
        List of sample results
    """
    samples = generate_samples(
        model=model,
        tokenizer=tokenizer,
        prompts=DEFAULT_PROMPTS,
        max_new_tokens=50,
        temperature=0.8,
        device=device,
    )
    
    log_samples(samples, step, output_dir, writer)
    
    return samples


class Evaluator:
    """Evaluation helper for training loop integration.
    
    Combines loss evaluation and sample generation.
    """
    
    def __init__(
        self,
        model: LLM,
        tokenizer: TokenizerWrapper,
        eval_dataloader=None,
        prompts: List[str] = None,
        output_dir: Optional[Path] = None,
        writer=None,
        device: torch.device = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataloader = eval_dataloader
        self.prompts = prompts or DEFAULT_PROMPTS
        self.output_dir = Path(output_dir) if output_dir else None
        self.writer = writer
        self.device = device or next(model.parameters()).device
    
    def evaluate(
        self,
        step: int,
        max_batches: int = 50,
        generate: bool = True,
    ) -> Dict[str, Any]:
        """Run full evaluation.
        
        Args:
            step: Training step number
            max_batches: Max batches for loss evaluation
            generate: Whether to generate samples
        
        Returns:
            Dictionary with all metrics
        """
        results = {}
        
        # Loss evaluation
        if self.eval_dataloader:
            loss_metrics = evaluate_loss(
                self.model,
                self.eval_dataloader,
                max_batches=max_batches,
                device=self.device,
            )
            results.update(loss_metrics)
            
            if self.writer:
                self.writer.add_scalar("eval/loss", loss_metrics["loss"], step)
                self.writer.add_scalar("eval/perplexity", loss_metrics["perplexity"], step)
            
            print(f"Step {step} - Eval Loss: {loss_metrics['loss']:.4f}, "
                  f"Perplexity: {loss_metrics['perplexity']:.2f}")
        
        # Sample generation
        if generate:
            samples = generate_samples(
                self.model,
                self.tokenizer,
                self.prompts,
                device=self.device,
            )
            log_samples(samples, step, self.output_dir, self.writer)
            results["samples"] = samples
        
        return results

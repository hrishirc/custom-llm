"""Training loop and utilities for the LLM.

Implements:
- Gradient accumulation
- Mixed precision (BF16)
- Layer freezing
- Context curriculum
- Checkpointing
- Logging
"""

import os
import math
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..model.config import ModelConfig, TrainingConfig
from ..model.llm import LLM
from ..data.dataloader import CurriculumDataLoader
from .metrics_logger import MetricsLogger, TrainingMetrics

# Import Adafactor from transformers
try:
    from transformers import Adafactor
    ADAFACTOR_AVAILABLE = True
except ImportError:
    ADAFACTOR_AVAILABLE = False


@dataclass
class TrainerState:
    """Tracks training progress."""
    global_step: int = 0
    epoch: int = 0
    total_tokens: int = 0
    best_loss: float = float("inf")
    
    def progress(self, total_steps: int) -> float:
        """Training progress as fraction [0, 1]."""
        return self.global_step / max(total_steps, 1)


class Trainer:
    """Trainer for the LLM.
    
    Implements all training optimizations:
    - torch.compile for speedup
    - BF16 mixed precision
    - Gradient accumulation
    - Layer freezing
    - Context curriculum
    - Checkpointing
    """
    
    def __init__(
        self,
        model: LLM,
        train_config: TrainingConfig,
        data_loader: CurriculumDataLoader,
        checkpoint_dir: Path,
        log_dir: Optional[Path] = None,
    ):
        # Disable tokenizers parallelism to avoid warnings when DataLoader forks workers
        # This must be set before DataLoader creates worker processes
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self.model = model
        self.config = train_config
        self.data_loader = data_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        self.device = self._get_device(train_config.device)
        self.model = self.model.to(self.device)
        
        # Compile model for speedup (PyTorch 2.0+)
        # Note: torch.compile is not supported on Python 3.14+ as of PyTorch 2.x
        if train_config.compile_model and hasattr(torch, "compile"):
            try:
                print("Compiling model with torch.compile...")
                self.model = torch.compile(self.model, mode="default")  # "default" more stable for MPS
            except RuntimeError as e:
                if "Python 3.14" in str(e):
                    print(f"Warning: torch.compile not supported on this Python version, skipping compilation")
                else:
                    raise
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = None  # Created when we know total steps
        
        # Mixed precision
        self.use_amp = train_config.dtype in ("float16", "bfloat16")
        self.amp_dtype = torch.bfloat16 if train_config.dtype == "bfloat16" else torch.float16
        
        # GradScaler only needed for FP16 (not BF16)
        self.scaler = GradScaler() if train_config.dtype == "float16" else None
        
        # State
        self.state = TrainerState()
        
        # Logging - use new comprehensive MetricsLogger
        self.log_dir = Path(log_dir) if log_dir else None
        self.metrics_logger = None
        if self.log_dir:
            self.metrics_logger = MetricsLogger(
                log_dir=self.log_dir,
                tensorboard=True,
                csv_logging=True,
                log_interval=train_config.logging_steps,
                detailed_interval=train_config.logging_steps * 10,
            )
        
        # Legacy TensorBoard writer for backward compatibility
        self.writer = SummaryWriter(log_dir) if log_dir else None
    
    def _get_device(self, device_str: str) -> torch.device:
        """Get torch device."""
        # Handle 'auto' detection
        if device_str == "auto":
            device_str = self.config._get_device()
        
        if device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_str == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create Adafactor optimizer (lower memory than Adam).
        
        Adafactor uses factored second moment estimation which reduces
        memory usage by ~20% compared to Adam/AdamW.
        """
        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        # Use Adafactor if available (recommended for Transformers)
        if ADAFACTOR_AVAILABLE:
            optimizer = Adafactor(
                param_groups,
                lr=self.config.learning_rate,
                scale_parameter=False,  # We handle LR ourselves
                relative_step=False,    # We handle scheduling ourselves
                warmup_init=False,      # We handle warmup ourselves
            )
            print("Using Adafactor optimizer (lower memory)")
        else:
            # Fallback to AdamW
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
            )
            print("Using AdamW optimizer (Adafactor not available)")
        
        return optimizer
    
    def _create_scheduler(self, total_steps: int):
        """Create cosine learning rate scheduler with warmup."""
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                # Linear warmup
                return step / max(warmup_steps, 1)
            else:
                # Cosine decay
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                min_lr_ratio = self.config.min_learning_rate / self.config.learning_rate
                return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _update_layer_freezing(self, progress: float):
        """Update which layers are frozen based on progress."""
        n_freeze = 0
        for threshold, layers in sorted(self.config.freeze_schedule.items()):
            if progress >= threshold:
                n_freeze = layers
        
        if hasattr(self.model, "freeze_layers"):
            self.model.freeze_layers(n_freeze)
        elif hasattr(self.model, "_orig_mod"):
            # Handle compiled model
            self.model._orig_mod.freeze_layers(n_freeze)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step.
        
        Returns:
            Loss value
        """
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass with autocast
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
            )
            loss = outputs["loss"]
            
            # Scale for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        if self.scheduler:
            self.scheduler.step()
    
    def train(
        self,
        total_steps: int,
        eval_fn: Optional[Callable] = None,
        phase_name: str = "train",
    ) -> Dict[str, float]:
        """Main training loop.
        
        Args:
            total_steps: Total optimization steps
            eval_fn: Optional evaluation function
            phase_name: Name for checkpoints/logs
        
        Returns:
            Training metrics
        """
        print(f"\n{'='*60}")
        print(f"Starting training: {phase_name}")
        print(f"Total steps: {total_steps}")
        print(f"Effective batch size: {self.config.effective_batch_size}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        # Create scheduler
        self._create_scheduler(total_steps)
        
        # Training loop
        running_loss = 0.0
        step_times = []
        
        pbar = tqdm(total=total_steps, desc=phase_name)
        
        while self.state.global_step < total_steps:
            # Update progress-based settings
            progress = self.state.progress(total_steps)
            self._update_layer_freezing(progress)
            
            # Get data loader for current context length
            loader = self.data_loader.get_loader(progress)
            data_iter = iter(loader)
            
            while self.state.global_step < total_steps:
                step_start = time.time()
                
                # Accumulation loop
                for accum_step in range(self.config.gradient_accumulation_steps):
                    try:
                        micro_batch = next(data_iter)
                    except StopIteration:
                        # Refresh iterator when exhausted
                        data_iter = iter(loader)
                        micro_batch = next(data_iter)
                    
                    loss = self.train_step(micro_batch)
                    running_loss += loss / self.config.gradient_accumulation_steps
                
                # Optimizer step
                self.optimizer_step()
                self.state.global_step += 1
                
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                # Logging
                if self.state.global_step % self.config.logging_steps == 0:
                    avg_loss = running_loss / self.config.logging_steps
                    avg_time = sum(step_times[-100:]) / len(step_times[-100:])
                    lr = self.scheduler.get_last_lr()[0]
                    
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}",
                        "s/step": f"{avg_time:.2f}",
                    })
                    
                    # Comprehensive metrics logging
                    if self.metrics_logger:
                        # Compute current context length from data loader
                        context_len = self.data_loader.get_context_length(progress)
                        
                        # Determine frozen layers count
                        n_frozen = 0
                        for threshold, layers in sorted(self.config.freeze_schedule.items()):
                            if progress >= threshold:
                                n_frozen = layers
                        
                        # Compute detailed metrics every 10x logging interval
                        is_detailed = (self.state.global_step % (self.config.logging_steps * 10) == 0)
                        
                        metrics = self.metrics_logger.compute_metrics(
                            model=self.model,
                            loss=avg_loss,
                            step=self.state.global_step,
                            learning_rate=lr,
                            step_time=avg_time,
                            tokens_processed=self.config.effective_batch_size * context_len,
                            batch_size=self.config.effective_batch_size,
                            phase=phase_name,
                            frozen_layers=n_frozen,
                            context_length=context_len,
                            detailed=is_detailed,
                        )
                        self.metrics_logger.log(metrics)
                    
                    # Legacy TensorBoard (for backward compatibility)
                    if self.writer:
                        self.writer.add_scalar("train/loss", avg_loss, self.state.global_step)
                        self.writer.add_scalar("train/lr", lr, self.state.global_step)
                    
                    running_loss = 0.0
                
                # Checkpointing
                if self.state.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"{phase_name}_step{self.state.global_step}")
                
                # Evaluation
                if eval_fn and self.state.global_step % self.config.eval_steps == 0:
                    eval_loss = eval_fn(self.model)
                    if self.writer:
                        self.writer.add_scalar("eval/loss", eval_loss, self.state.global_step)
                    
                    if eval_loss < self.state.best_loss:
                        self.state.best_loss = eval_loss
                        self.save_checkpoint(f"{phase_name}_best")
                
                pbar.update(1)
                
                if self.state.global_step >= total_steps:
                    break
        
        pbar.close()
        
        # Final checkpoint
        self.save_checkpoint(f"{phase_name}_final")
        
        return {
            "final_loss": running_loss / max(self.config.logging_steps, 1),
            "best_loss": self.state.best_loss,
            "total_steps": self.state.global_step,
        }
    
    def save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        
        # Handle compiled model
        model_to_save = self.model
        if hasattr(self.model, "_orig_mod"):
            model_to_save = self.model._orig_mod
        
        checkpoint = {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "state": {
                "global_step": self.state.global_step,
                "epoch": self.state.epoch,
                "total_tokens": self.state.total_tokens,
                "best_loss": self.state.best_loss,
            },
            "config": self.config,
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Cleanup old checkpoints - keep only last 30
        self._cleanup_old_checkpoints(keep_last=30)
    
    def _cleanup_old_checkpoints(self, keep_last: int = 30):
        """Delete old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_last: Number of most recent checkpoints to keep
        """
        import os
        
        # Get all checkpoint files (exclude 'best' and 'final' checkpoints)
        all_ckpts = list(self.checkpoint_dir.glob("*_step*.pt"))
        
        if len(all_ckpts) <= keep_last:
            return
        
        # Sort by modification time (oldest first)
        all_ckpts.sort(key=lambda p: p.stat().st_mtime)
        
        # Delete oldest checkpoints
        to_delete = all_ckpts[:-keep_last]
        for ckpt in to_delete:
            try:
                ckpt.unlink()
                print(f"Deleted old checkpoint: {ckpt.name}")
            except Exception as e:
                print(f"Warning: Failed to delete {ckpt}: {e}")
    
    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Handle compiled model
        model_to_load = self.model
        if hasattr(self.model, "_orig_mod"):
            model_to_load = self.model._orig_mod
        
        model_to_load.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if checkpoint.get("scheduler_state_dict") and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        state = checkpoint["state"]
        self.state.global_step = state["global_step"]
        self.state.epoch = state["epoch"]
        self.state.total_tokens = state["total_tokens"]
        self.state.best_loss = state["best_loss"]
        
        print(f"Checkpoint loaded: {path}")
        print(f"Resuming from step {self.state.global_step}")

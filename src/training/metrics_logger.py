"""Comprehensive training metrics logging.

Logs training metrics to multiple backends:
- TensorBoard (default, local)
- Weights & Biases (optional, cloud)
- CSV files (simple, queryable)
- Console (human-readable)

Metrics tracked:
- Loss and learning rate
- Gradient norms (total and per-layer)
- Weight norms and updates
- Memory usage
- Throughput (tokens/sec, samples/sec)
"""
import csv
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Snapshot of training metrics at a given step."""
    step: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Loss metrics
    loss: float = 0.0
    learning_rate: float = 0.0
    
    # Gradient metrics
    grad_norm_total: float = 0.0
    grad_norm_clipped: bool = False
    grad_norms_by_layer: Dict[str, float] = field(default_factory=dict)
    
    # Weight metrics  
    weight_norm_total: float = 0.0
    weight_norms_by_layer: Dict[str, float] = field(default_factory=dict)
    weight_update_ratio: float = 0.0  # ||delta_w|| / ||w||
    
    # Performance metrics
    step_time_ms: float = 0.0
    tokens_per_sec: float = 0.0
    samples_per_sec: float = 0.0
    
    # Memory metrics (MB)
    memory_allocated_mb: float = 0.0
    memory_reserved_mb: float = 0.0
    
    # Training state
    phase: str = ""
    frozen_layers: int = 0
    context_length: int = 0


class MetricsLogger:
    """Multi-backend training metrics logger.
    
    Usage:
        metrics_logger = MetricsLogger(log_dir="logs/run1")
        
        for step in training_loop:
            # ... training code ...
            metrics = metrics_logger.compute_metrics(model, loss, step)
            metrics_logger.log(metrics)
    """
    
    def __init__(
        self,
        log_dir: Path,
        tensorboard: bool = True,
        csv_logging: bool = True,
        wandb_project: Optional[str] = None,
        log_interval: int = 10,
        detailed_interval: int = 100,
    ):
        """Initialize metrics logger.
        
        Args:
            log_dir: Directory for log files
            tensorboard: Enable TensorBoard logging
            csv_logging: Enable CSV file logging
            wandb_project: W&B project name (None to disable)
            log_interval: Steps between basic logging
            detailed_interval: Steps between detailed per-layer logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_interval = log_interval
        self.detailed_interval = detailed_interval
        
        # TensorBoard
        self.tb_writer = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(self.log_dir / "tensorboard")
        
        # CSV logging
        self.csv_path = None
        self.csv_file = None
        self.csv_writer = None
        if csv_logging:
            self.csv_path = self.log_dir / "metrics.csv"
            self._init_csv()
        
        # Weights & Biases
        self.wandb_run = None
        if wandb_project:
            try:
                import wandb
                self.wandb_run = wandb.init(project=wandb_project, dir=str(self.log_dir))
                logger.info(f"W&B initialized: {wandb_project}")
            except ImportError:
                logger.warning("wandb not installed, skipping W&B logging")
        
        # Track previous weights for update ratio
        self._prev_weight_norm = None
        
        logger.info(f"MetricsLogger initialized: {self.log_dir}")
        logger.info(f"  TensorBoard: {tensorboard}, CSV: {csv_logging}, W&B: {wandb_project or 'disabled'}")
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        # Write header
        self.csv_writer.writerow([
            "step", "timestamp", "loss", "learning_rate", 
            "grad_norm_total", "weight_norm_total", "weight_update_ratio",
            "step_time_ms", "tokens_per_sec", "memory_allocated_mb",
            "phase", "frozen_layers", "context_length"
        ])
        self.csv_file.flush()
    
    def compute_metrics(
        self,
        model: nn.Module,
        loss: float,
        step: int,
        learning_rate: float,
        step_time: float,
        tokens_processed: int,
        batch_size: int,
        phase: str = "",
        frozen_layers: int = 0,
        context_length: int = 0,
        detailed: bool = False,
    ) -> TrainingMetrics:
        """Compute comprehensive training metrics.
        
        Args:
            model: The model being trained
            loss: Current loss value
            step: Current training step
            learning_rate: Current learning rate
            step_time: Time for this step in seconds
            tokens_processed: Number of tokens in this step
            batch_size: Batch size
            phase: Training phase name
            frozen_layers: Number of frozen layers
            context_length: Current context length
            detailed: Whether to compute per-layer metrics
        
        Returns:
            TrainingMetrics dataclass
        """
        metrics = TrainingMetrics(
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            step_time_ms=step_time * 1000,
            tokens_per_sec=tokens_processed / max(step_time, 1e-6),
            samples_per_sec=batch_size / max(step_time, 1e-6),
            phase=phase,
            frozen_layers=frozen_layers,
            context_length=context_length,
        )
        
        # Gradient norms
        grad_norm = self._compute_grad_norms(model, detailed)
        metrics.grad_norm_total = grad_norm["total"]
        metrics.grad_norm_clipped = grad_norm["total"] > 1.0  # Assuming max_norm=1.0
        if detailed:
            metrics.grad_norms_by_layer = grad_norm["by_layer"]
        
        # Weight norms
        weight_norm = self._compute_weight_norms(model, detailed)
        metrics.weight_norm_total = weight_norm["total"]
        if detailed:
            metrics.weight_norms_by_layer = weight_norm["by_layer"]
        
        # Weight update ratio
        if self._prev_weight_norm is not None:
            delta = abs(weight_norm["total"] - self._prev_weight_norm)
            metrics.weight_update_ratio = delta / max(self._prev_weight_norm, 1e-8)
        self._prev_weight_norm = weight_norm["total"]
        
        # Memory (MPS/CUDA)
        if torch.cuda.is_available():
            metrics.memory_allocated_mb = torch.cuda.memory_allocated() / 1024**2
            metrics.memory_reserved_mb = torch.cuda.memory_reserved() / 1024**2
        elif torch.backends.mps.is_available():
            # MPS doesn't have detailed memory API, estimate from model
            metrics.memory_allocated_mb = sum(p.element_size() * p.nelement() for p in model.parameters()) / 1024**2
        
        return metrics
    
    def _compute_grad_norms(self, model: nn.Module, by_layer: bool = False) -> Dict[str, Any]:
        """Compute gradient norms."""
        total_norm = 0.0
        by_layer_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                
                if by_layer:
                    # Group by layer number
                    layer_key = name.split(".")[0]
                    if layer_key not in by_layer_norms:
                        by_layer_norms[layer_key] = 0.0
                    by_layer_norms[layer_key] += param_norm ** 2
        
        total_norm = total_norm ** 0.5
        
        if by_layer:
            by_layer_norms = {k: v ** 0.5 for k, v in by_layer_norms.items()}
        
        return {"total": total_norm, "by_layer": by_layer_norms}
    
    def _compute_weight_norms(self, model: nn.Module, by_layer: bool = False) -> Dict[str, Any]:
        """Compute weight norms."""
        total_norm = 0.0
        by_layer_norms = {}
        
        for name, param in model.named_parameters():
            param_norm = param.data.norm(2).item()
            total_norm += param_norm ** 2
            
            if by_layer:
                layer_key = name.split(".")[0]
                if layer_key not in by_layer_norms:
                    by_layer_norms[layer_key] = 0.0
                by_layer_norms[layer_key] += param_norm ** 2
        
        total_norm = total_norm ** 0.5
        
        if by_layer:
            by_layer_norms = {k: v ** 0.5 for k, v in by_layer_norms.items()}
        
        return {"total": total_norm, "by_layer": by_layer_norms}
    
    def log(self, metrics: TrainingMetrics):
        """Log metrics to all backends."""
        step = metrics.step
        
        # Console logging
        if step % self.log_interval == 0:
            logger.info(
                f"Step {step:6d} | Loss: {metrics.loss:.4f} | "
                f"LR: {metrics.learning_rate:.2e} | Grad: {metrics.grad_norm_total:.2f} | "
                f"Tok/s: {metrics.tokens_per_sec:.0f}"
            )
        
        # TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar("train/loss", metrics.loss, step)
            self.tb_writer.add_scalar("train/learning_rate", metrics.learning_rate, step)
            self.tb_writer.add_scalar("train/grad_norm", metrics.grad_norm_total, step)
            self.tb_writer.add_scalar("train/weight_norm", metrics.weight_norm_total, step)
            self.tb_writer.add_scalar("train/weight_update_ratio", metrics.weight_update_ratio, step)
            self.tb_writer.add_scalar("perf/tokens_per_sec", metrics.tokens_per_sec, step)
            self.tb_writer.add_scalar("perf/step_time_ms", metrics.step_time_ms, step)
            self.tb_writer.add_scalar("perf/memory_mb", metrics.memory_allocated_mb, step)
            
            # Per-layer metrics (less frequent)
            if step % self.detailed_interval == 0 and metrics.grad_norms_by_layer:
                for layer, norm in metrics.grad_norms_by_layer.items():
                    self.tb_writer.add_scalar(f"grad_norm/{layer}", norm, step)
                for layer, norm in metrics.weight_norms_by_layer.items():
                    self.tb_writer.add_scalar(f"weight_norm/{layer}", norm, step)
        
        # CSV
        if self.csv_writer:
            self.csv_writer.writerow([
                metrics.step, metrics.timestamp, metrics.loss, metrics.learning_rate,
                metrics.grad_norm_total, metrics.weight_norm_total, metrics.weight_update_ratio,
                metrics.step_time_ms, metrics.tokens_per_sec, metrics.memory_allocated_mb,
                metrics.phase, metrics.frozen_layers, metrics.context_length
            ])
            self.csv_file.flush()
        
        # Weights & Biases
        if self.wandb_run:
            import wandb
            log_dict = {
                "loss": metrics.loss,
                "learning_rate": metrics.learning_rate,
                "grad_norm": metrics.grad_norm_total,
                "weight_norm": metrics.weight_norm_total,
                "tokens_per_sec": metrics.tokens_per_sec,
                "step_time_ms": metrics.step_time_ms,
            }
            if step % self.detailed_interval == 0 and metrics.grad_norms_by_layer:
                for layer, norm in metrics.grad_norms_by_layer.items():
                    log_dict[f"grad_norm/{layer}"] = norm
            wandb.log(log_dict, step=step)
    
    def log_histograms(self, model: nn.Module, step: int):
        """Log weight and gradient histograms (expensive, call sparingly)."""
        if not self.tb_writer:
            return
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.tb_writer.add_histogram(f"weights/{name}", param.data, step)
                if param.grad is not None:
                    self.tb_writer.add_histogram(f"gradients/{name}", param.grad.data, step)
    
    def close(self):
        """Close all logging backends."""
        if self.tb_writer:
            self.tb_writer.close()
        if self.csv_file:
            self.csv_file.close()
        if self.wandb_run:
            import wandb
            wandb.finish()


# Visualization tools reference
VISUALIZATION_TOOLS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        TRAINING VISUALIZATION TOOLS                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  1. TensorBoard (Default - Local)                                            ║
║     ───────────────────────────                                              ║
║     tensorboard --logdir logs/                                               ║
║     Open: http://localhost:6006                                              ║
║                                                                               ║
║     Features:                                                                 ║
║     • Scalar plots (loss, LR, grad norms)                                    ║
║     • Histograms (weight distributions)                                      ║
║     • Images (attention patterns)                                            ║
║     • Graphs (model architecture)                                            ║
║                                                                               ║
║  2. Weights & Biases (Cloud - Optional)                                      ║
║     ────────────────────────────────                                         ║
║     pip install wandb                                                         ║
║     wandb login                                                               ║
║                                                                               ║
║     Features:                                                                 ║
║     • Real-time cloud dashboards                                             ║
║     • Automatic hyperparameter tracking                                       ║
║     • Experiment comparison                                                   ║
║     • Team collaboration                                                      ║
║                                                                               ║
║  3. CSV + Pandas (Simple Analysis)                                           ║
║     ─────────────────────────────                                            ║
║     import pandas as pd                                                       ║
║     df = pd.read_csv('logs/metrics.csv')                                     ║
║     df.plot(x='step', y=['loss', 'grad_norm'])                               ║
║                                                                               ║
║  4. MLflow (Experiment Tracking)                                             ║
║     ────────────────────────────                                             ║
║     pip install mlflow                                                        ║
║     mlflow ui                                                                 ║
║                                                                               ║
║  5. Aim (Fast Local Alternative)                                             ║
║     ──────────────────────────                                               ║
║     pip install aim                                                           ║
║     aim up                                                                    ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

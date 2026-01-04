"""Model configuration for the 60M parameter LLM.

Architecture: 60 layers, 320 hidden, 5 heads, 2× MLP ratio
Total parameters: ~59M
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the LLM architecture.
    
    Default values match the 60M parameter spec:
    - 60 layers for deep reasoning
    - 320 hidden size balanced with depth
    - 5 attention heads (64 dim each)
    - 2× MLP expansion (640 intermediate)
    """
    
    # Vocabulary & Embeddings
    vocab_size: int = 32000
    max_seq_len: int = 512
    
    # Architecture
    n_layers: int = 60
    hidden_size: int = 320
    n_heads: int = 5
    head_dim: int = 64  # hidden_size // n_heads
    intermediate_size: int = 640  # 2 × hidden_size
    
    # Normalization
    norm_eps: float = 1e-5
    
    # Dropout (minimal for deep models)
    dropout: float = 0.05
    attention_dropout: float = 0.0
    
    # Positional encoding
    rope_theta: float = 10000.0
    
    # Weight initialization
    initializer_range: float = 0.02
    
    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Training options
    use_gradient_checkpointing: bool = True
    tie_word_embeddings: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden_size == self.n_heads * self.head_dim, (
            f"hidden_size ({self.hidden_size}) must equal "
            f"n_heads ({self.n_heads}) × head_dim ({self.head_dim})"
        )
        assert self.intermediate_size == 2 * self.hidden_size, (
            f"intermediate_size should be 2× hidden_size for this architecture"
        )
    
    @property
    def residual_scale(self) -> float:
        """Scaling factor for residual connections (depth-scaled init)."""
        import math
        return self.initializer_range / math.sqrt(2 * self.n_layers)
    
    def count_parameters(self) -> dict:
        """Estimate parameter counts for each component."""
        # Embeddings
        embed_params = self.vocab_size * self.hidden_size
        
        # Per-layer attention: Q, K, V, O projections
        attn_per_layer = 4 * (self.hidden_size * self.hidden_size)
        
        # Per-layer MLP: up and down projections
        mlp_per_layer = 2 * (self.hidden_size * self.intermediate_size)
        
        # Per-layer norms: 2 × hidden_size (gamma only, no bias typically)
        norm_per_layer = 2 * self.hidden_size
        
        # Total per layer
        per_layer = attn_per_layer + mlp_per_layer + norm_per_layer
        
        # Body (all layers)
        body_params = self.n_layers * per_layer
        
        # Final LayerNorm
        final_norm = self.hidden_size
        
        # LM Head (tied to embeddings if tie_word_embeddings=True)
        lm_head = 0 if self.tie_word_embeddings else self.vocab_size * self.hidden_size
        
        total = embed_params + body_params + final_norm + lm_head
        
        return {
            "embeddings": embed_params,
            "per_layer": per_layer,
            "body": body_params,
            "final_norm": final_norm,
            "lm_head": lm_head,
            "total": total,
            "total_millions": total / 1e6,
        }


@dataclass
class TrainingConfig:
    """Training hyperparameters for each phase."""
    
    # Optimizer
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Schedule
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # Batch size
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_workers: int = 4  # Parallel data loading (use efficiency cores)
    
    # Training duration
    max_steps: Optional[int] = None
    num_epochs: Optional[int] = None
    
    # Context curriculum
    context_schedule: dict = None
    
    # Layer freezing
    freeze_schedule: dict = None
    
    # Checkpointing
    save_steps: int = 50
    eval_steps: int = 500
    logging_steps: int = 10
    
    # Device - auto-detect (can override with TRAINING_DEVICE env var)
    device: str = "auto"  # "auto", "cuda", "mps", or "cpu"
    dtype: str = "bfloat16"
    compile_model: bool = True
    
    def _get_device(self) -> str:
        """Auto-detect best available device."""
        import os
        import torch
        
        # Environment override
        env_device = os.environ.get("TRAINING_DEVICE")
        if env_device:
            return env_device
        
        if self.device != "auto":
            return self.device
        
        # Auto-detect
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def __post_init__(self):
        if self.context_schedule is None:
            # Default: 128 → 256 → 512 as training progresses
            self.context_schedule = {
                0.0: 128,
                0.3: 256,
                0.7: 512,
            }
        
        if self.freeze_schedule is None:
            # Default: freeze bottom layers progressively
            self.freeze_schedule = {
                0.0: 0,    # Train all layers
                0.2: 20,   # Freeze bottom 20 layers
                0.5: 40,   # Freeze bottom 40 layers
            }
    
    @property
    def effective_batch_size(self) -> int:
        """Effective batch size after gradient accumulation."""
        return self.micro_batch_size * self.gradient_accumulation_steps


# Preset configurations for each training phase
PHASE_CONFIGS = {
    "phase1_grammar": TrainingConfig(
        learning_rate=3e-4,
        num_epochs=1,
        context_schedule={0.0: 128, 0.3: 256, 0.7: 512},
        freeze_schedule={},  # Explicitly disable freezing for Phase 1
    ),
    "phase2_vocabulary": TrainingConfig(
        learning_rate=1e-4,  # Lower LR
        num_epochs=1,
        context_schedule={0.0: 256, 0.5: 512},
    ),
    "phase2b_scientific": TrainingConfig(
        learning_rate=5e-5,  # Even lower for fine-tuning
        num_epochs=1,
        context_schedule={0.0: 512},
        freeze_schedule={0.0: 30},  # Keep more layers frozen
    ),
}

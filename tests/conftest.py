"""Shared test fixtures and configuration.

Provides:
- Reproducible randomness via seeding
- Common model configurations for testing
- Device selection helpers
"""

import pytest
import torch
import numpy as np
import random
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.config import ModelConfig, TrainingConfig


# ============================================================================
# Seed Management
# ============================================================================

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


@pytest.fixture(autouse=True)
def seed_everything():
    """Auto-seed before each test for reproducibility."""
    set_seed(42)
    yield


# ============================================================================
# Device Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture
def cpu_device():
    """Force CPU device for deterministic tests."""
    return torch.device("cpu")


# ============================================================================
# Model Config Fixtures
# ============================================================================

@pytest.fixture
def small_config():
    """Small config for fast testing (fewer layers)."""
    return ModelConfig(
        vocab_size=1000,
        max_seq_len=64,
        n_layers=4,  # Reduced for speed
        hidden_size=64,
        n_heads=4,
        head_dim=16,
        intermediate_size=128,
        dropout=0.0,
        attention_dropout=0.0,
        use_gradient_checkpointing=False,
    )


@pytest.fixture
def full_config():
    """Full 60M parameter config as per spec."""
    return ModelConfig(
        vocab_size=32000,
        max_seq_len=512,
        n_layers=60,
        hidden_size=320,
        n_heads=5,
        head_dim=64,
        intermediate_size=640,
        dropout=0.05,
        attention_dropout=0.0,
        use_gradient_checkpointing=True,
    )


@pytest.fixture
def medium_config():
    """Medium config for testing with reasonable depth."""
    return ModelConfig(
        vocab_size=4000,
        max_seq_len=128,
        n_layers=12,
        hidden_size=128,
        n_heads=4,
        head_dim=32,
        intermediate_size=256,
        dropout=0.0,
        attention_dropout=0.0,
        use_gradient_checkpointing=False,
    )


# ============================================================================
# Training Config Fixtures
# ============================================================================

@pytest.fixture
def train_config():
    """Default training config for tests."""
    return TrainingConfig(
        learning_rate=3e-4,
        min_learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        save_steps=100,
        eval_steps=50,
        logging_steps=10,
        device="cpu",
        compile_model=False,
    )


# ============================================================================
# Tensor Fixtures
# ============================================================================

@pytest.fixture
def sample_batch(small_config):
    """Generate a sample training batch."""
    batch_size = 2
    seq_len = small_config.max_seq_len
    
    input_ids = torch.randint(
        0, small_config.vocab_size, 
        (batch_size, seq_len), 
        dtype=torch.long
    )
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
    labels = input_ids.clone()
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ============================================================================
# Tolerance Constants
# ============================================================================

# For initialization tests
INIT_STD_TOLERANCE = 0.01  # Allow 1% deviation from expected std
INIT_STD_BASE = 0.02       # Base std from spec

# For numerical comparisons
ATOL = 1e-5  # Absolute tolerance
RTOL = 1e-4  # Relative tolerance

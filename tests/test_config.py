"""Tests for model configuration.

Validates:
- Config constraints and validation
- Residual scaling computation (critical for 60-layer stability)
- Parameter counting accuracy
"""

import pytest
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.config import ModelConfig, TrainingConfig, PHASE_CONFIGS


class TestModelConfig:
    """Test ModelConfig validation and properties."""
    
    def test_default_config_valid(self):
        """Default config should pass validation."""
        config = ModelConfig()
        assert config.hidden_size == config.n_heads * config.head_dim
        assert config.intermediate_size == 2 * config.hidden_size
    
    def test_hidden_size_validation(self):
        """Config should reject mismatched hidden_size."""
        with pytest.raises(AssertionError):
            ModelConfig(
                hidden_size=320,
                n_heads=5,
                head_dim=32,  # 5 * 32 = 160 != 320
            )
    
    def test_intermediate_size_validation(self):
        """Config should reject wrong intermediate_size."""
        with pytest.raises(AssertionError):
            ModelConfig(
                hidden_size=320,
                n_heads=5,
                head_dim=64,
                intermediate_size=512,  # Should be 640
            )
    
    def test_residual_scale_formula(self, full_config):
        """Residual scale must match spec formula: 0.02 / sqrt(2 * n_layers)."""
        expected = 0.02 / math.sqrt(2 * 60)  # ~0.00183
        actual = full_config.residual_scale
        
        assert abs(actual - expected) < 1e-6, (
            f"Residual scale {actual} != expected {expected}"
        )
        assert abs(actual - 0.00183) < 0.0001, (
            f"Residual scale should be ~0.00183 for 60 layers"
        )
    
    def test_residual_scale_varies_with_depth(self):
        """Residual scale should decrease with more layers."""
        shallow = ModelConfig(
            n_layers=12,
            hidden_size=320,
            n_heads=5,
            head_dim=64,
            intermediate_size=640,
        )
        deep = ModelConfig(
            n_layers=60,
            hidden_size=320,
            n_heads=5,
            head_dim=64,
            intermediate_size=640,
        )
        
        assert shallow.residual_scale > deep.residual_scale
        
        # Deep model should have ~sqrt(5) smaller scale
        ratio = shallow.residual_scale / deep.residual_scale
        expected_ratio = math.sqrt(60 / 12)  # sqrt(5) ≈ 2.24
        assert abs(ratio - expected_ratio) < 0.01


class TestParameterCounting:
    """Test parameter count estimation."""
    
    def test_parameter_count_full_model(self, full_config):
        """Full model should be approximately 59M parameters."""
        counts = full_config.count_parameters()
        
        # Check total is in expected range (55-65M)
        total_m = counts["total_millions"]
        assert 55 < total_m < 65, f"Total params {total_m}M out of expected range"
    
    def test_embedding_count(self, full_config):
        """Embedding params = vocab_size * hidden_size."""
        counts = full_config.count_parameters()
        expected_embed = 32000 * 320
        assert counts["embeddings"] == expected_embed
    
    def test_per_layer_breakdown(self, full_config):
        """Per-layer params should match spec formula."""
        counts = full_config.count_parameters()
        
        # Attention: 4 * hidden^2 (Q, K, V, O projections)
        attn_expected = 4 * 320 * 320
        
        # MLP: 2 * hidden * intermediate
        mlp_expected = 2 * 320 * 640
        
        # LayerNorm: 2 * hidden (gamma for attn_norm and mlp_norm)
        norm_expected = 2 * 320
        
        total_per_layer = attn_expected + mlp_expected + norm_expected
        
        # Allow small deviation due to implementation details
        assert abs(counts["per_layer"] - total_per_layer) < 100
    
    def test_tied_embeddings_saves_params(self, full_config):
        """With tie_word_embeddings, lm_head should have 0 params."""
        full_config.tie_word_embeddings = True
        counts_tied = full_config.count_parameters()
        
        full_config.tie_word_embeddings = False
        counts_untied = full_config.count_parameters()
        
        assert counts_tied["lm_head"] == 0
        assert counts_untied["lm_head"] == 32000 * 320
        assert counts_untied["total"] > counts_tied["total"]


class TestTrainingConfig:
    """Test TrainingConfig validation and properties."""
    
    def test_effective_batch_size(self):
        """Effective batch = micro_batch * gradient_accumulation."""
        config = TrainingConfig(
            micro_batch_size=2,
            gradient_accumulation_steps=16,
        )
        assert config.effective_batch_size == 32
    
    def test_default_context_schedule(self):
        """Default context schedule should be 128 → 256 → 512."""
        config = TrainingConfig()
        assert config.context_schedule[0.0] == 128
        assert config.context_schedule[0.3] == 256
        assert config.context_schedule[0.7] == 512
    
    def test_default_freeze_schedule(self):
        """Default freeze schedule should progressively freeze layers."""
        config = TrainingConfig()
        assert config.freeze_schedule[0.0] == 0
        assert config.freeze_schedule[0.2] == 20
        assert config.freeze_schedule[0.5] == 40


class TestPhaseConfigs:
    """Test preset phase configurations."""
    
    def test_phase_configs_exist(self):
        """All expected phases should be defined."""
        assert "phase1_grammar" in PHASE_CONFIGS
        assert "phase2_vocabulary" in PHASE_CONFIGS
        assert "phase2b_scientific" in PHASE_CONFIGS
    
    def test_phase_lr_decreases(self):
        """Learning rate should decrease across phases."""
        phase1_lr = PHASE_CONFIGS["phase1_grammar"].learning_rate
        phase2_lr = PHASE_CONFIGS["phase2_vocabulary"].learning_rate
        phase2b_lr = PHASE_CONFIGS["phase2b_scientific"].learning_rate
        
        assert phase1_lr > phase2_lr > phase2b_lr

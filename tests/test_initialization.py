"""Tests for weight initialization.

CRITICAL: Incorrect initialization causes training divergence in 60-layer models.

Validates:
- Base std = 0.02 for most weights
- Depth-scaled std ≈ 0.00183 for residual projections
- LayerNorm initialized to identity
"""

import pytest
import torch
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.config import ModelConfig
from model.llm import LLM, create_model


class TestBaseInitialization:
    """Test base weight initialization (std = 0.02)."""
    
    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        config = ModelConfig(
            vocab_size=1000,
            max_seq_len=64,
            n_layers=4,
            hidden_size=64,
            n_heads=4,
            head_dim=16,
            intermediate_size=128,
            use_gradient_checkpointing=False,
        )
        return LLM(config)
    
    def test_embedding_std(self, model):
        """Token embeddings should have std ≈ 0.02."""
        weights = model.token_embedding.weight.data
        
        # Remove padding idx row (set to zero)
        non_pad = weights[1:]  # Skip first row (padding)
        
        actual_std = non_pad.std().item()
        expected_std = 0.02
        
        # Allow 20% tolerance due to random sampling variance
        assert abs(actual_std - expected_std) / expected_std < 0.2, (
            f"Embedding std {actual_std} not close to expected {expected_std}"
        )
    
    def test_qkv_projection_std(self, model):
        """Q, K, V projections should have std ≈ 0.02."""
        for layer_idx, layer in enumerate(model.layers):
            weights = layer.attention.qkv_proj.weight.data
            actual_std = weights.std().item()
            
            # Should be close to base initializer_range (0.02)
            assert 0.01 < actual_std < 0.04, (
                f"Layer {layer_idx} QKV std {actual_std} out of range"
            )
    
    def test_mlp_up_projection_std(self, model):
        """MLP up-projection should have std ≈ 0.02."""
        for layer_idx, layer in enumerate(model.layers):
            weights = layer.mlp.up_proj.weight.data
            actual_std = weights.std().item()
            
            assert 0.01 < actual_std < 0.04, (
                f"Layer {layer_idx} MLP up_proj std {actual_std} out of range"
            )


class TestResidualScaledInitialization:
    """Test depth-scaled initialization for residual projections - CRITICAL."""
    
    @pytest.fixture
    def deep_model(self):
        """Create the full 60-layer model for testing."""
        config = ModelConfig(
            vocab_size=1000,
            max_seq_len=64,
            n_layers=60,  # Full depth
            hidden_size=320,
            n_heads=5,
            head_dim=64,
            intermediate_size=640,
            use_gradient_checkpointing=False,
        )
        return LLM(config)
    
    def test_attention_output_projection_scaled(self, deep_model):
        """Attention output projection should have scaled std ≈ 0.00183."""
        expected_std = 0.02 / math.sqrt(2 * 60)  # ~0.00183
        
        stds = []
        for layer in deep_model.layers:
            weights = layer.attention.o_proj.weight.data
            stds.append(weights.std().item())
        
        mean_std = sum(stds) / len(stds)
        
        # Allow 30% tolerance for statistical variance
        assert abs(mean_std - expected_std) / expected_std < 0.3, (
            f"Attention o_proj mean std {mean_std} not close to expected {expected_std}"
        )
    
    def test_mlp_down_projection_scaled(self, deep_model):
        """MLP down-projection should have scaled std ≈ 0.00183."""
        expected_std = 0.02 / math.sqrt(2 * 60)  # ~0.00183
        
        stds = []
        for layer in deep_model.layers:
            weights = layer.mlp.down_proj.weight.data
            stds.append(weights.std().item())
        
        mean_std = sum(stds) / len(stds)
        
        assert abs(mean_std - expected_std) / expected_std < 0.3, (
            f"MLP down_proj mean std {mean_std} not close to expected {expected_std}"
        )
    
    def test_residual_scale_changes_with_depth(self):
        """Deeper models should have smaller residual scaling."""
        # Shallow model
        shallow_config = ModelConfig(
            vocab_size=1000,
            max_seq_len=64,
            n_layers=12,
            hidden_size=64,
            n_heads=4,
            head_dim=16,
            intermediate_size=128,
        )
        shallow = LLM(shallow_config)
        
        # Deep model
        deep_config = ModelConfig(
            vocab_size=1000,
            max_seq_len=64,
            n_layers=48,
            hidden_size=64,
            n_heads=4,
            head_dim=16,
            intermediate_size=128,
        )
        deep = LLM(deep_config)
        
        shallow_std = shallow.layers[0].attention.o_proj.weight.data.std().item()
        deep_std = deep.layers[0].attention.o_proj.weight.data.std().item()
        
        # Deep model should have smaller residual projection std
        assert deep_std < shallow_std, (
            f"Deep model residual std ({deep_std}) should be less than "
            f"shallow model ({shallow_std})"
        )


class TestLayerNormInitialization:
    """Test LayerNorm initialization."""
    
    @pytest.fixture
    def model(self):
        config = ModelConfig(
            vocab_size=1000,
            max_seq_len=64,
            n_layers=4,
            hidden_size=64,
            n_heads=4,
            head_dim=16,
            intermediate_size=128,
        )
        return LLM(config)
    
    def test_layernorm_gamma_is_one(self, model):
        """LayerNorm gamma (weight) should be initialized to 1."""
        # Check final norm
        assert torch.allclose(
            model.final_norm.weight, 
            torch.ones_like(model.final_norm.weight)
        ), "Final LayerNorm gamma not initialized to 1"
        
        # Check layer norms in blocks
        for layer_idx, layer in enumerate(model.layers):
            assert torch.allclose(
                layer.attn_norm.weight,
                torch.ones_like(layer.attn_norm.weight)
            ), f"Layer {layer_idx} attn_norm gamma not 1"
            
            assert torch.allclose(
                layer.mlp_norm.weight,
                torch.ones_like(layer.mlp_norm.weight)
            ), f"Layer {layer_idx} mlp_norm gamma not 1"
    
    def test_layernorm_beta_is_zero(self, model):
        """LayerNorm beta (bias) should be initialized to 0."""
        assert torch.allclose(
            model.final_norm.bias,
            torch.zeros_like(model.final_norm.bias)
        ), "Final LayerNorm beta not initialized to 0"
        
        for layer_idx, layer in enumerate(model.layers):
            assert torch.allclose(
                layer.attn_norm.bias,
                torch.zeros_like(layer.attn_norm.bias)
            ), f"Layer {layer_idx} attn_norm beta not 0"
            
            assert torch.allclose(
                layer.mlp_norm.bias,
                torch.zeros_like(layer.mlp_norm.bias)
            ), f"Layer {layer_idx} mlp_norm beta not 0"


class TestPaddingEmbedding:
    """Test padding token embedding is zeroed."""
    
    def test_padding_embedding_zero(self):
        """Padding token embedding should be all zeros."""
        config = ModelConfig(
            vocab_size=1000,
            max_seq_len=64,
            n_layers=2,
            hidden_size=64,
            n_heads=4,
            head_dim=16,
            intermediate_size=128,
            pad_token_id=0,
        )
        model = LLM(config)
        
        padding_embedding = model.token_embedding.weight.data[0]
        
        assert torch.allclose(
            padding_embedding,
            torch.zeros_like(padding_embedding)
        ), "Padding token embedding should be zeros"


class TestInitializationStats:
    """Statistical tests for initialization."""
    
    def test_overall_weight_distribution(self):
        """All weights should have reasonable statistics."""
        config = ModelConfig(
            vocab_size=1000,
            max_seq_len=64,
            n_layers=8,
            hidden_size=64,
            n_heads=4,
            head_dim=16,
            intermediate_size=128,
        )
        model = LLM(config)
        
        for name, param in model.named_parameters():
            if "weight" in name and "norm" not in name:
                # Check no extreme values
                max_val = param.data.abs().max().item()
                assert max_val < 1.0, f"{name} has extreme value: {max_val}"
                
                # Check mean is near zero
                mean_val = param.data.mean().item()
                assert abs(mean_val) < 0.1, f"{name} has non-zero mean: {mean_val}"
    
    def test_no_zero_weights(self):
        """Non-bias weights should not be all zero after init."""
        config = ModelConfig(
            vocab_size=1000,
            max_seq_len=64,
            n_layers=4,
            hidden_size=64,
            n_heads=4,
            head_dim=16,
            intermediate_size=128,
        )
        model = LLM(config)
        
        for name, param in model.named_parameters():
            if "weight" in name:
                # Skip padding embedding check separately
                if "embedding" in name:
                    continue
                
                # Weight should not be all zeros
                assert param.data.abs().sum() > 0, f"{name} is all zeros!"


class TestInitializationFormulas:
    """Verify initialization matches spec formulas exactly."""
    
    def test_residual_scaling_formula(self):
        """Verify: residual_std = base_std / sqrt(2 * n_layers)."""
        n_layers = 60
        base_std = 0.02
        
        expected = base_std / math.sqrt(2 * n_layers)
        
        config = ModelConfig(
            vocab_size=1000,
            max_seq_len=64,
            n_layers=n_layers,
            hidden_size=320,
            n_heads=5,
            head_dim=64,
            intermediate_size=640,
        )
        
        actual = config.residual_scale
        
        assert abs(actual - expected) < 1e-10, (
            f"Residual scale {actual} != {expected}"
        )
    
    def test_60_layer_residual_scale_value(self):
        """For 60 layers, residual scale should be approximately 0.00183."""
        config = ModelConfig(
            vocab_size=1000,
            max_seq_len=64,
            n_layers=60,
            hidden_size=320,
            n_heads=5,
            head_dim=64,
            intermediate_size=640,
        )
        
        expected_approx = 0.00183
        actual = config.residual_scale
        
        # Should be within 1% of expected
        assert abs(actual - expected_approx) / expected_approx < 0.01, (
            f"60-layer residual scale {actual} not ≈ {expected_approx}"
        )

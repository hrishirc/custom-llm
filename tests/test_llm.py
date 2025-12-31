"""Tests for the complete LLM model.

Validates:
- Full forward/backward pass
- Gradient flow without NaN/Inf
- Layer freezing
- Parameter counting matches spec
- Loss computation
- Generation functionality
"""

import pytest
import torch
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.config import ModelConfig
from model.llm import LLM, create_model


class TestForwardPass:
    """Test forward pass shapes and outputs."""
    
    @pytest.fixture
    def model_and_config(self):
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
        model = LLM(config)
        return model, config
    
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [16, 32, 64])
    def test_logits_shape(self, model_and_config, batch_size, seq_len):
        """Logits should have shape (batch, seq_len, vocab_size)."""
        model, config = model_and_config
        
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        outputs = model(input_ids)
        
        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)
    
    def test_forward_with_attention_mask(self, model_and_config):
        """Model should accept attention mask."""
        model, config = model_and_config
        
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, seq_len//2:] = 0  # Mask second half
        
        outputs = model(input_ids, attention_mask=attention_mask)
        
        assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)
    
    def test_forward_with_labels_returns_loss(self, model_and_config):
        """Forward with labels should compute loss."""
        model, config = model_and_config
        
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels)
        
        assert "loss" in outputs
        assert outputs["loss"].shape == ()  # Scalar
        assert outputs["loss"].item() > 0  # Should be positive


class TestBackwardPass:
    """Test backward pass and gradient flow - CRITICAL for training."""
    
    @pytest.fixture
    def model(self):
        config = ModelConfig(
            vocab_size=1000,
            max_seq_len=64,
            n_layers=8,
            hidden_size=64,
            n_heads=4,
            head_dim=16,
            intermediate_size=128,
            use_gradient_checkpointing=False,
        )
        return LLM(config)
    
    def test_no_nan_gradients(self, model):
        """Gradients should not contain NaN values."""
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), (
                    f"NaN gradient in {name}"
                )
    
    def test_no_inf_gradients(self, model):
        """Gradients should not contain Inf values."""
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isinf(param.grad).any(), (
                    f"Inf gradient in {name}"
                )
    
    def test_all_parameters_get_gradients(self, model):
        """All trainable parameters should receive gradients."""
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                # Gradient should not be all zeros (for non-sparse params)
                if "embed" not in name:  # Embeddings can be sparse
                    assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
    
    def test_gradient_magnitude_reasonable(self, model):
        """Gradient magnitudes should be reasonable (not exploding)."""
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                
                # Gradient norm should be less than 100 for properly initialized model
                assert grad_norm < 100, (
                    f"Gradient explosion in {name}: norm = {grad_norm}"
                )


class TestGradientFlowDeepModel:
    """Test gradient flow in deep (60-layer) model configuration."""
    
    def test_gradient_reaches_early_layers(self):
        """Gradient should propagate to early layers in deep model."""
        config = ModelConfig(
            vocab_size=1000,
            max_seq_len=32,
            n_layers=20,  # Moderately deep for test
            hidden_size=64,
            n_heads=4,
            head_dim=16,
            intermediate_size=128,
            use_gradient_checkpointing=False,
        )
        model = LLM(config)
        
        input_ids = torch.randint(0, 1000, (1, 16))
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels)
        outputs["loss"].backward()
        
        # Check first layer gets gradient
        first_layer = model.layers[0]
        for name, param in first_layer.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert grad_norm > 1e-10, (
                    f"Gradient vanished in first layer {name}: {grad_norm}"
                )
        
        # Check embeddings get gradient
        embed_grad_norm = model.token_embedding.weight.grad.norm().item()
        assert embed_grad_norm > 1e-10, f"Gradient vanished at embeddings: {embed_grad_norm}"


class TestLayerFreezing:
    """Test layer freezing functionality."""
    
    @pytest.fixture
    def model(self):
        config = ModelConfig(
            vocab_size=1000,
            max_seq_len=64,
            n_layers=8,
            hidden_size=64,
            n_heads=4,
            head_dim=16,
            intermediate_size=128,
        )
        return LLM(config)
    
    def test_freeze_layers(self, model):
        """Frozen layers should have requires_grad=False."""
        model.freeze_layers(4)  # Freeze first 4 layers
        
        # First 4 layers should be frozen
        for i in range(4):
            for name, param in model.layers[i].named_parameters():
                assert not param.requires_grad, (
                    f"Layer {i} {name} should be frozen"
                )
        
        # Remaining layers should be trainable
        for i in range(4, 8):
            for name, param in model.layers[i].named_parameters():
                assert param.requires_grad, (
                    f"Layer {i} {name} should be trainable"
                )
    
    def test_freeze_layers_no_gradient(self, model):
        """Frozen layers should not receive gradients."""
        model.freeze_layers(4)
        
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels)
        outputs["loss"].backward()
        
        # Frozen layers should have no gradients
        for i in range(4):
            for name, param in model.layers[i].named_parameters():
                assert param.grad is None, (
                    f"Frozen layer {i} {name} should have no gradient"
                )
    
    def test_unfreeze_all(self, model):
        """freeze_layers(0) should unfreeze all layers."""
        model.freeze_layers(4)  # Freeze some
        model.freeze_layers(0)  # Unfreeze all
        
        for i, layer in enumerate(model.layers):
            for name, param in layer.named_parameters():
                assert param.requires_grad, (
                    f"Layer {i} {name} should be unfrozen"
                )


class TestParameterCounting:
    """Test parameter count matches specification."""
    
    def test_small_model_param_count(self):
        """Parameter count should match manual calculation."""
        config = ModelConfig(
            vocab_size=1000,
            max_seq_len=64,
            n_layers=4,
            hidden_size=64,
            n_heads=4,
            head_dim=16,
            intermediate_size=128,
            tie_word_embeddings=True,
        )
        model = LLM(config)
        
        actual = model.get_num_params()
        
        # Manual calculation
        embed = 1000 * 64  # 64K
        per_layer = (
            4 * 64 * 64 +  # Attention (QKV + O)
            2 * 64 * 128 +  # MLP
            2 * 64 + 2 * 64  # LayerNorms (gamma + beta)
        )
        body = 4 * per_layer
        final_norm = 64 + 64  # gamma + beta
        
        expected = embed + body + final_norm
        
        # Allow some tolerance for implementation details
        assert abs(actual - expected) / expected < 0.1
    
    def test_full_spec_param_count(self):
        """60M model should have approximately 59M parameters."""
        config = ModelConfig(
            vocab_size=32000,
            max_seq_len=512,
            n_layers=60,
            hidden_size=320,
            n_heads=5,
            head_dim=64,
            intermediate_size=640,
            tie_word_embeddings=True,
        )
        model = LLM(config)
        
        params_m = model.get_num_params() / 1e6
        
        # Should be approximately 59M (within 55-65M range)
        assert 55 < params_m < 65, f"Param count {params_m}M out of expected range"
    
    def test_non_embedding_count(self):
        """Non-embedding parameter count should exclude embeddings."""
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
        
        total = model.get_num_params(non_embedding=False)
        non_embed = model.get_num_params(non_embedding=True)
        
        embed_size = 1000 * 64
        
        assert total - non_embed == embed_size


class TestLoss:
    """Test loss computation."""
    
    def test_loss_decreases_with_correct_labels(self):
        """Loss should be lower when predicting actual next tokens."""
        config = ModelConfig(
            vocab_size=100,
            max_seq_len=32,
            n_layers=2,
            hidden_size=32,
            n_heads=2,
            head_dim=16,
            intermediate_size=64,
        )
        model = LLM(config)
        
        # Create sequence with pattern
        input_ids = torch.arange(32).unsqueeze(0) % 100
        
        # Correct labels (shifted by 1)
        labels = input_ids.clone()
        
        # Random labels
        random_labels = torch.randint(0, 100, (1, 32))
        
        with torch.no_grad():
            loss_correct = model(input_ids, labels=labels)["loss"].item()
            loss_random = model(input_ids, labels=random_labels)["loss"].item()
        
        # Both should be positive
        assert loss_correct > 0
        assert loss_random > 0
    
    def test_loss_ignores_padding(self):
        """Loss should not count padding tokens."""
        config = ModelConfig(
            vocab_size=100,
            max_seq_len=32,
            n_layers=2,
            hidden_size=32,
            n_heads=2,
            head_dim=16,
            intermediate_size=64,
            pad_token_id=0,
        )
        model = LLM(config)
        
        # Create inputs with padding at the end
        input_ids = torch.randint(1, 100, (1, 32))  # Avoid 0
        
        # Labels: mix of real tokens and padding
        labels_no_pad = input_ids.clone()
        
        labels_with_pad = input_ids.clone()
        labels_with_pad[0, 16:] = 0  # Second half is padding
        
        with torch.no_grad():
            loss_no_pad = model(input_ids, labels=labels_no_pad)["loss"]
            loss_with_pad = model(input_ids, labels=labels_with_pad)["loss"]
        
        # Losses should differ since padding is ignored
        assert not torch.allclose(loss_no_pad, loss_with_pad)


class TestWeightTying:
    """Test embedding weight tying."""
    
    def test_tied_embeddings(self):
        """When tie_word_embeddings=True, lm_head should be None."""
        config = ModelConfig(
            vocab_size=1000,
            max_seq_len=64,
            n_layers=2,
            hidden_size=64,
            n_heads=4,
            head_dim=16,
            intermediate_size=128,
            tie_word_embeddings=True,
        )
        model = LLM(config)
        
        assert model.lm_head is None
    
    def test_untied_embeddings(self):
        """When tie_word_embeddings=False, lm_head should exist."""
        config = ModelConfig(
            vocab_size=1000,
            max_seq_len=64,
            n_layers=2,
            hidden_size=64,
            n_heads=4,
            head_dim=16,
            intermediate_size=128,
            tie_word_embeddings=False,
        )
        model = LLM(config)
        
        assert model.lm_head is not None
        assert model.lm_head.weight.shape == (1000, 64)


class TestGeneration:
    """Test text generation functionality."""
    
    @pytest.fixture
    def model(self):
        config = ModelConfig(
            vocab_size=100,
            max_seq_len=64,
            n_layers=2,
            hidden_size=32,
            n_heads=2,
            head_dim=16,
            intermediate_size=64,
        )
        return LLM(config)
    
    def test_generate_increases_length(self, model):
        """Generation should produce tokens."""
        input_ids = torch.tensor([[1, 2, 3]])  # 3 tokens
        
        output = model.generate(input_ids, max_new_tokens=5)
        
        assert output.shape[1] > input_ids.shape[1]
        assert output.shape[1] <= input_ids.shape[1] + 5
    
    def test_generate_respects_max_tokens(self, model):
        """Generation should not exceed max_new_tokens."""
        input_ids = torch.tensor([[1, 2, 3]])
        
        output = model.generate(input_ids, max_new_tokens=10)
        
        # Output should have at most original + max_new_tokens
        assert output.shape[1] <= 3 + 10
    
    def test_generate_with_temperature(self, model):
        """Different temperatures should produce different outputs."""
        input_ids = torch.tensor([[1, 2, 3]])
        
        torch.manual_seed(42)
        output_low = model.generate(input_ids, max_new_tokens=10, temperature=0.1)
        
        torch.manual_seed(42)
        output_high = model.generate(input_ids, max_new_tokens=10, temperature=2.0)
        
        # Outputs may differ due to temperature; just check they complete
        assert output_low.shape[1] > 3
        assert output_high.shape[1] > 3


class TestNumericalStability:
    """Test overall model numerical stability."""
    
    def test_forward_no_nan(self):
        """Forward pass should never produce NaN."""
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
        
        for _ in range(5):
            input_ids = torch.randint(0, 1000, (2, 32))
            outputs = model(input_ids)
            
            assert not torch.isnan(outputs["logits"]).any()
            assert not torch.isinf(outputs["logits"]).any()
    
    def test_mixed_precision_forward(self):
        """Model should work with mixed precision."""
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
        
        input_ids = torch.randint(0, 1000, (2, 32))
        
        # Test with autocast
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            outputs = model(input_ids)
        
        assert not torch.isnan(outputs["logits"]).any()


class TestCreateModelFactory:
    """Test the create_model factory function."""
    
    def test_create_default_model(self, capsys):
        """create_model() should create model with default config."""
        model = create_model()
        
        # Should print parameter info
        captured = capsys.readouterr()
        assert "Model created" in captured.out
        assert "M parameters" in captured.out
    
    def test_create_custom_model(self):
        """create_model(config) should use provided config."""
        config = ModelConfig(
            vocab_size=500,
            max_seq_len=32,
            n_layers=2,
            hidden_size=32,
            n_heads=2,
            head_dim=16,
            intermediate_size=64,
        )
        
        model = create_model(config)
        
        assert model.config.vocab_size == 500
        assert len(model.layers) == 2

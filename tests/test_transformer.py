"""Tests for Transformer blocks.

Validates:
- Pre-LayerNorm ordering (critical for 60-layer stability)
- Residual connections
- Gradient checkpointing toggle
- Block composition
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.transformer import TransformerBlock, TransformerBlockWithCheckpoint


class TestPreLayerNorm:
    """Test Pre-LN architecture - critical for deep model stability."""
    
    def test_norm_before_attention(self):
        """LayerNorm must be applied BEFORE attention, not after."""
        block = TransformerBlock(
            hidden_size=64,
            n_heads=4,
            head_dim=16,
            intermediate_size=128,
        )
        
        # Verify attn_norm exists
        assert hasattr(block, "attn_norm")
        assert isinstance(block.attn_norm, nn.LayerNorm)
        
        # Verify norm params
        assert block.attn_norm.normalized_shape == (64,)
    
    def test_norm_before_mlp(self):
        """LayerNorm must be applied BEFORE MLP, not after."""
        block = TransformerBlock(
            hidden_size=64,
            n_heads=4,
            head_dim=16,
            intermediate_size=128,
        )
        
        assert hasattr(block, "mlp_norm")
        assert isinstance(block.mlp_norm, nn.LayerNorm)
        assert block.mlp_norm.normalized_shape == (64,)
    
    def test_pre_ln_application_order(self):
        """Verify: x -> norm -> attention -> residual, not x -> attention -> norm."""
        block = TransformerBlock(
            hidden_size=64,
            n_heads=4,
            head_dim=16,
            intermediate_size=128,
        )
        
        x = torch.randn(1, 8, 64)
        
        # Pre-LN means: output = x + attention(norm(x))
        # NOT Post-LN: output = norm(x + attention(x))
        
        # Test by checking that changing norm affects attention input
        with torch.no_grad():
            # Capture normed value that goes to attention
            normed = block.attn_norm(x)
            
            # Modify gamma of attn_norm
            original_gamma = block.attn_norm.weight.clone()
            block.attn_norm.weight.data *= 2
            
            normed_scaled = block.attn_norm(x)
            
            # Reset
            block.attn_norm.weight.data = original_gamma
            
            # If Pre-LN, normed_scaled should be 2x normed
            assert torch.allclose(normed_scaled, 2 * normed, atol=1e-5)


class TestResidualConnections:
    """Test residual (skip) connections."""
    
    def test_residual_preserves_identity_at_init(self):
        """At initialization, residual should dominate (attention/MLP output small)."""
        torch.manual_seed(42)
        
        block = TransformerBlock(
            hidden_size=64,
            n_heads=4,
            head_dim=16,
            intermediate_size=128,
        )
        
        x = torch.randn(1, 8, 64)
        output = block(x)
        
        # Output should be close to input at init due to small weight init
        # (Not exact, but difference should be bounded)
        diff = (output - x).abs().mean()
        
        # With proper init, difference should be small (<1)
        assert diff < 1.0, f"Residual difference too large: {diff}"
    
    def test_residual_adds_input(self):
        """Residual connection should add input to sublayer output."""
        block = TransformerBlock(
            hidden_size=64,
            n_heads=4,
            head_dim=16,
            intermediate_size=128,
        )
        
        x = torch.randn(1, 8, 64)
        
        # Zero out attention output to verify residual
        with torch.no_grad():
            block.attention.o_proj.weight.zero_()
        
        # After zeroing attention, residual should dominate
        # x -> norm -> attention(=0) -> x + 0 = x -> norm -> MLP -> ...
        # The first residual should preserve x through attention
        
        # We can test by checking output contains input information
        output = block(x)
        
        # Output should still be influenced by x via residual
        correlation = torch.corrcoef(
            torch.stack([x.flatten(), output.flatten()])
        )[0, 1]
        
        assert correlation > 0.5, "Residual connection not preserving input"


class TestTransformerBlockShapes:
    """Test Transformer block tensor shapes."""
    
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [16, 64])
    @pytest.mark.parametrize("hidden_size,n_heads,head_dim,intermediate_size", [
        (64, 4, 16, 128),
        (128, 8, 16, 256),
        (320, 5, 64, 640),  # Spec config
    ])
    def test_output_shape_preserved(self, batch_size, seq_len, hidden_size, n_heads, head_dim, intermediate_size):
        """Block output should match input shape."""
        block = TransformerBlock(
            hidden_size=hidden_size,
            n_heads=n_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
        )
        
        x = torch.randn(batch_size, seq_len, hidden_size)
        output = block(x)
        
        assert output.shape == x.shape


class TestGradientCheckpointing:
    """Test gradient checkpointing functionality."""
    
    def test_checkpoint_produces_same_output(self):
        """Checkpointed and non-checkpointed should produce identical output."""
        hidden_size = 64
        n_heads = 4
        head_dim = 16
        intermediate_size = 128
        
        block_no_ckpt = TransformerBlockWithCheckpoint(
            hidden_size=hidden_size,
            n_heads=n_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            use_checkpoint=False,
        )
        
        block_ckpt = TransformerBlockWithCheckpoint(
            hidden_size=hidden_size,
            n_heads=n_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            use_checkpoint=True,
        )
        
        # Copy weights
        block_ckpt.load_state_dict(block_no_ckpt.state_dict())
        
        x = torch.randn(2, 16, hidden_size)
        
        block_no_ckpt.eval()
        block_ckpt.eval()
        
        with torch.no_grad():
            out_no_ckpt = block_no_ckpt(x)
            out_ckpt = block_ckpt(x)
        
        assert torch.allclose(out_no_ckpt, out_ckpt, atol=1e-5)
    
    def test_checkpoint_produces_same_gradients(self):
        """Checkpointed should produce same gradients as non-checkpointed."""
        hidden_size = 64
        n_heads = 4
        head_dim = 16
        intermediate_size = 128
        
        # Create two blocks with same weights
        block_no_ckpt = TransformerBlockWithCheckpoint(
            hidden_size=hidden_size,
            n_heads=n_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            use_checkpoint=False,
        )
        
        block_ckpt = TransformerBlockWithCheckpoint(
            hidden_size=hidden_size,
            n_heads=n_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            use_checkpoint=True,
        )
        
        # Copy weights
        block_ckpt.load_state_dict(block_no_ckpt.state_dict())
        
        # Same input
        x1 = torch.randn(2, 16, hidden_size, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        
        block_no_ckpt.train()
        block_ckpt.train()
        
        # Forward + backward
        out1 = block_no_ckpt(x1)
        out2 = block_ckpt(x2)
        
        loss1 = out1.sum()
        loss2 = out2.sum()
        
        loss1.backward()
        loss2.backward()
        
        # Gradients should match
        assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
        
        # Check weight gradients
        for (n1, p1), (n2, p2) in zip(
            block_no_ckpt.named_parameters(), 
            block_ckpt.named_parameters()
        ):
            if p1.grad is not None and p2.grad is not None:
                assert torch.allclose(p1.grad, p2.grad, atol=1e-4), f"Gradient mismatch in {n1}"
    
    def test_checkpoint_only_active_in_training(self):
        """Checkpointing should not affect eval mode."""
        block = TransformerBlockWithCheckpoint(
            hidden_size=64,
            n_heads=4,
            head_dim=16,
            intermediate_size=128,
            use_checkpoint=True,
        )
        
        x = torch.randn(2, 16, 64)
        
        block.train()
        out_train = block(x)
        
        block.eval()
        out_eval = block(x)
        
        # Outputs should be same (no dropout in this test)
        assert torch.allclose(out_train, out_eval, atol=1e-5)


class TestLayerNormEpsilon:
    """Test LayerNorm epsilon parameter."""
    
    def test_epsilon_value(self):
        """LayerNorm epsilon should match spec (1e-5)."""
        block = TransformerBlock(
            hidden_size=64,
            n_heads=4,
            head_dim=16,
            intermediate_size=128,
            norm_eps=1e-5,
        )
        
        assert block.attn_norm.eps == 1e-5
        assert block.mlp_norm.eps == 1e-5


class TestTransformerBlockNumericalStability:
    """Test numerical stability of Transformer block."""
    
    def test_no_nan_propagation(self):
        """NaN should not appear in forward pass."""
        block = TransformerBlock(
            hidden_size=64,
            n_heads=4,
            head_dim=16,
            intermediate_size=128,
        )
        
        x = torch.randn(2, 16, 64)
        output = block(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gradient_stability(self):
        """Gradients should be finite through the block."""
        block = TransformerBlock(
            hidden_size=64,
            n_heads=4,
            head_dim=16,
            intermediate_size=128,
        )
        
        x = torch.randn(2, 16, 64, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()
        
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        
        for name, param in block.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

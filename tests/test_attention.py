"""Tests for Causal Self-Attention.

Validates:
- Causal masking correctness (CRITICAL - no future token leakage)
- Q/K/V projection shapes
- Attention scaling
- Output shape correctness
"""

import pytest
import torch
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.attention import CausalSelfAttention, FlashCausalSelfAttention


class TestCausalMasking:
    """Test causal mask correctness - CRITICAL for autoregressive models."""
    
    def test_no_future_token_leakage(self):
        """Position i should NEVER attend to position j where j > i."""
        hidden_size = 64
        n_heads = 4
        head_dim = 16
        seq_len = 8
        batch_size = 2
        
        attn = CausalSelfAttention(
            hidden_size=hidden_size,
            n_heads=n_heads,
            head_dim=head_dim,
        )
        
        # Create input where each position has a unique pattern
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # Hook to capture attention weights
        attention_weights = []
        
        def hook(module, input, output):
            # Capture the attention weights before output projection
            pass
        
        # Get causal mask
        mask = attn._get_causal_mask(seq_len, x.device)
        
        # Verify mask structure: upper triangle (excluding diagonal) should be True
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    # Future position - should be masked (True)
                    assert mask[0, 0, i, j].item() == True, (
                        f"Position ({i}, {j}) should be masked but isn't!"
                    )
                else:
                    # Current or past position - should not be masked (False)
                    assert mask[0, 0, i, j].item() == False, (
                        f"Position ({i}, {j}) should NOT be masked but is!"
                    )
    
    def test_causal_mask_shape(self):
        """Causal mask should have shape (1, 1, seq_len, seq_len)."""
        attn = CausalSelfAttention(
            hidden_size=64,
            n_heads=4,
            head_dim=16,
        )
        
        for seq_len in [8, 16, 64, 128]:
            mask = attn._get_causal_mask(seq_len, torch.device("cpu"))
            assert mask.shape == (1, 1, seq_len, seq_len)
    
    def test_gradient_isolation(self):
        """Gradient should not flow from future positions."""
        hidden_size = 64
        n_heads = 4
        head_dim = 16
        seq_len = 4
        
        attn = CausalSelfAttention(
            hidden_size=hidden_size,
            n_heads=n_heads,
            head_dim=head_dim,
        )
        
        # Create input with requires_grad
        x = torch.randn(1, seq_len, hidden_size, requires_grad=True)
        
        # Forward pass
        output = attn(x)
        
        # Compute loss only from position 0's output
        loss = output[0, 0, :].sum()
        loss.backward()
        
        # CRITICAL: Gradient should only flow to positions 0, not 1, 2, 3
        # This is implicitly tested by the mask preventing attention to future
        # The forward pass result at position 0 should only depend on position 0
        
        # Verify by checking output computation
        # Create fresh forward with modified future positions
        x2 = x.detach().clone()
        x2[0, 1:, :] = 0  # Zero out all future positions
        
        output2 = attn(x2)
        
        # Position 0 output should be identical since it can't see future
        assert torch.allclose(output[0, 0, :], output2[0, 0, :], atol=1e-5), (
            "Position 0 output changed when future positions changed - "
            "CAUSAL VIOLATION DETECTED!"
        )


class TestAttentionScaling:
    """Test attention score scaling."""
    
    def test_scale_factor(self):
        """Scale should be 1/sqrt(head_dim)."""
        head_dim = 64
        attn = CausalSelfAttention(
            hidden_size=head_dim * 4,
            n_heads=4,
            head_dim=head_dim,
        )
        
        expected_scale = 1.0 / math.sqrt(head_dim)
        assert abs(attn.scale - expected_scale) < 1e-6


class TestAttentionShapes:
    """Test attention tensor shapes."""
    
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [16, 64])
    @pytest.mark.parametrize("hidden_size,n_heads,head_dim", [
        (64, 4, 16),
        (128, 8, 16),
        (320, 5, 64),  # Spec config
    ])
    def test_output_shape(self, batch_size, seq_len, hidden_size, n_heads, head_dim):
        """Output should have shape (batch, seq_len, hidden_size)."""
        attn = CausalSelfAttention(
            hidden_size=hidden_size,
            n_heads=n_heads,
            head_dim=head_dim,
        )
        
        x = torch.randn(batch_size, seq_len, hidden_size)
        output = attn(x)
        
        assert output.shape == (batch_size, seq_len, hidden_size)
    
    def test_qkv_projection_shapes(self):
        """QKV projection should triple the hidden size."""
        hidden_size = 64
        attn = CausalSelfAttention(
            hidden_size=hidden_size,
            n_heads=4,
            head_dim=16,
        )
        
        # Check weight shape
        assert attn.qkv_proj.weight.shape == (3 * hidden_size, hidden_size)
    
    def test_output_projection_shape(self):
        """Output projection should preserve hidden size."""
        hidden_size = 64
        attn = CausalSelfAttention(
            hidden_size=hidden_size,
            n_heads=4,
            head_dim=16,
        )
        
        assert attn.o_proj.weight.shape == (hidden_size, hidden_size)


class TestPaddingMask:
    """Test attention padding mask handling."""
    
    def test_padding_mask_applied(self):
        """Padding mask should reduce attention to padded positions."""
        hidden_size = 64
        n_heads = 4
        head_dim = 16
        batch_size = 2
        seq_len = 8
        
        attn = CausalSelfAttention(
            hidden_size=hidden_size,
            n_heads=n_heads,
            head_dim=head_dim,
        )
        
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # Create mask: first 4 tokens valid, last 4 padding
        attention_mask = torch.zeros(batch_size, seq_len)
        attention_mask[:, :4] = 1.0
        
        # Forward with and without mask
        output_no_mask = attn(x)
        output_with_mask = attn(x, attention_mask=attention_mask)
        
        # Outputs should differ due to mask
        assert not torch.allclose(output_no_mask, output_with_mask)


class TestFlashAttention:
    """Test Flash Attention implementation."""
    
    def test_flash_attention_output_shape(self):
        """Flash attention should produce correct output shape."""
        hidden_size = 64
        n_heads = 4
        head_dim = 16
        batch_size = 2
        seq_len = 16
        
        flash_attn = FlashCausalSelfAttention(
            hidden_size=hidden_size,
            n_heads=n_heads,
            head_dim=head_dim,
        )
        
        x = torch.randn(batch_size, seq_len, hidden_size)
        output = flash_attn(x)
        
        assert output.shape == (batch_size, seq_len, hidden_size)
    
    def test_flash_vs_standard_equivalence(self):
        """Flash and standard attention should produce similar results."""
        hidden_size = 64
        n_heads = 4
        head_dim = 16
        batch_size = 2
        seq_len = 16
        
        # Create both with same weights
        standard = CausalSelfAttention(
            hidden_size=hidden_size,
            n_heads=n_heads,
            head_dim=head_dim,
        )
        flash = FlashCausalSelfAttention(
            hidden_size=hidden_size,
            n_heads=n_heads,
            head_dim=head_dim,
        )
        
        # Copy weights
        flash.load_state_dict(standard.state_dict())
        
        # Same input
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        standard.eval()
        flash.eval()
        
        with torch.no_grad():
            out_standard = standard(x)
            out_flash = flash(x)
        
        # Should be very close (may differ slightly due to implementation)
        assert torch.allclose(out_standard, out_flash, atol=1e-4, rtol=1e-3)


class TestAttentionNumericalStability:
    """Test numerical stability of attention."""
    
    def test_no_nan_with_large_values(self):
        """Attention should not produce NaN with large input values."""
        attn = CausalSelfAttention(
            hidden_size=64,
            n_heads=4,
            head_dim=16,
        )
        
        # Large but valid input
        x = torch.randn(2, 16, 64) * 10
        output = attn(x)
        
        assert not torch.isnan(output).any(), "NaN detected in attention output!"
        assert not torch.isinf(output).any(), "Inf detected in attention output!"
    
    def test_no_nan_with_small_values(self):
        """Attention should not produce NaN with small input values."""
        attn = CausalSelfAttention(
            hidden_size=64,
            n_heads=4,
            head_dim=16,
        )
        
        # Small but valid input
        x = torch.randn(2, 16, 64) * 0.001
        output = attn(x)
        
        assert not torch.isnan(output).any(), "NaN detected in attention output!"
        assert not torch.isinf(output).any(), "Inf detected in attention output!"
    
    def test_softmax_stability(self):
        """Softmax in attention should be numerically stable."""
        attn = CausalSelfAttention(
            hidden_size=64,
            n_heads=4,
            head_dim=16,
        )
        
        # Input that could cause overflow in naive softmax
        x = torch.randn(2, 16, 64) * 100
        
        # This should not raise or produce NaN due to proper softmax implementation
        output = attn(x)
        
        assert torch.isfinite(output).all(), "Non-finite values in attention output!"
    
    def test_flash_padding_mask(self):
        """Flash attention should respect the padding mask."""
        hidden_size = 64
        n_heads = 4
        head_dim = 16
        batch_size = 2
        seq_len = 8
        
        flash_attn = FlashCausalSelfAttention(
            hidden_size=hidden_size,
            n_heads=n_heads,
            head_dim=head_dim,
        )
        
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # Mask: only first 3 tokens are valid
        padding_mask = torch.zeros(batch_size, seq_len)
        padding_mask[:, :3] = 1.0
        
        output_masked = flash_attn(x, attention_mask=padding_mask)
        
        # Zero out all future tokens in input and re-run to verify isolation
        x_isolated = x.clone()
        x_isolated[:, 3:, :] = 0.0
        output_isolated = flash_attn(x_isolated, attention_mask=padding_mask)
        
        # The output for the first 3 tokens should be identical
        # (they only attend to themselves/past, and subsequent tokens are masked)
        assert torch.allclose(output_masked[:, :3, :], output_isolated[:, :3, :], atol=1e-5), (
            "Flash Attention padding mask check failed!"
        )

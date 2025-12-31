"""Tests for Rotary Position Embeddings (RoPE).

Validates:
- Mathematical correctness of rotation formula
- Cache behavior for increasing sequence lengths
- Shape preservation through forward pass
- Frequency band computation
"""

import pytest
import torch
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.embeddings import RotaryPositionEmbedding, precompute_rope_cache


class TestRoPEMathematics:
    """Test the mathematical properties of RoPE."""
    
    def test_inv_freq_computation(self):
        """Inverse frequencies should follow theta^(-2i/d) formula."""
        dim = 64
        theta = 10000.0
        rope = RotaryPositionEmbedding(dim=dim, theta=theta)
        
        # Expected: 1 / (theta^(2i/d)) for i = 0, 2, 4, ...
        expected = []
        for i in range(0, dim, 2):
            expected.append(1.0 / (theta ** (i / dim)))
        expected = torch.tensor(expected)
        
        assert torch.allclose(rope.inv_freq, expected, rtol=1e-5)
    
    def test_rotate_half_correctness(self):
        """_rotate_half should swap and negate halves: (-x2, x1)."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 1, 1, 4)
        
        rope = RotaryPositionEmbedding(dim=4)
        rotated = rope._rotate_half(x)
        
        # Expected: [-x2, x1] = [-3, -4, 1, 2]
        expected = torch.tensor([-3.0, -4.0, 1.0, 2.0]).view(1, 1, 1, 4)
        
        assert torch.allclose(rotated, expected)
    
    def test_rotation_formula(self):
        """Forward should implement x * cos + rotate_half(x) * sin."""
        dim = 4
        rope = RotaryPositionEmbedding(dim=dim, max_seq_len=8)
        
        # Create simple input
        batch, n_heads, seq_len = 1, 1, 2
        q = torch.ones(batch, n_heads, seq_len, dim)
        k = torch.ones(batch, n_heads, seq_len, dim)
        
        q_rot, k_rot = rope(q, k)
        
        # Force cache update
        rope._update_cos_sin_cache(seq_len, q.device, q.dtype)
        cos = rope._cos_cached[:, :, :seq_len, :]
        sin = rope._sin_cached[:, :, :seq_len, :]
        
        # Manual computation
        q_expected = q * cos + rope._rotate_half(q) * sin
        
        assert torch.allclose(q_rot, q_expected, rtol=1e-5)


class TestRoPEShapes:
    """Test that RoPE preserves tensor shapes correctly."""
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("seq_len", [16, 64, 128])
    @pytest.mark.parametrize("n_heads", [1, 4, 8])
    @pytest.mark.parametrize("head_dim", [32, 64])
    def test_output_shape_preserved(self, batch_size, seq_len, n_heads, head_dim):
        """Output shape should match input shape."""
        rope = RotaryPositionEmbedding(dim=head_dim, max_seq_len=256)
        
        q = torch.randn(batch_size, n_heads, seq_len, head_dim)
        k = torch.randn(batch_size, n_heads, seq_len, head_dim)
        
        q_rot, k_rot = rope(q, k)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
    
    def test_dtype_preserved(self):
        """Output dtype should match input dtype (or be promoted for precision)."""
        rope = RotaryPositionEmbedding(dim=64)
        
        # Float32 should be preserved
        q32 = torch.randn(2, 4, 16, 64, dtype=torch.float32)
        k32 = torch.randn(2, 4, 16, 64, dtype=torch.float32)
        q_rot32, k_rot32 = rope(q32, k32)
        assert q_rot32.dtype == torch.float32
        
        # Note: RoPE may cast to float32 for numerical precision
        # This is acceptable behavior for half-precision inputs


class TestRoPECache:
    """Test RoPE caching behavior."""
    
    def test_cache_grows_with_seq_len(self):
        """Cache should expand when sequence length increases."""
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=1024)
        
        # First call with short sequence
        q1 = torch.randn(1, 4, 16, 64)
        k1 = torch.randn(1, 4, 16, 64)
        rope(q1, k1)
        
        assert rope._seq_len_cached == 16
        
        # Second call with longer sequence
        q2 = torch.randn(1, 4, 64, 64)
        k2 = torch.randn(1, 4, 64, 64)
        rope(q2, k2)
        
        assert rope._seq_len_cached == 64
    
    def test_cache_reused_for_shorter_seq(self):
        """Cache should be reused, not recomputed, for shorter sequences."""
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=1024)
        
        # First call with long sequence
        q1 = torch.randn(1, 4, 64, 64)
        k1 = torch.randn(1, 4, 64, 64)
        rope(q1, k1)
        
        cached_len_after_long = rope._seq_len_cached
        
        # Second call with shorter sequence
        q2 = torch.randn(1, 4, 16, 64)
        k2 = torch.randn(1, 4, 16, 64)
        rope(q2, k2)
        
        # Cache should not shrink
        assert rope._seq_len_cached == cached_len_after_long


class TestRoPERelativePosition:
    """Test that RoPE encodes relative positions."""
    
    def test_dot_product_depends_on_distance(self):
        """Dot product of rotated vectors should depend on relative position."""
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=64)
        
        # Create identical vectors at different positions
        batch, n_heads, head_dim = 1, 1, 64
        
        q = torch.randn(batch, n_heads, 4, head_dim)
        k = q.clone()
        
        q_rot, k_rot = rope(q, k)
        
        # Dot product at same position (distance 0)
        dot_0 = (q_rot[0, 0, 0] * k_rot[0, 0, 0]).sum()
        
        # Dot product at distance 1
        dot_1 = (q_rot[0, 0, 0] * k_rot[0, 0, 1]).sum()
        
        # Dot product at distance 2
        dot_2 = (q_rot[0, 0, 0] * k_rot[0, 0, 2]).sum()
        
        # Self dot product should be largest
        assert dot_0 > dot_1
        assert dot_0 > dot_2
    
    def test_position_invariance(self):
        """Same relative position should give same dot product ratio."""
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=64)
        
        batch, n_heads, head_dim = 1, 1, 64
        
        # Create random vectors
        vec = torch.randn(head_dim)
        q = vec.view(1, 1, 1, head_dim).expand(1, 1, 10, head_dim).clone()
        k = q.clone()
        
        q_rot, k_rot = rope(q, k)
        
        # Distance-1 dot product from position 0→1
        dot_01 = (q_rot[0, 0, 0] * k_rot[0, 0, 1]).sum()
        
        # Distance-1 dot product from position 5→6
        dot_56 = (q_rot[0, 0, 5] * k_rot[0, 0, 6]).sum()
        
        # Should be approximately equal (RoPE encodes relative position)
        assert torch.allclose(dot_01, dot_56, rtol=0.1)


class TestPrecomputeRoPECache:
    """Test the standalone cache precomputation function."""
    
    def test_precompute_shapes(self):
        """Precomputed cache should have correct shapes."""
        dim = 64
        max_seq_len = 512
        
        cos_cache, sin_cache = precompute_rope_cache(dim, max_seq_len)
        
        assert cos_cache.shape == (1, 1, max_seq_len, dim)
        assert sin_cache.shape == (1, 1, max_seq_len, dim)
    
    def test_precompute_values(self):
        """Precomputed values should match RoPE module."""
        dim = 64
        max_seq_len = 128
        theta = 10000.0
        
        # Precompute
        cos_cache, sin_cache = precompute_rope_cache(dim, max_seq_len, theta)
        
        # Create RoPE module and trigger cache
        rope = RotaryPositionEmbedding(dim, max_seq_len, theta)
        q = torch.randn(1, 1, max_seq_len, dim)
        k = torch.randn(1, 1, max_seq_len, dim)
        rope(q, k)
        
        assert torch.allclose(cos_cache, rope._cos_cached, rtol=1e-5)
        assert torch.allclose(sin_cache, rope._sin_cached, rtol=1e-5)

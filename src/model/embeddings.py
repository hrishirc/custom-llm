"""Rotary Position Embeddings (RoPE) implementation.

RoPE applies a rotation to query and key vectors based on position,
enabling relative position encoding without learned embeddings.
"""

import torch
import torch.nn as nn
from typing import Tuple


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).
    
    Applies rotation matrices to Q and K based on their positions.
    This enables relative position encoding implicitly through
    the dot product of rotated vectors.
    
    Args:
        dim: Dimension of each attention head
        max_seq_len: Maximum sequence length to precompute
        theta: Base for the frequency computation (default: 10000)
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequency bands
        # inv_freq shape: (dim // 2,)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        
        # Cache for cos/sin values
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
    
    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cached cos/sin values if sequence length increased."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            
            # Position indices: (seq_len,)
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            
            # Outer product: (seq_len, dim // 2)
            freqs = torch.outer(t, self.inv_freq.to(device))
            
            # Concatenate to get full dimension: (seq_len, dim)
            emb = torch.cat((freqs, freqs), dim=-1)
            
            # Cache cos and sin: (1, 1, seq_len, dim)
            self._cos_cached = emb.cos().to(dtype)[None, None, :, :]
            self._sin_cached = emb.sin().to(dtype)[None, None, :, :]
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors.
        
        Args:
            q: Query tensor of shape (batch, n_heads, seq_len, head_dim)
            k: Key tensor of shape (batch, n_heads, seq_len, head_dim)
            position_ids: Optional position indices (batch, seq_len)
        
        Returns:
            Tuple of rotated (q, k) tensors
        """
        seq_len = q.shape[2]
        
        # Update cache if needed
        self._update_cos_sin_cache(seq_len, q.device, q.dtype)
        
        # Get cos/sin for this sequence length
        cos = self._cos_cached[:, :, :seq_len, :]
        sin = self._sin_cached[:, :, :seq_len, :]
        
        # Apply rotation
        q_rot = self._apply_rotary(q, cos, sin)
        k_rot = self._apply_rotary(k, cos, sin)
        
        return q_rot, k_rot
    
    def _apply_rotary(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary embedding to a single tensor.
        
        Uses the rotation formula:
        x_rot = x * cos + rotate_half(x) * sin
        
        where rotate_half swaps and negates half the dimensions.
        """
        # x shape: (batch, n_heads, seq_len, head_dim)
        x_rot = (x * cos) + (self._rotate_half(x) * sin)
        return x_rot
    
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input.
        
        For input [..., d], returns [..., d] where:
        - First half is negated second half
        - Second half is first half
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


def precompute_rope_cache(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin cache for a given configuration.
    
    This is useful for pre-allocating the cache before training.
    
    Returns:
        Tuple of (cos_cache, sin_cache) tensors of shape (1, 1, max_seq_len, dim)
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    
    cos_cache = emb.cos().to(dtype)[None, None, :, :]
    sin_cache = emb.sin().to(dtype)[None, None, :, :]
    
    return cos_cache, sin_cache

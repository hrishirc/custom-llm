"""Causal Multi-Head Self-Attention module.

Implements efficient causal attention with:
- RoPE positional encoding
- Causal masking (no future token attention)
- Optional gradient checkpointing
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .embeddings import RotaryPositionEmbedding


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE.
    
    Args:
        hidden_size: Model hidden dimension
        n_heads: Number of attention heads
        head_dim: Dimension of each attention head
        dropout: Dropout probability for attention weights
        rope_theta: Base theta for RoPE
        max_seq_len: Maximum sequence length for RoPE cache
    """
    
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dropout = dropout
        
        # Ensure dimensions match
        assert hidden_size == n_heads * head_dim
        
        # Scaling factor for dot-product attention
        self.scale = 1.0 / math.sqrt(head_dim)
        
        # Q, K, V projections (combined for efficiency)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # RoPE
        self.rope = RotaryPositionEmbedding(
            dim=head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
        )
        
        # Dropout (applied to attention weights)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Causal mask (registered as buffer, expanded on demand)
        self._causal_mask = None
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or create causal attention mask.
        
        Returns a mask of shape (1, 1, seq_len, seq_len) where
        True indicates positions that should be masked (not attended to).
        """
        if self._causal_mask is None or self._causal_mask.size(-1) < seq_len:
            # Create upper triangular mask (True = masked)
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                diagonal=1,
            )
            self._causal_mask = mask[None, None, :, :]
        
        return self._causal_mask[:, :, :seq_len, :seq_len]
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for causal self-attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Optional mask for padding (batch, seq_len)
                           1 = attend, 0 = mask
        
        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * hidden)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, n_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE to Q and K
        q, k = self.rope(q, k)
        
        # Compute attention scores
        # (batch, n_heads, seq_len, seq_len)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        causal_mask = self._get_causal_mask(seq_len, x.device)
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
        
        # Apply padding mask if provided
        if attention_mask is not None:
            # attention_mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            padding_mask = (1 - attention_mask[:, None, None, :]) * float("-inf")
            attn_weights = attn_weights + padding_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        # (batch, n_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch, seq_len, n_heads, head_dim)
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)
        
        return output


class FlashCausalSelfAttention(CausalSelfAttention):
    """Causal self-attention using PyTorch's scaled_dot_product_attention.
    
    Uses the efficient fused implementation when available (PyTorch 2.0+).
    Falls back to standard attention otherwise.
    """
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass using scaled_dot_product_attention."""
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE
        q, k = self.rope(q, k)
        
        # Handle masks
        attn_mask = None
        if attention_mask is not None:
            # attention_mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            # 1 = attend, 0 = mask. F.sdpa expects True to keep, False to mask OR an additive mask.
            # Here we create a boolean mask (batch, 1, 1, seq_len)
            attn_mask = attention_mask[:, None, None, :].to(torch.bool)
            
            # When using attn_mask with is_causal=True, PyTorch 2.1+ supports this
            # but it must be a boolean mask and is_causal=True will still apply the causal mask.
        
        # Use PyTorch's efficient attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)
        
        return output

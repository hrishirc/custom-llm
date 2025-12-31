"""Transformer block with Pre-LN architecture.

Pre-LayerNorm is critical for training stability in deep (60-layer) models.
Each block contains: Attention + MLP with residual connections.
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import CausalSelfAttention, FlashCausalSelfAttention
from .mlp import MLP


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block.
    
    Architecture:
        x -> LayerNorm -> Attention -> + -> LayerNorm -> MLP -> +
             └──────────────────────────┘   └────────────────────┘
    
    Pre-LN applies normalization before each sublayer, which is
    critical for training stability in very deep models.
    
    Args:
        hidden_size: Model hidden dimension
        n_heads: Number of attention heads
        head_dim: Dimension per attention head
        intermediate_size: MLP intermediate dimension
        dropout: Dropout probability
        attention_dropout: Dropout for attention weights
        rope_theta: Base theta for RoPE
        max_seq_len: Maximum sequence length
        use_flash_attention: Use efficient attention implementation
        layer_idx: Layer index (for logging/debugging)
    """
    
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        head_dim: int,
        intermediate_size: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
        use_flash_attention: bool = True,
        layer_idx: int = 0,
    ):
        super().__init__()
        
        self.layer_idx = layer_idx
        
        # Pre-attention LayerNorm
        self.attn_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        
        # Self-attention
        AttentionClass = FlashCausalSelfAttention if use_flash_attention else CausalSelfAttention
        self.attention = AttentionClass(
            hidden_size=hidden_size,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=attention_dropout,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
        )
        
        # Pre-MLP LayerNorm
        self.mlp_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        
        # MLP
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the Transformer block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Optional padding mask (batch, seq_len)
        
        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        # Attention with residual
        residual = x
        x = self.attn_norm(x)
        x = self.attention(x, attention_mask)
        x = residual + x
        
        # MLP with residual
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class TransformerBlockWithCheckpoint(TransformerBlock):
    """Transformer block with gradient checkpointing support.
    
    Gradient checkpointing trades compute for memory by not storing
    intermediate activations during forward pass. This is essential
    for training 60-layer models on laptop-class hardware.
    """
    
    def __init__(self, *args, use_checkpoint: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_checkpoint = use_checkpoint
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional gradient checkpointing."""
        if self.use_checkpoint and self.training:
            # Use gradient checkpointing
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl,
                x,
                attention_mask,
                use_reentrant=False,
            )
        else:
            return self._forward_impl(x, attention_mask)
    
    def _forward_impl(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Actual forward implementation (called by checkpoint)."""
        # Attention with residual
        residual = x
        x = self.attn_norm(x)
        x = self.attention(x, attention_mask)
        x = residual + x
        
        # MLP with residual
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x

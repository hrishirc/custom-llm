"""MLP (Feed-Forward) block for the Transformer.

Implements a standard 2-layer MLP with GELU activation.
Can be extended to SwiGLU in future iterations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Feed-forward MLP block.
    
    Standard Transformer MLP with:
    - Up-projection: hidden_size → intermediate_size
    - GELU activation
    - Down-projection: intermediate_size → hidden_size
    
    Args:
        hidden_size: Input/output dimension
        intermediate_size: Hidden dimension (typically 2-4× hidden_size)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Up-projection
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        
        # Down-projection
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
        
        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        # Up-project and activate
        hidden = self.up_proj(x)
        hidden = F.gelu(hidden, approximate="tanh")
        
        # Down-project and dropout
        output = self.down_proj(hidden)
        output = self.dropout(output)
        
        return output


class SwiGLUMLP(nn.Module):
    """SwiGLU-based MLP block (optional enhancement).
    
    SwiGLU uses a gated linear unit with SiLU (Swish) activation,
    which has shown improved performance in modern LLMs.
    
    Architecture:
    - Gate projection: hidden_size → intermediate_size
    - Up projection: hidden_size → intermediate_size
    - Element-wise: SiLU(gate) * up
    - Down projection: intermediate_size → hidden_size
    
    Note: This adds ~50% more parameters per MLP block.
    Only use if parameter budget allows.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Gate and up projections (combined for efficiency)
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        
        # Down projection
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SwiGLU activation."""
        # Combined gate and up projection
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        
        # SwiGLU: SiLU(gate) * up
        hidden = F.silu(gate) * up
        
        # Down-project and dropout
        output = self.down_proj(hidden)
        output = self.dropout(output)
        
        return output

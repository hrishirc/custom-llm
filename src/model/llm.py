"""Complete LLM model implementation.

Assembles all components into a full autoregressive language model:
- Token embeddings
- 60 Transformer blocks with RoPE
- Final LayerNorm
- LM Head (optionally tied to embeddings)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .config import ModelConfig
from .transformer import TransformerBlockWithCheckpoint


class LLM(nn.Module):
    """60M Parameter Language Model.
    
    Architecture:
        Token IDs → Embedding → [Transformer Block × 60] → LayerNorm → LM Head
    
    Features:
        - RoPE positional encoding (no learned positions)
        - Pre-LN Transformer blocks for deep model stability
        - Gradient checkpointing for memory efficiency
        - Weight tying between embeddings and LM head
        - Depth-scaled initialization for 60-layer training
    
    Args:
        config: ModelConfig with all architecture parameters
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        
        # Dropout after embeddings
        self.embedding_dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlockWithCheckpoint(
                hidden_size=config.hidden_size,
                n_heads=config.n_heads,
                head_dim=config.head_dim,
                intermediate_size=config.intermediate_size,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                norm_eps=config.norm_eps,
                rope_theta=config.rope_theta,
                max_seq_len=config.max_seq_len,
                use_flash_attention=True,
                use_checkpoint=config.use_gradient_checkpointing,
                layer_idx=i,
            )
            for i in range(config.n_layers)
        ])
        
        # Final LayerNorm
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        
        # LM Head
        if config.tie_word_embeddings:
            # Share weights with token embeddings
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        self._init_residual_scaling()
        
        # Track frozen layers
        self._frozen_layers = 0
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights using standard LLM practices."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def _init_residual_scaling(self):
        """Apply depth-scaled initialization to residual projections.
        
        This prevents gradient explosion in deep models by scaling
        the output projections by 1/√(2*n_layers).
        """
        residual_scale = self.config.residual_scale
        
        for layer in self.layers:
            # Scale attention output projection
            torch.nn.init.normal_(
                layer.attention.o_proj.weight,
                mean=0.0,
                std=residual_scale,
            )
            # Scale MLP down projection
            torch.nn.init.normal_(
                layer.mlp.down_proj.weight,
                mean=0.0,
                std=residual_scale,
            )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Padding mask of shape (batch, seq_len), 1=attend, 0=mask
            labels: Target token IDs for loss computation (batch, seq_len)
                   Typically input_ids shifted left by 1
        
        Returns:
            Dictionary with:
                - logits: (batch, seq_len, vocab_size)
                - loss: scalar if labels provided
        """
        # Token embeddings
        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Transformer blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Final normalization
        hidden_states = self.final_norm(hidden_states)
        
        # LM Head
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden_states, self.token_embedding.weight)
        else:
            logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        result = {"logits": logits}
        
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )
            result["loss"] = loss
        
        return result
    
    def freeze_layers(self, n_layers: int):
        """Freeze the bottom N layers for efficient training.
        
        This reduces memory and compute by not computing gradients
        for early layers that have already converged.
        
        Args:
            n_layers: Number of bottom layers to freeze
        """
        # Unfreeze all first
        for param in self.parameters():
            param.requires_grad = True
        
        # Freeze specified layers
        for i in range(min(n_layers, len(self.layers))):
            for param in self.layers[i].parameters():
                param.requires_grad = False
        
        self._frozen_layers = n_layers
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        """Count model parameters.
        
        Args:
            non_embedding: If True, exclude embedding parameters
        
        Returns:
            Total parameter count
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate text autoregressively.
        
        Args:
            input_ids: Starting token IDs (batch, seq_len)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k filtering (0 = disabled)
            top_p: Nucleus sampling threshold
            eos_token_id: Stop token (defaults to config)
        
        Returns:
            Generated token IDs (batch, seq_len + new_tokens)
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop to max sequence length if needed
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            
            # Forward pass
            outputs = self(idx_cond)
            logits = outputs["logits"][:, -1, :]  # Last position
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check for EOS
            if (next_token == eos_token_id).all():
                break
        
        return input_ids


def create_model(config: Optional[ModelConfig] = None) -> LLM:
    """Factory function to create the LLM.
    
    Args:
        config: ModelConfig, uses defaults if not provided
    
    Returns:
        Initialized LLM model
    """
    if config is None:
        config = ModelConfig()
    
    model = LLM(config)
    
    # Print parameter count
    param_info = config.count_parameters()
    print(f"Model created with {param_info['total_millions']:.2f}M parameters")
    print(f"  - Embeddings: {param_info['embeddings'] / 1e6:.2f}M")
    print(f"  - Body ({config.n_layers} layers): {param_info['body'] / 1e6:.2f}M")
    
    return model

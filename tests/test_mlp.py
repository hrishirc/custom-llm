"""Tests for MLP (Feed-Forward) blocks.

Validates:
- Forward pass shapes
- GELU activation behavior
- SwiGLU gate mechanism
- Expansion/contraction ratios
"""

import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.mlp import MLP, SwiGLUMLP


class TestMLPShapes:
    """Test MLP tensor shapes."""
    
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [16, 64])
    @pytest.mark.parametrize("hidden_size,intermediate_size", [
        (64, 128),
        (128, 256),
        (320, 640),  # Spec config
    ])
    def test_output_shape_unchanged(self, batch_size, seq_len, hidden_size, intermediate_size):
        """MLP output should have same shape as input."""
        mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        
        x = torch.randn(batch_size, seq_len, hidden_size)
        output = mlp(x)
        
        assert output.shape == x.shape
    
    def test_weight_shapes(self):
        """Weight matrices should have correct dimensions."""
        hidden = 64
        intermediate = 128
        mlp = MLP(hidden_size=hidden, intermediate_size=intermediate)
        
        assert mlp.up_proj.weight.shape == (intermediate, hidden)
        assert mlp.down_proj.weight.shape == (hidden, intermediate)
    
    def test_no_bias(self):
        """MLP should not have biases (as per spec)."""
        mlp = MLP(hidden_size=64, intermediate_size=128)
        
        assert mlp.up_proj.bias is None
        assert mlp.down_proj.bias is None


class TestGELUActivation:
    """Test GELU activation in MLP."""
    
    def test_gelu_applied(self):
        """MLP should apply GELU activation."""
        mlp = MLP(hidden_size=64, intermediate_size=128)
        
        # Create input where GELU behavior is testable
        x = torch.randn(1, 1, 64)
        
        # Get intermediate representation
        with torch.no_grad():
            intermediate = mlp.up_proj(x)
            expected_activated = F.gelu(intermediate, approximate="tanh")
            expected_output = mlp.down_proj(expected_activated)
        
        output = mlp(x)
        
        assert torch.allclose(output, expected_output, atol=1e-5)
    
    def test_gelu_nonlinearity(self):
        """GELU should introduce nonlinearity."""
        mlp = MLP(hidden_size=64, intermediate_size=128)
        
        # Linear test: 2*x should not equal 2*f(x)
        x1 = torch.randn(1, 16, 64)
        x2 = 2 * x1
        
        out1 = mlp(x1)
        out2 = mlp(x2)
        
        # If linear, out2 would equal 2*out1
        # GELU ensures this is false
        assert not torch.allclose(2 * out1, out2, atol=0.1)


class TestMLPDropout:
    """Test dropout in MLP."""
    
    def test_dropout_applied_in_training(self):
        """Dropout should cause different outputs in training mode."""
        mlp = MLP(hidden_size=64, intermediate_size=128, dropout=0.3)
        mlp.train()
        
        x = torch.randn(1, 16, 64)
        
        outputs = []
        for _ in range(10):
            outputs.append(mlp(x))
        
        # At least some outputs should differ due to dropout
        all_same = all(torch.allclose(outputs[0], o, atol=1e-5) for o in outputs[1:])
        assert not all_same, "Dropout not active in training mode"
    
    def test_no_dropout_in_eval(self):
        """Dropout should be disabled in eval mode."""
        mlp = MLP(hidden_size=64, intermediate_size=128, dropout=0.3)
        mlp.eval()
        
        x = torch.randn(1, 16, 64)
        
        outputs = [mlp(x) for _ in range(3)]
        
        assert all(torch.allclose(outputs[0], o) for o in outputs[1:])


class TestSwiGLUMLP:
    """Test SwiGLU variant of MLP."""
    
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [16, 64])
    def test_swiglu_output_shape(self, batch_size, seq_len):
        """SwiGLU should produce same output shape as input."""
        hidden_size = 64
        intermediate_size = 128
        
        swiglu = SwiGLUMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        
        x = torch.randn(batch_size, seq_len, hidden_size)
        output = swiglu(x)
        
        assert output.shape == x.shape
    
    def test_swiglu_gate_mechanism(self):
        """SwiGLU should apply gate: SiLU(gate) * up."""
        hidden = 64
        intermediate = 128
        
        swiglu = SwiGLUMLP(hidden_size=hidden, intermediate_size=intermediate)
        
        x = torch.randn(1, 1, hidden)
        
        with torch.no_grad():
            # Manual computation
            gate_up = swiglu.gate_up_proj(x)
            gate, up = gate_up.chunk(2, dim=-1)
            hidden_manual = F.silu(gate) * up
            expected = swiglu.down_proj(hidden_manual)
        
        output = swiglu(x)
        
        assert torch.allclose(output, expected, atol=1e-5)
    
    def test_swiglu_weight_shapes(self):
        """SwiGLU gate_up_proj should be 2x intermediate."""
        hidden = 64
        intermediate = 128
        
        swiglu = SwiGLUMLP(hidden_size=hidden, intermediate_size=intermediate)
        
        # gate_up_proj: hidden -> 2*intermediate
        assert swiglu.gate_up_proj.weight.shape == (2 * intermediate, hidden)
        # down_proj: intermediate -> hidden
        assert swiglu.down_proj.weight.shape == (hidden, intermediate)


class TestMLPNumericalStability:
    """Test numerical stability of MLP."""
    
    def test_no_nan_with_large_input(self):
        """MLP should handle large inputs without NaN."""
        mlp = MLP(hidden_size=64, intermediate_size=128)
        
        x = torch.randn(2, 16, 64) * 10
        output = mlp(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_no_nan_with_small_input(self):
        """MLP should handle small inputs without NaN."""
        mlp = MLP(hidden_size=64, intermediate_size=128)
        
        x = torch.randn(2, 16, 64) * 0.001
        output = mlp(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gradient_flow(self):
        """Gradients should flow through MLP without explosion."""
        mlp = MLP(hidden_size=64, intermediate_size=128)
        
        x = torch.randn(2, 16, 64, requires_grad=True)
        output = mlp(x)
        loss = output.sum()
        loss.backward()
        
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()


class TestMLPExpansionRatio:
    """Test MLP intermediate expansion."""
    
    def test_2x_expansion_as_per_spec(self):
        """Spec requires 2× MLP ratio: intermediate = 2 * hidden."""
        hidden = 320  # Spec value
        intermediate = 640  # 2 × 320
        
        mlp = MLP(hidden_size=hidden, intermediate_size=intermediate)
        
        # Verify expansion in up_proj
        assert mlp.up_proj.weight.shape[0] == 2 * mlp.up_proj.weight.shape[1]

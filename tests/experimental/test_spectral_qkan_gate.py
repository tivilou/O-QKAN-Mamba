import pytest
import torch

from src.qkan.experimental.spectral_qkan_gate import SpectralQKANGate


@pytest.fixture
def gate():
    return SpectralQKANGate(d_model=32, latent_dim=8, reps=2, solver="exact", device="cpu")


def test_gate_shape(gate):
    x = torch.randn(2, 10, 32)
    out = gate(x)
    assert out.shape == x.shape


def test_gate_values_in_range(gate):
    x = torch.randn(4, 8, 32)
    out = gate(x)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_gate_backward(gate):
    x = torch.randn(2, 5, 32, requires_grad=True)
    out = gate(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_qkan_params_receive_gradient(gate):
    x = torch.randn(2, 5, 32)
    out = gate(x)
    loss = out.sum()
    loss.backward()
    theta = gate.qkan.layers[0].theta
    assert theta.grad is not None
    assert theta.grad.abs().sum() > 0

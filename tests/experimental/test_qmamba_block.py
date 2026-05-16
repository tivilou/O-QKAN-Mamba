import pytest
import torch

from src.qkan.experimental.qmamba_block import QMambaBlock, _MAMBA_AVAILABLE


@pytest.fixture
def block_cpu():
    return QMambaBlock(d_model=32, latent_dim=8, reps=2, use_mamba=False, device="cpu")


def test_block_shape(block_cpu):
    x = torch.randn(2, 10, 32)
    out = block_cpu(x)
    assert out.shape == x.shape


def test_block_with_fallback_mixer(block_cpu):
    assert not block_cpu.use_mamba
    x = torch.randn(2, 10, 32)
    out = block_cpu(x)
    assert out.shape == x.shape
    assert not torch.allclose(out, x)


@pytest.mark.skipif(not _MAMBA_AVAILABLE, reason="mamba-ssm not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_block_with_mamba():
    block = QMambaBlock(d_model=32, latent_dim=8, reps=2, use_mamba=True, device="cuda")
    assert block.use_mamba
    x = torch.randn(2, 10, 32, device="cuda")
    out = block(x)
    assert out.shape == x.shape


def test_block_backward(block_cpu):
    x = torch.randn(2, 5, 32, requires_grad=True)
    out = block_cpu(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None

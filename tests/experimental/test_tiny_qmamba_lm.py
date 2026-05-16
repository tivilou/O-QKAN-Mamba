import pytest
import torch

from src.qkan.experimental.tiny_qmamba_lm import TinyQMambaLM


@pytest.fixture
def model():
    return TinyQMambaLM(
        vocab_size=100, d_model=32, n_layers=2, latent_dim=8, reps=2,
        use_mamba=False, solver="exact", device="cpu",
    )


def test_forward_shape(model):
    ids = torch.randint(0, 100, (2, 16))
    logits, loss = model(ids)
    assert logits.shape == (2, 16, 100)
    assert loss is None


def test_forward_with_targets(model):
    ids = torch.randint(0, 100, (2, 16))
    targets = torch.randint(0, 100, (2, 16))
    logits, loss = model(ids, targets)
    assert logits.shape == (2, 16, 100)
    assert loss is not None
    assert loss.item() > 0


def test_one_training_step(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ids = torch.randint(0, 100, (4, 16))
    targets = torch.randint(0, 100, (4, 16))

    losses = []
    for _ in range(5):
        _, loss = model(ids, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"

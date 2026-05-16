"""Tests for OQKANMambaBlock."""
import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.qkan.experimental.o_qkan_mamba_block import OQKANMambaBlock


class TestOQKANMambaBlock:
    @pytest.fixture
    def block(self):
        return OQKANMambaBlock(
            d_model=32, latent_dim=8, d_ontology=16, reps=2,
            use_mamba=False, device="cpu",
        )

    def test_block_forward_shape(self, block):
        x = torch.randn(2, 10, 32)
        c = torch.randn(2, 16)
        out = block(x, c)
        assert out.shape == (2, 10, 32)

    def test_block_with_seq_concepts(self, block):
        """concept_emb can be (B, L, d_ontology) for per-timestep concepts."""
        x = torch.randn(2, 10, 32)
        c = torch.randn(2, 10, 16)
        out = block(x, c)
        assert out.shape == (2, 10, 32)

    def test_different_concepts_different_outputs(self, block):
        x = torch.randn(2, 10, 32)
        c1 = torch.zeros(2, 16)
        c1[:, 0] = 1.0
        c2 = torch.zeros(2, 16)
        c2[:, 8] = 1.0

        out1 = block(x, c1)
        out2 = block(x, c2)
        diff = (out1 - out2).abs().mean()
        assert diff > 1e-4, f"Outputs too similar: diff={diff.item():.6f}"

    def test_one_training_step(self, block):
        optimizer = torch.optim.Adam(block.parameters(), lr=1e-3)
        x = torch.randn(4, 8, 32)
        c = torch.randn(4, 16)
        target = torch.randn(4, 8, 32)

        losses = []
        for _ in range(5):
            out = block(x, c)
            loss = ((out - target) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"

    def test_gradient_flows_to_W_KG(self, block):
        x = torch.randn(2, 5, 32)
        c = torch.randn(2, 16)
        out = block(x, c)
        loss = out.sum()
        loss.backward()
        W_KG = block.gate.daruan.W_KG
        assert W_KG.grad is not None
        assert W_KG.grad.abs().sum() > 0

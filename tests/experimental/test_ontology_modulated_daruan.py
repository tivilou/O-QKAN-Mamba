"""Tests for OntologyModulatedDARUAN — core soundness checks."""
import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.qkan.experimental.ontology_modulated_daruan import OntologyModulatedDARUAN
from src.qkan.experimental.ontology_modulated_qkan import OntologyModulatedQKAN


class TestOntologyModulatedDARUAN:
    @pytest.fixture
    def daruan(self):
        return OntologyModulatedDARUAN(dim=8, reps=3, d_ontology=16, device="cpu")

    def test_shape(self, daruan):
        x = torch.randn(4, 8)
        c = torch.randn(4, 16)
        out = daruan(x, c)
        assert out.shape == (4, 8)

    def test_modulation_changes_output(self, daruan):
        """Different concept_emb must produce measurably different outputs."""
        x = torch.randn(4, 8)
        c1 = torch.zeros(4, 16)
        c1[:, 0] = 1.0  # "healthy" concept
        c2 = torch.zeros(4, 16)
        c2[:, 5] = 1.0  # "septic" concept

        out1 = daruan(x, c1)
        out2 = daruan(x, c2)

        diff = (out1 - out2).abs().mean()
        assert diff > 1e-4, f"Outputs too similar: diff={diff.item():.6f}"

    def test_zero_concept_recovers_base(self, daruan):
        """With zero concept embedding, modulation = softplus(b).
        Two calls with same zero concept should give identical output."""
        x = torch.randn(4, 8)
        c_zero = torch.zeros(4, 16)

        out1 = daruan(x, c_zero)
        out2 = daruan(x, c_zero)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_gradient_flows_through_W_KG(self, daruan):
        """W_KG must receive non-zero gradients."""
        x = torch.randn(4, 8)
        c = torch.randn(4, 16)
        out = daruan(x, c)
        loss = out.sum()
        loss.backward()
        assert daruan.W_KG.grad is not None
        assert daruan.W_KG.grad.abs().sum() > 0

    def test_gradient_flows_through_theta(self, daruan):
        x = torch.randn(4, 8)
        c = torch.randn(4, 16)
        out = daruan(x, c)
        loss = out.sum()
        loss.backward()
        assert daruan.theta.grad is not None
        assert daruan.theta.grad.abs().sum() > 0

    def test_gradient_flows_through_w_base(self, daruan):
        x = torch.randn(4, 8)
        c = torch.randn(4, 16)
        out = daruan(x, c)
        loss = out.sum()
        loss.backward()
        assert daruan.w_base.grad is not None
        assert daruan.w_base.grad.abs().sum() > 0

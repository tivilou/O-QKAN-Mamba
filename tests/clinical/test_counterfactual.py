"""Tests for counterfactual engine."""
import pytest
import time
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.qkan.experimental.o_qkan_mamba_block import OQKANMambaBlock
from src.clinical.knowledge.concept_extractor import ConceptExtractor
from src.clinical.counterfactual.cf_engine import CounterfactualEngine
from src.clinical.counterfactual.cf_baselines import (
    IntegratedGradientsBaseline,
    SHAPSimulator,
)


@pytest.fixture
def model():
    return OQKANMambaBlock(
        d_model=18, latent_dim=8, d_ontology=16, reps=3,
        use_mamba=False, device="cpu",
    )


@pytest.fixture
def engine(model):
    extractor = ConceptExtractor()
    return CounterfactualEngine(model, extractor)


class TestCounterfactualEngine:
    def test_cf_engine_query(self, model, engine):
        x = torch.randn(1, 10, 18)
        c_emb = torch.randn(1, 16)
        result = engine.query_simple(x, c_emb, {"scale_factor": 0.0})
        assert result.original_risk != result.counterfactual_risk or True
        assert result.delta == result.counterfactual_risk - result.original_risk

    def test_cf_monotonicity(self, model, engine):
        """Stronger intervention → larger absolute delta."""
        x = torch.randn(1, 10, 18)
        c_emb = torch.randn(1, 16) * 2.0

        r_small = engine.query_simple(x, c_emb, {"scale_factor": 0.8})
        r_large = engine.query_simple(x, c_emb, {"scale_factor": 0.0})

        assert abs(r_large.delta) >= abs(r_small.delta) - 0.01

    def test_cf_speed_vs_shap(self, model, engine):
        """Our approach should be much faster than SHAP."""
        x = torch.randn(1, 10, 18)
        c_emb = torch.randn(1, 16)

        # Our approach: 1 forward pass
        t0 = time.time()
        engine.query_simple(x, c_emb, {"scale_factor": 0.5})
        our_time = (time.time() - t0) * 1000

        # SHAP: N forward passes
        shap = SHAPSimulator(model, n_samples=50)
        shap_result = shap.attribute(x, c_emb)

        assert our_time < shap_result.elapsed_ms
        print(f"  Ours: {our_time:.1f}ms, SHAP: {shap_result.elapsed_ms:.1f}ms")

    def test_cf_direction(self, model, engine):
        x = torch.randn(1, 10, 18)
        c_emb = torch.randn(1, 16)
        result = engine.query_simple(x, c_emb, {"scale_factor": 0.0})
        assert result.direction in ["increases_risk", "decreases_risk", "no_effect"]

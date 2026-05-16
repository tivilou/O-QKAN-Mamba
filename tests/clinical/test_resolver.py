"""Tests for Spectral Conflict Resolver."""
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.qkan.experimental.o_qkan_mamba_block import OQKANMambaBlock
from src.clinical.validator.dag_validator import DAGValidator
from src.clinical.resolver.spectral_attribution import SpectralAttributor
from src.clinical.resolver.spectral_resolver import SpectralConflictResolver
from src.clinical.resolver.hard_projection import HardProjectionResolver


@pytest.fixture
def model():
    return OQKANMambaBlock(
        d_model=18, latent_dim=8, d_ontology=16, reps=3,
        use_mamba=False, device="cpu",
    )


@pytest.fixture
def validator():
    return DAGValidator()


class TestSpectralAttributor:
    def test_attribution_shape(self, model):
        attr = SpectralAttributor(model)
        x = torch.randn(1, 10, 18)
        c = torch.randn(1, 16)
        scores = attr.compute_attribution_simple(x, c)
        assert scores.shape == (3,)  # reps=3

    def test_attribution_non_negative(self, model):
        attr = SpectralAttributor(model)
        x = torch.randn(1, 10, 18)
        c = torch.randn(1, 16)
        scores = attr.compute_attribution_simple(x, c)
        assert (scores >= 0).all()


class TestSpectralResolver:
    def test_resolver_reduces_violations(self, model, validator):
        resolver = SpectralConflictResolver(
            model, validator, max_iterations=5, dampening_factor=0.1
        )
        x = torch.randn(1, 10, 18)
        c = torch.randn(1, 16)

        # Create a prediction that violates type rules
        prediction = {
            "sepsis_label": 1, "sepsis_risk": 0.9,
            "predicted_codes": [],
            "prediction_timeline": [{"event": "sepsis_onset", "hour": 12}],
        }
        patient = {
            "features": {"wbc": 7.0, "temperature": 37.0, "heart_rate": 75,
                        "platelet": 250, "map": 80, "creatinine": 0.8},
            "history": [], "meds": [], "allergies": [],
        }

        result = resolver.resolve(x, c, prediction, patient)
        assert result.violations_after <= result.violations_before
        assert len(result.edit_trace) > 0

    def test_resolver_preserves_neural_signal(self, model, validator):
        """Spectral resolver should have less drift than hard projection."""
        resolver = SpectralConflictResolver(
            model, validator, max_iterations=5, dampening_factor=0.1
        )
        hard = HardProjectionResolver(validator)
        x = torch.randn(1, 10, 18)
        c = torch.randn(1, 16)

        prediction = {
            "sepsis_label": 1, "sepsis_risk": 0.85,
            "predicted_codes": [],
            "prediction_timeline": [{"event": "sepsis_onset", "hour": 12}],
        }
        patient = {
            "features": {"wbc": 7.0, "temperature": 37.0, "heart_rate": 75,
                        "platelet": 250, "map": 80, "creatinine": 0.8},
            "history": [], "meds": [], "allergies": [],
        }

        spectral_result = resolver.resolve(x, c, prediction, patient)
        hard_result = hard.resolve(prediction, patient)

        # Hard projection forces risk to 0 (drift = 0.85)
        # Spectral should have less drift
        assert spectral_result.prediction_drift <= hard_result.prediction_drift

    def test_max_iterations_termination(self, model, validator):
        resolver = SpectralConflictResolver(
            model, validator, max_iterations=2, dampening_factor=0.5
        )
        x = torch.randn(1, 10, 18)
        c = torch.randn(1, 16)

        prediction = {
            "sepsis_label": 1, "sepsis_risk": 0.9,
            "predicted_codes": [],
            "prediction_timeline": [{"event": "sepsis_onset", "hour": 12}],
        }
        patient = {
            "features": {"wbc": 7.0, "temperature": 37.0},
            "history": [], "meds": [], "allergies": [],
        }

        result = resolver.resolve(x, c, prediction, patient)
        assert result.iterations_used <= 2


class TestHardProjection:
    def test_resolves_type_violation(self, validator):
        hard = HardProjectionResolver(validator)
        prediction = {
            "sepsis_label": 1, "sepsis_risk": 0.9,
            "predicted_codes": [], "prediction_timeline": [],
        }
        patient = {
            "features": {"wbc": 7.0, "temperature": 37.0, "heart_rate": 75,
                        "platelet": 250, "map": 80, "creatinine": 0.8},
            "history": [], "meds": [], "allergies": [],
        }
        result = hard.resolve(prediction, patient)
        assert result.violations_after < result.violations_before
        assert result.resolved_prediction["sepsis_risk"] == 0.0

    def test_drift_is_large(self, validator):
        """Hard projection should have large drift (forces to 0)."""
        hard = HardProjectionResolver(validator)
        prediction = {
            "sepsis_label": 1, "sepsis_risk": 0.9,
            "predicted_codes": [], "prediction_timeline": [],
        }
        patient = {
            "features": {"wbc": 7.0, "temperature": 37.0},
            "history": [], "meds": [], "allergies": [],
        }
        result = hard.resolve(prediction, patient)
        assert result.prediction_drift == pytest.approx(0.9, abs=0.01)

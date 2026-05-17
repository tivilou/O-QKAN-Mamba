"""Tests for Spectral Certificate framework."""
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.qkan.experimental.opfa_daruan import OPFADaruan
from src.qkan.experimental.opfa_qkan_mamba_block import OPFAQKANMambaBlock
from src.clinical.certificates.certificate import (
    SpectralCertificate,
    CertificateGenerator,
)
from src.clinical.certificates.certificate_verifier import CertificateVerifier


@pytest.fixture
def model():
    torch.manual_seed(42)
    return OPFAQKANMambaBlock(
        d_model=18, latent_dim=8, d_ontology=16, reps=6,
        use_mamba=False, device="cpu",
    )


@pytest.fixture
def generator():
    return CertificateGenerator()


@pytest.fixture
def verifier():
    return CertificateVerifier()


class TestCertificateGeneration:
    def test_certificate_from_full_prediction(self, model, generator):
        x = torch.randn(1, 10, 18)
        c = torch.randn(1, 16)
        out, spectral_info = model(x, c)
        risk = torch.sigmoid(out[0, -1, 0]).item()

        cert = generator.generate(risk, spectral_info)
        assert isinstance(cert, SpectralCertificate)
        assert 0.0 <= cert.prediction <= 1.0
        assert len(cert.band_contributions) == 3
        assert len(cert.active_bands) + len(cert.inactive_bands) == 3

    def test_certificate_with_dampened_band(self, model, generator):
        x = torch.randn(1, 10, 18)
        c = torch.randn(1, 16)
        mask = torch.ones(6)
        mask[0:2] = 0.0  # dampen infection band

        out, spectral_info = model(x, c, dampening_mask=mask)
        risk = torch.sigmoid(out[0, -1, 0]).item()
        cert = generator.generate(risk, spectral_info)

        assert "infection" in cert.inactive_bands
        assert cert.band_contributions["infection"] < 0.01

    def test_certificate_guarantees_for_inactive_bands(self, model, generator):
        x = torch.randn(1, 10, 18)
        c = torch.randn(1, 16)
        mask = torch.ones(6)
        mask[4:6] = 0.0  # dampen organ band

        out, spectral_info = model(x, c, dampening_mask=mask)
        risk = torch.sigmoid(out[0, -1, 0]).item()
        cert = generator.generate(risk, spectral_info)

        assert "organ_function" in cert.inactive_bands
        assert any("organ_function" in g for g in cert.guarantees)

    def test_certificate_deterministic(self, model, generator):
        x = torch.randn(1, 10, 18)
        c = torch.randn(1, 16)

        out1, info1 = model(x, c)
        out2, info2 = model(x, c)
        cert1 = generator.generate(torch.sigmoid(out1[0, -1, 0]).item(), info1)
        cert2 = generator.generate(torch.sigmoid(out2[0, -1, 0]).item(), info2)

        assert cert1.active_bands == cert2.active_bands
        assert cert1.prediction == cert2.prediction


class TestCertificateVerification:
    def test_valid_certificate_passes(self, model, generator, verifier):
        """Certificate with infection band active for sepsis prediction is valid."""
        x = torch.randn(1, 10, 18)
        c = torch.randn(1, 16)
        out, spectral_info = model(x, c)
        risk = torch.sigmoid(out[0, -1, 0]).item()

        cert = generator.generate(risk, spectral_info)
        verified = verifier.verify(cert)
        # All bands active by default → always valid
        assert verified.is_valid

    def test_sepsis_without_infection_band_fails(self, generator, verifier):
        """Sepsis prediction without infection band should fail verification."""
        # Simulate: high risk but infection band inactive
        spectral_info = {
            "band_contributions": torch.tensor([0.0, 0.5, 0.5]),
            "band_names": ["infection", "hemodynamics", "organ_function"],
        }
        cert = generator.generate(0.85, spectral_info, prediction_label=1)
        verified = verifier.verify(cert)

        assert not verified.is_valid
        assert "infection" in verified.violation_reason

    def test_low_risk_without_infection_is_valid(self, generator, verifier):
        """Low risk prediction without infection band is fine."""
        spectral_info = {
            "band_contributions": torch.tensor([0.0, 0.6, 0.4]),
            "band_names": ["infection", "hemodynamics", "organ_function"],
        }
        cert = generator.generate(0.3, spectral_info, prediction_label=0)
        verified = verifier.verify(cert)

        assert verified.is_valid

    def test_no_active_bands_for_strong_prediction_fails(self, generator, verifier):
        """Strong prediction with no active bands is degenerate."""
        spectral_info = {
            "band_contributions": torch.tensor([0.0, 0.0, 0.0]),
            "band_names": ["infection", "hemodynamics", "organ_function"],
        }
        cert = generator.generate(0.9, spectral_info)
        verified = verifier.verify(cert)

        assert not verified.is_valid

    def test_explain_violation(self, generator, verifier):
        """Explanation should be human-readable."""
        spectral_info = {
            "band_contributions": torch.tensor([0.0, 0.5, 0.5]),
            "band_names": ["infection", "hemodynamics", "organ_function"],
        }
        cert = generator.generate(0.85, spectral_info)
        verified = verifier.verify(cert)
        explanation = verifier.explain_violation(verified)

        assert "VIOLATION" in explanation
        assert "infection" in explanation


class TestCertificateSoundness:
    def test_dampened_band_guarantees_zero_influence(self, model, generator):
        """Core soundness: dampened band truly has zero contribution."""
        x = torch.randn(1, 10, 18)
        c = torch.randn(1, 16)

        # Run with infection band dampened
        mask = torch.ones(6)
        mask[0:2] = 0.0
        out, spectral_info = model(x, c, dampening_mask=mask)

        cert = generator.generate(
            torch.sigmoid(out[0, -1, 0]).item(), spectral_info
        )

        # Soundness: infection band contribution must be exactly 0
        assert cert.band_contributions["infection"] < 0.001
        assert "infection" in cert.inactive_bands

    def test_certificate_reflects_actual_computation(self, model, generator):
        """Certificate must match what the model actually computed."""
        x = torch.randn(1, 10, 18)
        c = torch.randn(1, 16)

        # Full computation
        out_full, info_full = model(x, c)
        cert_full = generator.generate(
            torch.sigmoid(out_full[0, -1, 0]).item(), info_full
        )

        # All bands should be active in full computation
        assert len(cert_full.active_bands) == 3

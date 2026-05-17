"""Tests for Layer Extension and Deliberation Controller."""
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.qkan.experimental.opfa_daruan import OPFADaruan
from src.qkan.experimental.layer_extension import DynamicLayerExtender
from src.clinical.certificates.certificate import CertificateGenerator
from src.clinical.certificates.certificate_verifier import CertificateVerifier
from src.clinical.agent.deliberation_controller import DeliberationController


@pytest.fixture
def model():
    torch.manual_seed(42)
    return OPFADaruan(dim=8, reps=6, d_ontology=16, device="cpu")


@pytest.fixture
def controller():
    return DeliberationController()


class TestDynamicLayerExtension:
    def test_extension_increases_reps(self, model):
        extended = DynamicLayerExtender.extend_band(model, "infection", n_extra_layers=2)
        assert extended.reps == 8
        assert model.reps == 6

    def test_extension_targets_correct_band(self, model):
        extended = DynamicLayerExtender.extend_band(model, "infection", n_extra_layers=2)
        inf_band = extended.allocator.get_band_by_name("infection")
        assert len(inf_band.layer_indices) == 4  # was 2, now 4

    def test_other_bands_unchanged(self, model):
        extended = DynamicLayerExtender.extend_band(model, "infection", n_extra_layers=2)
        hemo = extended.allocator.get_band_by_name("hemodynamics")
        organ = extended.allocator.get_band_by_name("organ_function")
        assert len(hemo.layer_indices) == 2
        assert len(organ.layer_indices) == 2

    def test_extended_model_runs(self, model):
        extended = DynamicLayerExtender.extend_band(model, "hemodynamics", n_extra_layers=1)
        x = torch.randn(2, 8)
        c = torch.randn(2, 16)
        out, info = extended(x, c)
        assert out.shape == (2, 8)
        assert len(info["band_names"]) == 3

    def test_more_layers_more_frequencies(self, model):
        """Extended band should have higher frequency capacity."""
        x = torch.randn(4, 8)
        c = torch.randn(4, 16)

        # Base model: infection band has 2 layers
        mask_inf = torch.zeros(6)
        mask_inf[0:2] = 1.0
        out_base, _ = model(x, c, dampening_mask=mask_inf)

        # Extended: infection band has 4 layers
        extended = DynamicLayerExtender.extend_band(model, "infection", n_extra_layers=2)
        mask_ext = torch.zeros(8)
        mask_ext[0:4] = 1.0
        out_ext, _ = extended(x, c, dampening_mask=mask_ext)

        # Extended should produce different (richer) output
        assert (out_base - out_ext).abs().mean() > 0.01


class TestDeliberationController:
    def test_no_deliberation_when_valid(self, model, controller):
        x = torch.randn(1, 8)
        c = torch.randn(1, 16)
        _, info = model(x, c)
        gen = CertificateGenerator()
        cert = gen.generate(0.3, info)

        assert not controller.should_deliberate(cert)

    def test_deliberation_triggered_on_violation(self, controller):
        gen = CertificateGenerator()
        info = {
            "band_contributions": torch.tensor([0.0, 0.5, 0.5]),
            "band_names": ["infection", "hemodynamics", "organ_function"],
        }
        cert = gen.generate(0.85, info)

        assert controller.should_deliberate(cert)

    def test_targets_correct_band(self, controller):
        gen = CertificateGenerator()
        info = {
            "band_contributions": torch.tensor([0.0, 0.5, 0.5]),
            "band_names": ["infection", "hemodynamics", "organ_function"],
        }
        cert = gen.generate(0.85, info)
        CertificateVerifier().verify(cert)

        target = controller.identify_target_band(cert)
        assert target == "infection"

    def test_deliberation_plan_structure(self, controller):
        gen = CertificateGenerator()
        info = {
            "band_contributions": torch.tensor([0.0, 0.5, 0.5]),
            "band_names": ["infection", "hemodynamics", "organ_function"],
        }
        cert = gen.generate(0.85, info)

        plan = controller.deliberation_plan(cert)
        assert plan["action"] == "extend"
        assert plan["target_band"] == "infection"
        assert plan["n_layers"] >= 1
        assert "reason" in plan

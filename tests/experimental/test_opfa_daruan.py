"""Tests for OPFA-DARUAN: Ontology-Partitioned Frequency-Allocated DARUAN."""
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.qkan.experimental.opfa_daruan import (
    OPFADaruan,
    FrequencyBandAllocator,
    AdaptiveMeasurement,
)


@pytest.fixture
def model():
    torch.manual_seed(42)
    return OPFADaruan(dim=8, reps=6, d_ontology=16, device="cpu")


class TestFrequencyBandAllocator:
    def test_default_partition(self):
        alloc = FrequencyBandAllocator(reps=6)
        assert alloc.n_bands == 3
        assert alloc.total_layers == 6
        assert alloc.bands[0].name == "infection"
        assert alloc.bands[0].encoding_axis == "z"
        assert alloc.bands[1].encoding_axis == "x"
        assert alloc.bands[2].encoding_axis == "y"

    def test_custom_partition(self):
        config = [
            {"name": "vitals", "axis": "z", "n_layers": 3},
            {"name": "labs", "axis": "x", "n_layers": 2},
        ]
        alloc = FrequencyBandAllocator(reps=5, band_config=config)
        assert alloc.n_bands == 2
        assert alloc.bands[0].layer_indices == [0, 1, 2]
        assert alloc.bands[1].layer_indices == [3, 4]

    def test_layer_to_band_mapping(self):
        alloc = FrequencyBandAllocator(reps=6)
        assert alloc.get_band_for_layer(0).name == "infection"
        assert alloc.get_band_for_layer(2).name == "hemodynamics"
        assert alloc.get_band_for_layer(5).name == "organ_function"


class TestMultiAxisEncoding:
    def test_different_axes_produce_different_outputs(self, model):
        """Bands with different encoding axes should produce distinct patterns."""
        x = torch.randn(4, 8)
        c = torch.randn(4, 16)

        # Only infection band active (σ_z encoding)
        mask_inf = torch.zeros(6)
        mask_inf[0:2] = 1.0
        out_inf, _ = model(x, c, dampening_mask=mask_inf)

        # Only hemodynamics band active (σ_x encoding)
        mask_hemo = torch.zeros(6)
        mask_hemo[2:4] = 1.0
        out_hemo, _ = model(x, c, dampening_mask=mask_hemo)

        # Outputs should differ (different encoding axes)
        diff = (out_inf - out_hemo).abs().mean()
        assert diff > 0.001, f"Different axes should produce different outputs, got diff={diff}"


class TestFrequencyBandIsolation:
    def test_zeroing_band_removes_contribution(self, model):
        """Zeroing a band should set its contribution to 0."""
        x = torch.randn(4, 8)
        c = torch.randn(4, 16)

        mask = torch.ones(6)
        mask[0:2] = 0.0  # zero infection band
        _, info = model(x, c, dampening_mask=mask)

        assert info["band_contributions"][0].item() < 0.001

    def test_single_band_isolation(self, model):
        """Activating only one band should give 100% contribution to that band."""
        x = torch.randn(4, 8)
        c = torch.randn(4, 16)

        mask = torch.zeros(6)
        mask[2:4] = 1.0  # only hemodynamics
        _, info = model(x, c, dampening_mask=mask)

        assert info["band_contributions"][1].item() > 0.99


class TestAdaptiveMeasurement:
    def test_default_is_sigma_z(self):
        """With zero concept embedding, should default to σ_z measurement."""
        m = AdaptiveMeasurement(d_ontology=16)
        c_zero = torch.zeros(2, 16)
        weights = torch.softmax(m.basis_proj(c_zero), dim=-1)
        # Bias initialized to [0, 0, 1] → softmax → mostly σ_z
        assert weights[0, 2] > weights[0, 0]
        assert weights[0, 2] > weights[0, 1]

    def test_different_contexts_different_measurements(self, model):
        """Different concept embeddings should produce different measurement weights."""
        x = torch.randn(2, 8)
        c1 = torch.randn(2, 16) * 2.0
        c2 = torch.randn(2, 16) * 2.0
        out1, _ = model(x, c1)
        out2, _ = model(x, c2)
        assert (out1 - out2).abs().mean() > 0.001


class TestGradientFlow:
    def test_all_parameters_receive_gradients(self, model):
        x = torch.randn(2, 8)
        c = torch.randn(2, 16)
        out, _ = model(x, c)
        out.sum().backward()

        assert model.theta.grad is not None
        assert model.w_base.grad is not None
        assert model.W_KG.grad is not None
        assert model.adaptive_measure.basis_proj.weight.grad is not None

    def test_gradient_magnitude_reasonable(self, model):
        x = torch.randn(2, 8)
        c = torch.randn(2, 16)
        out, _ = model(x, c)
        out.sum().backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(p.grad).any(), f"Inf gradient in {name}"


class TestExpressivity:
    def test_expressivity_vs_original(self):
        """OPFA-DARUAN should not lose more than 3x expressivity vs original."""
        from src.qkan.experimental.ontology_modulated_daruan import OntologyModulatedDARUAN
        import math

        torch.manual_seed(0)
        dim = 1
        n_epochs = 200
        lr = 0.01

        # Target function: sin(20x)/(20x)
        x_train = torch.linspace(0.01, 1.0, 200).unsqueeze(1)
        y_train = torch.sin(20 * x_train) / (20 * x_train)
        c = torch.zeros(200, 16)

        # Original DARUAN
        orig = OntologyModulatedDARUAN(dim=dim, reps=6, d_ontology=16, device="cpu")
        opt_orig = torch.optim.Adam(orig.parameters(), lr=lr)
        for _ in range(n_epochs):
            loss = ((orig(x_train, c) - y_train) ** 2).mean()
            opt_orig.zero_grad()
            loss.backward()
            opt_orig.step()

        # OPFA-DARUAN
        opfa = OPFADaruan(dim=dim, reps=6, d_ontology=16, device="cpu")
        opt_opfa = torch.optim.Adam(opfa.parameters(), lr=lr)
        for _ in range(n_epochs):
            out, _ = opfa(x_train, c)
            loss = ((out - y_train) ** 2).mean()
            opt_opfa.zero_grad()
            loss.backward()
            opt_opfa.step()

        with torch.no_grad():
            err_orig = ((orig(x_train, c) - y_train) ** 2).mean().item()
            out_opfa, _ = opfa(x_train, c)
            err_opfa = ((out_opfa - y_train) ** 2).mean().item()

        ratio = err_opfa / max(err_orig, 1e-8)
        assert ratio < 3.0, f"OPFA expressivity loss too large: {ratio:.2f}x"

    def test_multi_axis_orthogonality(self):
        """Band outputs should be near-orthogonal (mean |cos| < 0.5)."""
        torch.manual_seed(42)
        model = OPFADaruan(dim=8, reps=6, d_ontology=16, device="cpu")
        x = torch.randn(100, 8)
        c = torch.randn(100, 16)

        band_outputs = []
        for band in model.allocator.bands:
            mask = torch.zeros(6)
            for li in band.layer_indices:
                mask[li] = 1.0
            with torch.no_grad():
                out, _ = model(x, c, dampening_mask=mask)
            band_outputs.append(out.flatten())

        # Compute pairwise cosine similarities
        from torch.nn.functional import cosine_similarity
        cos_vals = []
        for i in range(len(band_outputs)):
            for j in range(i + 1, len(band_outputs)):
                cos = cosine_similarity(
                    band_outputs[i].unsqueeze(0),
                    band_outputs[j].unsqueeze(0),
                ).item()
                cos_vals.append(abs(cos))

        mean_cos = sum(cos_vals) / len(cos_vals)
        assert mean_cos < 0.5, f"Bands not orthogonal enough: mean |cos|={mean_cos:.3f}"

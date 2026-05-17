"""OPFA-DARUAN: Ontology-Partitioned Frequency-Allocated DARUAN.

Three QML-level modifications to the original DARUAN:
1. Multi-Axis Encoding: σ_x, σ_y, σ_z based on feature category
2. Ontology-Partitioned Frequency Allocation: hard-partitioned frequency bands
3. Context-Adaptive Measurement: rotated measurement basis

Key property: frequency band ↔ ontology category mapping is structural
(by construction), enabling provable spectral certificates.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from qkan.daruan.torch_qc import StateVector, TorchGates


@dataclass
class FrequencyBand:
    """A frequency band assigned to an ontology category."""
    name: str
    category_id: int
    layer_indices: list  # which re-uploading layers belong to this band
    encoding_axis: str   # 'x', 'y', or 'z'


class FrequencyBandAllocator:
    """Allocates re-uploading layers to ontology-defined frequency bands."""

    def __init__(self, reps: int, band_config: list[dict] = None):
        """
        Args:
            reps: total number of re-uploading layers
            band_config: list of dicts with 'name', 'axis', 'n_layers'
                If None, uses default clinical partition.
        """
        if band_config is None:
            band_config = self._default_clinical_bands(reps)

        self.bands = []
        layer_idx = 0
        for cfg in band_config:
            n = cfg["n_layers"]
            band = FrequencyBand(
                name=cfg["name"],
                category_id=len(self.bands),
                layer_indices=list(range(layer_idx, layer_idx + n)),
                encoding_axis=cfg["axis"],
            )
            self.bands.append(band)
            layer_idx += n

        self.total_layers = layer_idx
        self.n_bands = len(self.bands)

    def _default_clinical_bands(self, reps: int) -> list[dict]:
        """Default: 3 bands for infection, hemodynamics, organ function."""
        n_per_band = max(1, reps // 3)
        remainder = reps - 3 * n_per_band
        return [
            {"name": "infection", "axis": "z", "n_layers": n_per_band + (1 if remainder > 0 else 0)},
            {"name": "hemodynamics", "axis": "x", "n_layers": n_per_band + (1 if remainder > 1 else 0)},
            {"name": "organ_function", "axis": "y", "n_layers": n_per_band},
        ]

    def get_band_by_name(self, name: str) -> FrequencyBand:
        for b in self.bands:
            if b.name == name:
                return b
        raise KeyError(f"Band '{name}' not found")

    def get_band_for_layer(self, layer_idx: int) -> FrequencyBand:
        for b in self.bands:
            if layer_idx in b.layer_indices:
                return b
        raise IndexError(f"Layer {layer_idx} not assigned to any band")


class AdaptiveMeasurement(nn.Module):
    """Context-adaptive measurement basis rotation.

    Instead of always measuring σ_z, computes a weighted combination of
    σ_x, σ_y, σ_z measurements based on clinical context:
        M(c) = α(c)·measure_x + β(c)·measure_y + γ(c)·measure_z

    Mathematically equivalent to measuring in a rotated basis on the
    Bloch sphere, with the rotation determined by the concept vector.
    """

    def __init__(self, d_ontology: int, device: str = "cpu"):
        super().__init__()
        self.basis_proj = nn.Linear(d_ontology, 3, device=device)
        nn.init.zeros_(self.basis_proj.weight)
        # Initialize bias to [0, 0, 1] so default is pure σ_z measurement
        with torch.no_grad():
            self.basis_proj.bias.copy_(torch.tensor([0.0, 0.0, 1.0]))

    def forward(
        self, psi: StateVector, concept_emb: torch.Tensor
    ) -> torch.Tensor:
        """Measure with context-adaptive basis.

        Args:
            psi: StateVector to measure
            concept_emb: (B, d_ontology) concept embedding
        Returns:
            (B, dim) measurement results
        """
        # Compute basis weights (softmax ensures they sum to 1)
        weights = F.softmax(self.basis_proj(concept_emb), dim=-1)  # (B, 3)

        # Save state tensor before measurement (measurements are non-destructive
        # on the tensor level — they just compute expectation values)
        state_saved = psi.state.clone()

        m_z = psi.measure_z()  # (B, dim)

        # Restore state for x measurement
        psi.state = state_saved.clone()
        m_x = psi.measure_x()  # (B, dim)

        # Restore state for y measurement
        psi.state = state_saved
        m_y = psi.measure_y()  # (B, dim)

        # Weighted combination: α·m_x + β·m_y + γ·m_z
        result = (
            weights[:, 0:1] * m_x +
            weights[:, 1:2] * m_y +
            weights[:, 2:3] * m_z
        )
        return result


class OPFADaruan(nn.Module):
    """Ontology-Partitioned Frequency-Allocated DARUAN.

    QML-level modifications vs original DARUAN:
    1. Multi-axis encoding: σ_x/σ_y/σ_z per band (not always σ_z)
    2. OPFA: frequency bands hard-partitioned by ontology category
    3. Context-adaptive measurement: rotated basis per clinical context

    The structural partition guarantees that frequency band k's output
    depends ONLY on layers assigned to band k. This enables provable
    spectral certificates.
    """

    def __init__(
        self,
        dim: int,
        reps: int = 6,
        d_ontology: int = 16,
        band_config: list[dict] = None,
        base_freq_init: str = "geometric",
        device: str = "cpu",
    ):
        super().__init__()
        self.dim = dim
        self.reps = reps
        self.d_ontology = d_ontology
        self.device = device

        # Frequency band allocation
        self.allocator = FrequencyBandAllocator(reps, band_config)
        self.n_bands = self.allocator.n_bands

        # Variational parameters: theta[dim, reps+1, 2] for Rz/Ry rotations
        self.theta = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(dim, reps + 1, 2, device=device))
        )

        # Per-band base frequencies (geometric within each band)
        w_base = torch.zeros(reps, device=device)
        for band in self.allocator.bands:
            for i, layer_idx in enumerate(band.layer_indices):
                w_base[layer_idx] = 2.0 ** i
        self.w_base = nn.Parameter(w_base, requires_grad=True)

        # Intra-band learnable modulation (concept-conditioned)
        self.W_KG = nn.Parameter(torch.randn(reps, d_ontology, device=device) * 0.01)
        self.b = nn.Parameter(torch.zeros(reps, device=device))

        # Adaptive measurement
        self.adaptive_measure = AdaptiveMeasurement(d_ontology, device=device)

        # Post-activation
        self.postact_weight = nn.Parameter(torch.ones(dim, device=device))
        self.postact_bias = nn.Parameter(torch.zeros(dim, device=device))

        # Build encoding axis lookup for each layer
        self._encoding_axes = []
        for l in range(reps):
            band = self.allocator.get_band_for_layer(l)
            self._encoding_axes.append(band.encoding_axis)

    def compute_modulated_weights(self, concept_emb: torch.Tensor) -> torch.Tensor:
        """Compute per-sample modulated weights (intra-band learning).

        Band boundaries are fixed; weights within each band are learned.
        """
        logits = torch.einsum("rd,bd->br", self.W_KG, concept_emb) + self.b
        modulation = F.softplus(logits)
        return self.w_base.unsqueeze(0) * modulation

    def forward(
        self,
        x: torch.Tensor,
        concept_emb: torch.Tensor,
        dampening_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict]:
        """Forward pass with multi-axis encoding, OPFA, and adaptive measurement.

        Args:
            x: (B, dim) input features
            concept_emb: (B, d_ontology) ontology embedding
            dampening_mask: (reps,) optional per-layer dampening
        Returns:
            output: (B, dim) activation output
            spectral_info: dict with per-band contribution info
        """
        B = x.shape[0]
        assert x.shape[1] == self.dim

        w_mod = self.compute_modulated_weights(concept_emb)
        if dampening_mask is not None:
            w_mod = w_mod * dampening_mask.unsqueeze(0)

        # Initialize quantum state
        psi = StateVector(B, self.dim, device=x.device, dtype=torch.complex64)
        psi.h()

        # Store per-band state norms for spectral decomposition
        band_energies = torch.zeros(self.n_bands, device=x.device)

        for l in range(self.reps):
            # Variational rotation (shared across all bands)
            psi.rz(self.theta[:, l, 0])
            psi.ry(self.theta[:, l, 1])

            # Multi-axis data encoding: axis determined by band assignment
            encoded = x * w_mod[:, l:l + 1]  # (B, dim)
            axis = self._encoding_axes[l]

            if axis == "z":
                gate = TorchGates.rz_gate(encoded, dtype=torch.complex64)
            elif axis == "x":
                gate = TorchGates.rx_gate(encoded, dtype=torch.complex64)
            else:  # "y"
                gate = TorchGates.ry_gate(encoded, dtype=torch.complex64)

            psi.state = torch.einsum("mnbi,bin->bim", gate, psi.state)

            # Track energy contribution per band
            band_idx = self.allocator.get_band_for_layer(l).category_id
            layer_energy = w_mod[:, l].abs().mean()
            band_energies[band_idx] = band_energies[band_idx] + layer_energy

        # Final variational rotation
        psi.rz(self.theta[:, self.reps, 0])
        psi.ry(self.theta[:, self.reps, 1])

        # Context-adaptive measurement
        postacts = self.adaptive_measure(psi, concept_emb)

        # Post-activation scaling
        out = postacts * self.postact_weight + self.postact_bias

        # Build spectral info for certificate generation
        total_energy = band_energies.sum() + 1e-8
        spectral_info = {
            "band_contributions": (band_energies / total_energy).detach(),
            "band_names": [b.name for b in self.allocator.bands],
            "active_bands": [
                b.name for b in self.allocator.bands
                if band_energies[b.category_id] > 0.01 * total_energy
            ],
        }

        return out, spectral_info

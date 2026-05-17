"""OPFA-QKAN-Mamba Block: OPFA-DARUAN gated Mamba with spectral certificates.

Replaces the original OQKANMambaBlock with genuine QML-level modifications:
- Multi-axis encoding (σ_x/σ_y/σ_z per ontology category)
- Ontology-partitioned frequency allocation (structural band separation)
- Context-adaptive measurement (rotated Bloch sphere measurement)

Architecture: y = x + sigmoid(OPFA_QKAN(x, c)) * Mamba(x)
"""
import torch
import torch.nn as nn

from .opfa_daruan import OPFADaruan
from .o_qkan_mamba_block import CausalConv1dFallback

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class OPFAQKANGate(nn.Module):
    """Bottleneck gate using OPFA-DARUAN: down_proj → OPFA-DARUAN → up_proj."""

    def __init__(
        self,
        d_model: int,
        latent_dim: int,
        d_ontology: int = 16,
        reps: int = 6,
        band_config: list[dict] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.down_proj = nn.Linear(d_model, latent_dim, device=device)
        self.daruan = OPFADaruan(
            dim=latent_dim, reps=reps, d_ontology=d_ontology,
            band_config=band_config, device=device,
        )
        self.up_proj = nn.Linear(latent_dim, d_model, device=device)

    def forward(self, x: torch.Tensor, concept_emb: torch.Tensor,
                dampening_mask: torch.Tensor = None):
        """
        Args:
            x: (B*L, d_model) flattened sequence features
            concept_emb: (B*L, d_ontology) concept embeddings
            dampening_mask: (reps,) optional frequency dampening
        Returns:
            gate_logits: (B*L, d_model)
            spectral_info: dict with band contributions
        """
        h = self.down_proj(x)
        activated, spectral_info = self.daruan(h, concept_emb, dampening_mask)
        return self.up_proj(activated), spectral_info


class OPFAQKANMambaBlock(nn.Module):
    """OPFA-QKAN gated Mamba block with spectral certificate support.

    Architecture: y = x + sigmoid(OPFA_QKAN(x, c)) * Mamba(x)

    Key differences from OQKANMambaBlock:
    - Gate uses OPFA-DARUAN (multi-axis, partitioned bands, adaptive measurement)
    - Forward returns spectral_info for certificate generation
    - Supports per-band dampening for targeted conflict resolution
    """

    def __init__(
        self,
        d_model: int,
        latent_dim: int = 8,
        d_ontology: int = 16,
        reps: int = 6,
        band_config: list[dict] = None,
        use_mamba: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model, device=device)

        self.gate = OPFAQKANGate(
            d_model=d_model, latent_dim=latent_dim,
            d_ontology=d_ontology, reps=reps,
            band_config=band_config, device=device,
        )

        if use_mamba and MAMBA_AVAILABLE:
            self.mixer = Mamba(d_model=d_model, device=device)
        else:
            self.mixer = CausalConv1dFallback(d_model).to(device)

    def forward(
        self,
        x: torch.Tensor,
        concept_emb: torch.Tensor,
        dampening_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x: (B, L, d_model) input sequence
            concept_emb: (B, d_ontology) concept embedding
            dampening_mask: (reps,) optional per-layer dampening
        Returns:
            output: (B, L, d_model)
            spectral_info: dict with band contribution details
        """
        h = self.norm(x)
        B, L, D = h.shape

        # Expand concept_emb to match sequence length
        c_expanded = concept_emb.unsqueeze(1).expand(B, L, -1).reshape(B * L, -1)
        h_flat = h.reshape(B * L, D)

        # OPFA-QKAN gate (returns spectral info)
        gate_logits, spectral_info = self.gate(h_flat, c_expanded, dampening_mask)
        gate_val = torch.sigmoid(gate_logits).reshape(B, L, D)

        # Mamba mixer for temporal modeling
        mixed = self.mixer(h)

        # Gated residual
        output = x + gate_val * mixed

        return output, spectral_info

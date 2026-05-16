"""O-QKAN-Mamba Block: Mamba mixer + Ontology-Modulated QKAN gate.

This is the full ontology-aware block that combines:
- Mamba SSM for sequence modeling
- OntologyModulatedQKAN for context-dependent gating
"""
import torch
import torch.nn as nn

from .ontology_modulated_qkan import OntologyModulatedQKAN

try:
    from mamba_ssm import Mamba
    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False


class CausalConv1dFallback(nn.Module):
    """Simple causal conv1d fallback when mamba-ssm is not available."""

    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size, padding=kernel_size - 1, groups=d_model
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = self.conv(h)[:, :, :x.shape[1]]
        return h.transpose(1, 2)


class OQKANMambaBlock(nn.Module):
    """Mamba block with Ontology-Modulated QKAN gate.

    Output: y = x + gate(x, concept_emb) * mixer(x)
    Gate: sigmoid(OntologyModulatedQKAN(x, concept_emb))
    """

    def __init__(
        self,
        d_model: int,
        latent_dim: int = 16,
        d_ontology: int = 64,
        reps: int = 3,
        use_mamba: bool = True,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        device: str = "cpu",
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, device=device)

        self.gate = OntologyModulatedQKAN(
            d_model=d_model,
            latent_dim=latent_dim,
            d_ontology=d_ontology,
            reps=reps,
            device=device,
        )

        if use_mamba and _MAMBA_AVAILABLE:
            self.mixer = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            ).to(device)
            self.use_mamba = True
        else:
            self.mixer = CausalConv1dFallback(d_model).to(device)
            self.use_mamba = False

    def forward(self, x: torch.Tensor, concept_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
            concept_emb: (B, d_ontology) or (B, L, d_ontology)
        Returns:
            (B, L, d_model)
        """
        h = self.norm(x)
        gate_val = torch.sigmoid(self.gate(h, concept_emb))
        return x + gate_val * self.mixer(h)

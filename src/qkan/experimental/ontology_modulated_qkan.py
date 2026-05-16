"""Ontology-Modulated QKAN: QKAN with context-dependent activation spectrum.

Uses OntologyModulatedDARUAN as the activation function within a
bottleneck architecture (Linear → OntologyModulatedDARUAN → Linear).
"""
import torch
import torch.nn as nn

from .ontology_modulated_daruan import OntologyModulatedDARUAN


class OntologyModulatedQKAN(nn.Module):
    """QKAN-style module with ontology-modulated re-uploading frequencies.

    Bottleneck pattern: down_proj → OntologyModulatedDARUAN → up_proj
    """

    def __init__(
        self,
        d_model: int,
        latent_dim: int = 16,
        d_ontology: int = 64,
        reps: int = 3,
        base_freq_init: str = "geometric",
        device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim

        self.down_proj = nn.Linear(d_model, latent_dim, device=device)
        self.daruan = OntologyModulatedDARUAN(
            dim=latent_dim,
            reps=reps,
            d_ontology=d_ontology,
            base_freq_init=base_freq_init,
            device=device,
        )
        self.up_proj = nn.Linear(latent_dim, d_model, device=device)

    def forward(self, x: torch.Tensor, concept_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) or (B, d_model)
            concept_emb: (B, d_ontology) or (B, L, d_ontology)
        Returns:
            same shape as x
        """
        has_seq = x.dim() == 3
        if has_seq:
            B, L, D = x.shape
            x_flat = x.reshape(B * L, D)
            if concept_emb.dim() == 3:
                c_flat = concept_emb.reshape(B * L, -1)
            else:
                c_flat = concept_emb.unsqueeze(1).expand(B, L, -1).reshape(B * L, -1)
        else:
            x_flat = x
            c_flat = concept_emb

        h = self.down_proj(x_flat)
        h = self.daruan(h, c_flat)
        h = self.up_proj(h)

        if has_seq:
            return h.reshape(B, L, D)
        return h

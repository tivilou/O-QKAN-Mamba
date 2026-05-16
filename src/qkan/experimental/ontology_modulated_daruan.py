"""Ontology-Modulated DARUAN: context-dependent re-uploading frequencies.

Core contribution: w_ℓ(t, x_t, c_t) = w_ℓ^base · softplus(W_KG · φ(c_t) + b_ℓ)

This makes the activation spectrum dependent on clinical context without
modifying any existing QKAN public APIs. Uses StateVector/TorchGates
directly to implement per-sample modulated data re-uploading.

Reference: RESEARCH_PROPOSAL.md Section 2.1
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from qkan.daruan.torch_qc import StateVector, TorchGates


class OntologyModulatedDARUAN(nn.Module):
    """DARUAN with knowledge-conditioned re-uploading frequencies.

    Instead of static preacts_weight shared across the batch, computes
    per-sample modulation based on ontology embeddings.
    """

    def __init__(
        self,
        dim: int,
        reps: int = 3,
        d_ontology: int = 64,
        base_freq_init: str = "geometric",
        device: str = "cpu",
    ):
        super().__init__()
        self.dim = dim
        self.reps = reps
        self.d_ontology = d_ontology
        self.device = device

        # Variational parameters for the quantum circuit
        self.theta = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(dim, reps + 1, 2, device=device))
        )

        # Base frequencies: geometric series 2^(ℓ-1)
        if base_freq_init == "geometric":
            base = torch.tensor(
                [2.0 ** l for l in range(reps)], device=device
            )
        else:
            base = torch.ones(reps, device=device)
        self.w_base = nn.Parameter(base, requires_grad=True)

        # Ontology modulation parameters
        self.W_KG = nn.Parameter(
            torch.randn(reps, d_ontology, device=device) * 0.01
        )
        self.b = nn.Parameter(torch.zeros(reps, device=device))

        # Post-activation scaling
        self.postact_weight = nn.Parameter(torch.ones(dim, device=device))
        self.postact_bias = nn.Parameter(torch.zeros(dim, device=device))

    def compute_modulated_weights(self, concept_emb: torch.Tensor) -> torch.Tensor:
        """Compute per-sample modulated re-uploading weights.

        Args:
            concept_emb: (B, d_ontology) ontology embedding
        Returns:
            (B, reps) modulated weights
        """
        # W_KG: [reps, d_ontology], concept_emb: [B, d_ontology]
        logits = torch.einsum("rd,bd->br", self.W_KG, concept_emb) + self.b
        modulation = F.softplus(logits)  # [B, reps], always positive
        return self.w_base.unsqueeze(0) * modulation  # [B, reps]

    def forward(self, x: torch.Tensor, concept_emb: torch.Tensor, dampening_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with per-sample modulated re-uploading.

        Args:
            x: (B, dim) input features
            concept_emb: (B, d_ontology) ontology embedding
            dampening_mask: (reps,) optional mask to dampen specific frequencies.
                           Values in [0, 1]. Default: all ones (no dampening).
        Returns:
            (B, dim) modulated activation output
        """
        B = x.shape[0]
        assert x.shape[1] == self.dim

        # Compute per-sample modulated weights: [B, reps]
        w_mod = self.compute_modulated_weights(concept_emb)

        # Apply dampening mask if provided
        if dampening_mask is not None:
            w_mod = w_mod * dampening_mask.unsqueeze(0)

        # Data re-uploading circuit with per-sample frequencies
        # Using StateVector/TorchGates directly for per-sample control
        psi = StateVector(B, self.dim, device=x.device, dtype=torch.complex64)
        psi.h()

        for l in range(self.reps):
            # Variational rotation
            psi.rz(self.theta[:, l, 0])
            psi.ry(self.theta[:, l, 1])
            # Data re-uploading with per-sample modulated frequency
            # encoded_x[b, i] = w_mod[b, l] * x[b, i]
            encoded = x * w_mod[:, l:l+1]  # [B, dim]
            gate = TorchGates.rz_gate(encoded, dtype=torch.complex64)
            psi.state = torch.einsum("mnbi,bin->bim", gate, psi.state)

        # Final variational rotation
        psi.rz(self.theta[:, self.reps, 0])
        psi.ry(self.theta[:, self.reps, 1])

        # Measurement
        postacts = psi.measure_z()  # [B, dim]

        # Post-activation scaling
        out = postacts * self.postact_weight + self.postact_bias
        return out

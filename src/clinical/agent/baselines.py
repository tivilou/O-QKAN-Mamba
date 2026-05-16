"""Baseline architectures for ablation comparison."""
import torch
import torch.nn as nn

from src.qkan.experimental.o_qkan_mamba_block import OQKANMambaBlock, CausalConv1dFallback


class MambaOnlyBlock(nn.Module):
    """Mamba without any gate (direct residual)."""

    def __init__(self, d_model: int, device: str = "cpu"):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, device=device)
        self.mixer = CausalConv1dFallback(d_model).to(device)

    def forward(self, x, concept_emb=None, **kwargs):
        h = self.norm(x)
        return x + self.mixer(h)


class MambaMLPGateBlock(nn.Module):
    """Mamba with simple MLP gate (no quantum, no ontology)."""

    def __init__(self, d_model: int, device: str = "cpu"):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, device=device)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2, device=device),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model, device=device),
            nn.Sigmoid(),
        )
        self.mixer = CausalConv1dFallback(d_model).to(device)

    def forward(self, x, concept_emb=None, **kwargs):
        h = self.norm(x)
        B, L, D = h.shape
        gate_val = self.gate(h.reshape(B * L, D)).reshape(B, L, D)
        return x + gate_val * self.mixer(h)


class MambaQKANNoKGBlock(nn.Module):
    """Mamba + QKAN gate but with zero concept embedding (no KG)."""

    def __init__(self, d_model: int, d_ontology: int = 16, device: str = "cpu"):
        super().__init__()
        self.block = OQKANMambaBlock(
            d_model=d_model, latent_dim=8, d_ontology=d_ontology,
            reps=3, use_mamba=False, device=device,
        )

    def forward(self, x, concept_emb=None, **kwargs):
        c = torch.zeros(x.shape[0], self.block.gate.daruan.d_ontology, device=x.device)
        return self.block(x, c)


def create_model(arch: str, d_model: int = 18, d_ontology: int = 16, device: str = "cpu"):
    """Factory for creating baseline models."""
    if arch == "mamba_only":
        return MambaOnlyBlock(d_model, device=device)
    elif arch == "mamba_mlp_gate":
        return MambaMLPGateBlock(d_model, device=device)
    elif arch == "mamba_qkan_no_kg":
        return MambaQKANNoKGBlock(d_model, d_ontology=d_ontology, device=device)
    elif arch == "o_qkan_mamba":
        return OQKANMambaBlock(
            d_model=d_model, latent_dim=8, d_ontology=d_ontology,
            reps=3, use_mamba=False, device=device,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

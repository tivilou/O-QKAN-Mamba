import torch
import torch.nn as nn

from .spectral_qkan_gate import SpectralQKANGate

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
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> (B, L, D)"""
        h = x.transpose(1, 2)
        h = self.conv(h)[:, :, : x.shape[1]]
        return h.transpose(1, 2)


class QMambaBlock(nn.Module):
    """Mamba block with SpectralQKANGate.

    forward: y = x + gate(x) * mixer(x)
    """

    def __init__(
        self,
        d_model: int,
        latent_dim: int = 16,
        reps: int = 2,
        use_mamba: bool = True,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        solver: str = "exact",
        device: str = "cuda",
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, device=device)
        self.gate = SpectralQKANGate(
            d_model=d_model,
            latent_dim=latent_dim,
            reps=reps,
            solver=solver,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> (B, L, D)"""
        h = self.norm(x)
        return x + self.gate(h) * self.mixer(h)

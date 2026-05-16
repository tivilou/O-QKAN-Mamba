import torch
import torch.nn as nn

from qkan import QKAN


class SpectralQKANGate(nn.Module):
    """HQKAN-based sigmoid gate using bottleneck pattern.

    Flow: down-project → QKAN → up-project → sigmoid
    """

    def __init__(
        self,
        d_model: int,
        latent_dim: int = 16,
        reps: int = 2,
        solver: str = "exact",
        device: str = "cuda",
    ):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim

        self.down_proj = nn.Linear(d_model, latent_dim, device=device)
        self.qkan = QKAN(
            width=[latent_dim, latent_dim],
            reps=reps,
            group=3,
            solver=solver,
            preact_trainable=True,
            postact_weight_trainable=True,
            postact_bias_trainable=True,
            ba_trainable=True,
            device=device,
        )
        self.up_proj = nn.Linear(latent_dim, d_model, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> (B, L, D) sigmoid gate values."""
        B, L, D = x.shape
        h = self.down_proj(x.reshape(B * L, D))
        h = self.qkan(h)
        h = self.up_proj(h)
        return torch.sigmoid(h.reshape(B, L, D))

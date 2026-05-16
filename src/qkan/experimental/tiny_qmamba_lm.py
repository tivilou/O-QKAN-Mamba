import torch
import torch.nn as nn

from .qmamba_block import QMambaBlock


class TinyQMambaLM(nn.Module):
    """Tiny language model using QMambaBlocks for architecture validation."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 4,
        latent_dim: int = 16,
        reps: int = 2,
        use_mamba: bool = True,
        solver: str = "exact",
        device: str = "cuda",
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, device=device)
        self.blocks = nn.ModuleList([
            QMambaBlock(
                d_model=d_model,
                latent_dim=latent_dim,
                reps=reps,
                use_mamba=use_mamba,
                solver=solver,
                device=device,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model, device=device)
        self.lm_head = nn.Linear(d_model, vocab_size, device=device)

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor = None
    ):
        """
        input_ids: (B, L) long tensor
        targets: (B, L) long tensor, optional
        Returns: (logits, loss) where loss is None if targets not provided
        """
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

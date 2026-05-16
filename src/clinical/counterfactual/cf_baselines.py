"""Counterfactual baselines: Integrated Gradients and SHAP simulator.

Used for speed comparison against our single-forward-pass approach.
"""
import time
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class BaselineResult:
    attributions: torch.Tensor
    elapsed_ms: float
    n_forward_passes: int


class IntegratedGradientsBaseline:
    """Integrated Gradients attribution baseline."""

    def __init__(self, model: nn.Module, n_steps: int = 50):
        self.model = model
        self.n_steps = n_steps

    def attribute(
        self, x: torch.Tensor, concept_emb: torch.Tensor, target_idx: int = 0
    ) -> BaselineResult:
        """Compute IG attributions for concept_emb dimensions."""
        t0 = time.time()
        baseline = torch.zeros_like(concept_emb)
        attributions = torch.zeros_like(concept_emb)

        for step in range(self.n_steps):
            alpha = step / self.n_steps
            interp = baseline + alpha * (concept_emb - baseline)
            interp.requires_grad_(True)

            out = self.model(x, interp)
            if out.dim() == 3:
                target = out[0, -1, target_idx]
            else:
                target = out[0, target_idx]
            target.backward()

            if interp.grad is not None:
                attributions += interp.grad
            interp.requires_grad_(False)

        attributions = attributions * (concept_emb - baseline) / self.n_steps
        elapsed = (time.time() - t0) * 1000

        return BaselineResult(
            attributions=attributions.detach().squeeze(),
            elapsed_ms=elapsed,
            n_forward_passes=self.n_steps,
        )


class SHAPSimulator:
    """Simulates SHAP computation cost (N forward passes per feature)."""

    def __init__(self, model: nn.Module, n_samples: int = 100):
        self.model = model
        self.n_samples = n_samples

    def attribute(
        self, x: torch.Tensor, concept_emb: torch.Tensor
    ) -> BaselineResult:
        """Simulate SHAP by running N forward passes with masked inputs."""
        t0 = time.time()
        d = concept_emb.shape[-1]
        attributions = torch.zeros(d)

        for _ in range(self.n_samples):
            mask = torch.randint(0, 2, (1, d)).float()
            masked_emb = concept_emb * mask
            with torch.no_grad():
                out = self.model(x, masked_emb)

        elapsed = (time.time() - t0) * 1000

        return BaselineResult(
            attributions=attributions,
            elapsed_ms=elapsed,
            n_forward_passes=self.n_samples,
        )

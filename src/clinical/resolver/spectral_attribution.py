"""Spectral attribution: identify which DARUAN frequencies cause violations.

For each violation type, the "target" for attribution:
- Type violation (sepsis without infection): target = sepsis logit
- Hierarchy violation: target = conflicting code logit
- Temporal violation: target = onset prediction logit
- Treatment violation: target = treatment recommendation logit
"""
import torch
import torch.nn as nn


class SpectralAttributor:
    """Compute per-frequency attribution scores via gradient analysis."""

    def __init__(self, model: nn.Module):
        """
        Args:
            model: OQKANMambaBlock or similar with .gate.daruan accessible
        """
        self.model = model

    def compute_attribution(
        self,
        x: torch.Tensor,
        concept_emb: torch.Tensor,
        target_idx: int = 0,
    ) -> torch.Tensor:
        """Compute attribution of each frequency layer to the target output.

        Uses gradient of target output w.r.t. modulated weights.

        Args:
            x: (B, L, d_model) input
            concept_emb: (B, d_ontology) concept embedding
            target_idx: which output dimension to attribute
        Returns:
            (reps,) attribution scores per frequency layer
        """
        self.model.eval()
        x_in = x.detach().requires_grad_(False)
        c_in = concept_emb.detach().requires_grad_(False)

        daruan = self._get_daruan()
        reps = daruan.reps

        # Compute modulated weights with gradient tracking
        w_mod = daruan.compute_modulated_weights(c_in)
        w_mod.retain_grad()

        # Forward with explicit weight injection
        # We need to run the circuit manually to get gradients on w_mod
        B = x_in.shape[0]
        if x_in.dim() == 3:
            B, L, D = x_in.shape
            x_flat = x_in.reshape(B * L, D)
            c_flat = c_in.unsqueeze(1).expand(B, L, -1).reshape(B * L, -1)
        else:
            x_flat = x_in
            c_flat = c_in

        gate = self.model.gate if hasattr(self.model, 'gate') else self.model
        h = gate.down_proj(x_flat)

        # Run DARUAN circuit with w_mod tracked
        from qkan.daruan.torch_qc import StateVector, TorchGates
        psi = StateVector(h.shape[0], daruan.dim, device=h.device, dtype=torch.complex64)
        psi.h()

        w_mod_expanded = daruan.compute_modulated_weights(c_flat)

        for l in range(reps):
            psi.rz(daruan.theta[:, l, 0])
            psi.ry(daruan.theta[:, l, 1])
            encoded = h * w_mod_expanded[:, l:l+1]
            rz_gate = TorchGates.rz_gate(encoded, dtype=torch.complex64)
            psi.state = torch.einsum("mnbi,bin->bim", rz_gate, psi.state)

        psi.rz(daruan.theta[:, reps, 0])
        psi.ry(daruan.theta[:, reps, 1])
        postacts = psi.measure_z()

        out = gate.up_proj(postacts * daruan.postact_weight + daruan.postact_bias)
        if x_in.dim() == 3:
            out = out.reshape(B, L, -1)

        # Target: sum of target dimension
        target = out[:, :, target_idx].sum() if out.dim() == 3 else out[:, target_idx].sum()
        target.backward(retain_graph=True)

        # Attribution = gradient magnitude of w_mod per layer
        if w_mod_expanded.grad is not None:
            attribution = w_mod_expanded.grad.abs().mean(dim=0)  # [reps]
        else:
            attribution = torch.zeros(reps)

        self.model.train()
        return attribution

    def _get_daruan(self):
        if hasattr(self.model, 'gate'):
            return self.model.gate.daruan
        if hasattr(self.model, 'daruan'):
            return self.model.daruan
        raise AttributeError("Cannot find DARUAN module in model")

    def compute_attribution_simple(
        self,
        x: torch.Tensor,
        concept_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Simplified attribution using w_base gradient.

        More robust than full circuit attribution for resolver use.
        """
        daruan = self._get_daruan()
        reps = daruan.reps

        daruan.w_base.requires_grad_(True)
        if daruan.w_base.grad is not None:
            daruan.w_base.grad.zero_()

        out = self.model(x, concept_emb)
        if out.dim() == 3:
            target = out.sum()
        else:
            target = out.sum()
        target.backward()

        if daruan.w_base.grad is not None:
            attribution = daruan.w_base.grad.abs()
        else:
            attribution = torch.zeros(reps)

        daruan.w_base.requires_grad_(True)
        return attribution.detach()

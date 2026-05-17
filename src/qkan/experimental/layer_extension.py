"""Layer Extension for Deliberative Reasoning.

Maps QKAN's layer extension mechanism to agent deliberation:
- Reactive mode: base layers (fast, coarse prediction)
- Deliberative mode: extend specific bands (more frequencies, finer discrimination)

Theoretical basis: each additional re-uploading layer in a band multiplies
that band's frequency modes by up to 3x (Theorem 2.2 from QKAN paper).
More layers = exponentially more frequencies = finer discrimination.
"""
import torch
import torch.nn as nn

from .opfa_daruan import OPFADaruan, FrequencyBandAllocator


class DynamicLayerExtender:
    """Extends specific frequency bands by adding re-uploading layers.

    New layers are initialized to near-identity (small random weights)
    so they don't disrupt existing predictions until trained/adapted.
    """

    @staticmethod
    def extend_band(
        model: OPFADaruan,
        band_name: str,
        n_extra_layers: int = 1,
    ) -> OPFADaruan:
        """Create a new model with extended layers in the specified band.

        Args:
            model: existing OPFADaruan model
            band_name: which band to extend
            n_extra_layers: how many layers to add
        Returns:
            New OPFADaruan with extended band (weights copied from original)
        """
        band = model.allocator.get_band_by_name(band_name)
        old_reps = model.reps
        new_reps = old_reps + n_extra_layers

        # Build new band config with extended band
        new_config = []
        for b in model.allocator.bands:
            n_layers = len(b.layer_indices)
            if b.name == band_name:
                n_layers += n_extra_layers
            new_config.append({
                "name": b.name,
                "axis": b.encoding_axis,
                "n_layers": n_layers,
            })

        # Create new model
        new_model = OPFADaruan(
            dim=model.dim,
            reps=new_reps,
            d_ontology=model.d_ontology,
            band_config=new_config,
            device=model.device,
        )

        # Copy existing parameters
        with torch.no_grad():
            # Copy theta for existing layers
            new_model.theta[:, :old_reps + 1, :].copy_(model.theta)
            # Initialize new theta layers to small values (near-identity)
            if new_reps > old_reps:
                new_model.theta[:, old_reps + 1:, :].fill_(0.01)

            # Copy w_base for existing layers
            new_model.w_base[:old_reps].copy_(model.w_base)
            # New layers get geometric continuation within the band
            for i in range(n_extra_layers):
                idx = old_reps + i
                new_model.w_base[idx] = 2.0 ** (len(band.layer_indices) + i)

            # Copy modulation parameters
            new_model.W_KG[:old_reps].copy_(model.W_KG)
            new_model.b[:old_reps].copy_(model.b)

            # Copy post-activation and measurement
            new_model.postact_weight.copy_(model.postact_weight)
            new_model.postact_bias.copy_(model.postact_bias)
            new_model.adaptive_measure.load_state_dict(
                model.adaptive_measure.state_dict()
            )

        return new_model

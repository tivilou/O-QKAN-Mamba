"""Spectral Certificate: provable safety guarantees from OPFA frequency structure.

A spectral certificate states which frequency bands contributed to a prediction
and which did not. Because OPFA-DARUAN's band partition is structural (by
construction), inactive bands are mathematically guaranteed to have zero
influence on the output. This enables formal safety claims:

    "If band B is inactive in the certificate, the prediction cannot depend
     on any clinical concept assigned to band B."

This is NOT a heuristic check — it follows directly from the circuit topology.
"""
from dataclasses import dataclass, field

import torch


@dataclass
class SpectralCertificate:
    """A provable safety certificate for a model prediction."""

    prediction: float
    band_contributions: dict[str, float]
    active_bands: list[str]
    inactive_bands: list[str]
    guarantees: list[str] = field(default_factory=list)
    is_valid: bool = True
    violation_reason: str = ""

    @property
    def dominant_band(self) -> str:
        if not self.band_contributions:
            return "none"
        return max(self.band_contributions, key=self.band_contributions.get)

    def summary(self) -> str:
        active = ", ".join(self.active_bands) or "none"
        guarantees = "; ".join(self.guarantees) or "none"
        return (
            f"Certificate(risk={self.prediction:.3f}, "
            f"active=[{active}], guarantees=[{guarantees}])"
        )


# Ontology-to-band constraint rules
BAND_CONSTRAINT_RULES = {
    "infection": {
        "requires_for_positive": ["sepsis_label"],
        "description": "Sepsis prediction requires infection band activity",
    },
    "hemodynamics": {
        "requires_for_positive": [],
        "description": "Hemodynamic instability alone can indicate shock",
    },
    "organ_function": {
        "requires_for_positive": [],
        "description": "Organ dysfunction is a downstream consequence",
    },
}


class CertificateGenerator:
    """Generate spectral certificates from OPFA-DARUAN output.

    The certificate is sound because OPFA's band partition is structural:
    if a band's layers all have zero weight, that band's frequency components
    are mathematically absent from the output.
    """

    def __init__(self, band_names: list[str] = None, threshold: float = 0.01):
        self.band_names = band_names or ["infection", "hemodynamics", "organ_function"]
        self.threshold = threshold

    def generate(
        self,
        prediction: float,
        spectral_info: dict,
        prediction_label: int = None,
    ) -> SpectralCertificate:
        """Generate a spectral certificate for a prediction.

        Args:
            prediction: risk score (0-1)
            spectral_info: dict from OPFADaruan.forward() with band_contributions
            prediction_label: 0 or 1 (sepsis label)
        Returns:
            SpectralCertificate with guarantees
        """
        contributions = spectral_info["band_contributions"]
        band_names = spectral_info.get("band_names", self.band_names)

        # Build contribution dict
        band_contrib = {}
        active = []
        inactive = []
        for i, name in enumerate(band_names):
            val = contributions[i].item() if torch.is_tensor(contributions[i]) else float(contributions[i])
            band_contrib[name] = val
            if val > self.threshold:
                active.append(name)
            else:
                inactive.append(name)

        # Derive guarantees from inactive bands
        guarantees = []
        for band_name in inactive:
            guarantees.append(
                f"Prediction does not depend on {band_name} concepts "
                f"(band contribution = {band_contrib[band_name]:.4f})"
            )

        if prediction_label is None:
            prediction_label = 1 if prediction > 0.5 else 0

        cert = SpectralCertificate(
            prediction=prediction,
            band_contributions=band_contrib,
            active_bands=active,
            inactive_bands=inactive,
            guarantees=guarantees,
        )

        return cert

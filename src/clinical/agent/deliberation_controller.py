"""Deliberation Controller: decides when and where to extend layers.

Maps the QKAN layer extension mechanism to clinical agent deliberation:
- If certificate verification passes → reactive mode (no extension needed)
- If certificate fails → identify which band to extend → deliberative mode

This implements "Deliberative vs Reactive Agent Architectures" from the
JBHI special issue CFP, grounded in quantum circuit theory.
"""
import torch

from ..certificates.certificate import SpectralCertificate, CertificateGenerator
from ..certificates.certificate_verifier import CertificateVerifier


class DeliberationController:
    """Controls when the agent should 'think deeper' via layer extension."""

    def __init__(self, max_extensions: int = 3, max_extra_layers: int = 2):
        self.max_extensions = max_extensions
        self.max_extra_layers = max_extra_layers
        self.generator = CertificateGenerator()
        self.verifier = CertificateVerifier()

    def should_deliberate(self, certificate: SpectralCertificate) -> bool:
        """Determine if deliberation (layer extension) is needed."""
        verified = self.verifier.verify(certificate)
        return not verified.is_valid

    def identify_target_band(self, certificate: SpectralCertificate) -> str:
        """Identify which band needs extension based on the violation."""
        verified = self.verifier.verify(certificate)
        if verified.is_valid:
            return ""

        # If violation mentions a specific band, target that band
        for band_name in ["infection", "hemodynamics", "organ_function"]:
            if band_name in verified.violation_reason:
                return band_name

        # Default: extend the band with lowest contribution
        if certificate.band_contributions:
            return min(certificate.band_contributions, key=certificate.band_contributions.get)
        return "infection"

    def deliberation_plan(self, certificate: SpectralCertificate) -> dict:
        """Generate a deliberation plan: which band to extend and by how much.

        Returns:
            dict with 'action', 'target_band', 'n_layers', 'reason'
        """
        if not self.should_deliberate(certificate):
            return {
                "action": "none",
                "target_band": "",
                "n_layers": 0,
                "reason": "Certificate valid, no deliberation needed",
            }

        target = self.identify_target_band(certificate)
        return {
            "action": "extend",
            "target_band": target,
            "n_layers": 1,
            "reason": (
                f"Certificate violation: {certificate.violation_reason or 'unknown'}. "
                f"Extending {target} band to increase frequency resolution."
            ),
        }

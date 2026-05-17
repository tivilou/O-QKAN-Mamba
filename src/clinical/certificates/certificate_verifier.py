"""Certificate Verifier: checks that spectral certificates are consistent.

Verification rules:
1. Soundness: if a band is inactive, the prediction must not logically
   require that band's concepts
2. Consistency: active bands must be compatible with the prediction label
3. Completeness: at least one band must be active for any non-trivial prediction
"""
from .certificate import SpectralCertificate, BAND_CONSTRAINT_RULES


class CertificateVerifier:
    """Verify spectral certificates against clinical constraints."""

    def __init__(self, rules: dict = None):
        self.rules = rules or BAND_CONSTRAINT_RULES

    def verify(self, certificate: SpectralCertificate) -> SpectralCertificate:
        """Verify a certificate and update its validity status.

        Checks:
        1. If prediction is positive (sepsis=1), infection band must be active
        2. At least one band must be active for non-trivial predictions
        3. Band contributions must sum to ~1.0

        Returns the certificate with is_valid and violation_reason updated.
        """
        cert = certificate
        cert.is_valid = True
        cert.violation_reason = ""

        # Check 1: Sepsis prediction requires infection band
        is_positive = cert.prediction > 0.5
        if is_positive and "infection" in [b for b in self.rules]:
            rule = self.rules["infection"]
            if "infection" in cert.inactive_bands:
                cert.is_valid = False
                cert.violation_reason = (
                    "Sepsis prediction (risk={:.3f}) but infection band is "
                    "inactive (contribution={:.4f}). A sepsis diagnosis "
                    "requires evidence of infection.".format(
                        cert.prediction,
                        cert.band_contributions.get("infection", 0.0),
                    )
                )
                return cert

        # Check 2: Non-trivial prediction needs at least one active band
        if abs(cert.prediction - 0.5) > 0.1 and not cert.active_bands:
            cert.is_valid = False
            cert.violation_reason = (
                "Non-trivial prediction (risk={:.3f}) but no active bands. "
                "This indicates a degenerate model state.".format(cert.prediction)
            )
            return cert

        # Check 3: Contributions should sum to approximately 1.0
        total = sum(cert.band_contributions.values())
        if total > 0 and abs(total - 1.0) > 0.1:
            cert.is_valid = False
            cert.violation_reason = (
                f"Band contributions sum to {total:.3f}, expected ~1.0"
            )
            return cert

        return cert

    def explain_violation(self, certificate: SpectralCertificate) -> str:
        """Generate human-readable explanation of a certificate violation."""
        if certificate.is_valid:
            return "Certificate is valid. No violations detected."

        explanation = f"VIOLATION: {certificate.violation_reason}\n"
        explanation += f"  Prediction: {certificate.prediction:.3f}\n"
        explanation += f"  Active bands: {certificate.active_bands}\n"
        explanation += f"  Inactive bands: {certificate.inactive_bands}\n"

        if "infection" in certificate.inactive_bands and certificate.prediction > 0.5:
            explanation += (
                "\n  Resolution: The model predicts sepsis without relying on "
                "infection-related frequencies. Either:\n"
                "  (a) Extend the infection band (add layers) via deliberation, or\n"
                "  (b) Reduce the prediction confidence to below 0.5"
            )

        return explanation

"""Hard projection resolver: baseline that clips predictions to satisfy rules.

Simple approach: if prediction violates a rule, force the prediction to
the nearest rule-compliant value. This destroys neural signal but always
resolves violations. Used as comparison baseline for SpectralConflictResolver.
"""
from dataclasses import dataclass

from ..validator.dag_validator import DAGValidator
from ..validator.violation_types import ViolationType


@dataclass
class HardProjectionResult:
    original_prediction: dict
    resolved_prediction: dict
    violations_before: int
    violations_after: int
    prediction_drift: float


class HardProjectionResolver:
    """Resolve violations by hard-projecting predictions onto compliant space."""

    def __init__(self, validator: DAGValidator):
        self.validator = validator

    def resolve(self, prediction: dict, patient: dict) -> HardProjectionResult:
        """Force predictions to satisfy all rules.

        Strategy per violation type:
        - Type: if sepsis predicted without evidence → force risk to 0
        - Hierarchy: remove conflicting codes
        - Temporal: remove events without prerequisites
        - Treatment: flag only (don't modify treatment)
        """
        original_results = self.validator.validate_all(prediction, patient)
        violations_before = sum(len(v) for v in original_results.values())

        resolved = prediction.copy()

        # Type violations: force sepsis prediction to 0
        if original_results[ViolationType.TYPE]:
            resolved["sepsis_risk"] = 0.0
            resolved["sepsis_label"] = 0

        # Hierarchy violations: keep only most specific codes
        if original_results[ViolationType.HIERARCHY]:
            codes = resolved.get("predicted_codes", [])
            for violation in original_results[ViolationType.HIERARCHY]:
                if "ancestor" in violation.evidence:
                    ancestor = violation.evidence["ancestor"]
                    codes = [c for c in codes if c != ancestor]
            resolved["predicted_codes"] = codes

        # Temporal violations: remove events without prerequisites
        if original_results[ViolationType.TEMPORAL]:
            resolved["prediction_timeline"] = []

        final_results = self.validator.validate_all(resolved, patient)
        violations_after = sum(len(v) for v in final_results.values())

        orig_risk = prediction.get("sepsis_risk", 0)
        new_risk = resolved.get("sepsis_risk", 0)
        drift = abs(orig_risk - new_risk)

        return HardProjectionResult(
            original_prediction=prediction,
            resolved_prediction=resolved,
            violations_before=violations_before,
            violations_after=violations_after,
            prediction_drift=drift,
        )

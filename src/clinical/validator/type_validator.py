"""Type validator: checks predictions against Sepsis-3 clinical rules.

Reference: Singer et al., JAMA 2016;315(8):801-810.
"""
from ..knowledge.sepsis_rules import sofa_score, suspected_infection
from .violation_types import Violation, ViolationType


class TypeValidator:
    """Validates that predictions are consistent with Sepsis-3 criteria."""

    def validate(self, prediction: dict, patient_features: dict) -> list[Violation]:
        """
        Args:
            prediction: dict with 'sepsis_risk' (float 0-1), 'sepsis_label' (0/1)
            patient_features: dict with vital/lab values at prediction time
        Returns:
            List of type violations
        """
        violations = []

        sepsis_pred = prediction.get("sepsis_label", 0)
        risk = prediction.get("sepsis_risk", 0.0)

        if sepsis_pred == 1:
            # Rule 1: Sepsis requires suspected infection
            if not suspected_infection(patient_features):
                violations.append(Violation(
                    type=ViolationType.TYPE,
                    description="Sepsis predicted but no infection signs present "
                                "(WBC normal, temperature normal, HR normal)",
                    severity=0.8,
                    source_rule="Sepsis-3: Singer et al. JAMA 2016 — "
                                "sepsis requires suspected infection",
                    evidence={
                        "wbc": patient_features.get("wbc"),
                        "temperature": patient_features.get("temperature"),
                        "heart_rate": patient_features.get("heart_rate"),
                    },
                ))

            # Rule 2: Sepsis requires SOFA >= 2
            sofa = sofa_score(patient_features)
            if sofa < 2:
                violations.append(Violation(
                    type=ViolationType.TYPE,
                    description=f"Sepsis predicted but SOFA score is {sofa} (< 2)",
                    severity=0.9,
                    source_rule="Sepsis-3: SOFA increase >= 2 required for sepsis",
                    evidence={"sofa_score": sofa},
                ))

        # Rule 3: High risk but no supporting features
        if risk > 0.8 and sofa_score(patient_features) == 0:
            if not suspected_infection(patient_features):
                violations.append(Violation(
                    type=ViolationType.TYPE,
                    description=f"High sepsis risk ({risk:.2f}) but all vitals normal",
                    severity=0.6,
                    source_rule="Clinical plausibility: high risk requires abnormal signs",
                    evidence={"risk": risk, "sofa": 0},
                ))

        return violations

"""Treatment validator: checks drug-allergy and drug-drug contraindications.

Validates patient's actual medication list against their clinical state,
allergies, and known drug interactions.
"""
from ..knowledge.drugbank_mini import DrugInteractionChecker
from .violation_types import Violation, ViolationType


class TreatmentValidator:
    """Validates treatment safety against patient state."""

    def __init__(self):
        self.checker = DrugInteractionChecker()

    def validate(
        self, patient_meds: list[str], patient_state: dict
    ) -> list[Violation]:
        """
        Args:
            patient_meds: list of current medication names
            patient_state: dict with 'allergies' (list), 'conditions' (list),
                          and clinical features
        Returns:
            List of treatment violations
        """
        violations = []
        allergies = patient_state.get("allergies", [])

        # Check drug-allergy and drug-drug interactions
        warnings = self.checker.check_contraindication(patient_meds, allergies)
        for warning in warnings:
            if "ALLERGY" in warning:
                violations.append(Violation(
                    type=ViolationType.TREATMENT,
                    description=warning,
                    severity=1.0,
                    source_rule="Drug-allergy contraindication (DrugBank)",
                    evidence={"meds": patient_meds, "allergies": allergies},
                ))
            elif "major" in warning:
                violations.append(Violation(
                    type=ViolationType.TREATMENT,
                    description=warning,
                    severity=0.9,
                    source_rule="Major drug-drug interaction (DrugBank)",
                    evidence={"meds": patient_meds},
                ))
            else:
                violations.append(Violation(
                    type=ViolationType.TREATMENT,
                    description=warning,
                    severity=0.5,
                    source_rule="Drug-drug interaction (DrugBank)",
                    evidence={"meds": patient_meds},
                ))

        # Check vasopressor without hypotension
        vasopressors = ["norepinephrine", "vasopressin", "epinephrine"]
        has_vasopressor = any(m in vasopressors for m in patient_meds)
        map_val = patient_state.get("map")
        if has_vasopressor and map_val is not None and map_val > 80:
            violations.append(Violation(
                type=ViolationType.TREATMENT,
                description=(
                    f"Vasopressor administered but MAP is {map_val:.0f} "
                    f"(> 80 mmHg, no hypotension)"
                ),
                severity=0.4,
                source_rule="Clinical: vasopressors indicated for MAP < 65",
                evidence={"map": map_val, "vasopressor": True},
            ))

        return violations

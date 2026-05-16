"""DAG Validator: aggregates all 4 violation type validators."""
from .violation_types import Violation, ViolationType
from .type_validator import TypeValidator
from .hierarchy_validator import HierarchyValidator
from .temporal_validator import TemporalValidator
from .treatment_validator import TreatmentValidator


class DAGValidator:
    """Aggregates all clinical validators into a single interface."""

    def __init__(self):
        self.type_validator = TypeValidator()
        self.hierarchy_validator = HierarchyValidator()
        self.temporal_validator = TemporalValidator()
        self.treatment_validator = TreatmentValidator()

    def validate_all(self, prediction: dict, patient: dict) -> dict[ViolationType, list[Violation]]:
        """Run all validators on a prediction + patient context.

        Args:
            prediction: dict with model outputs:
                - sepsis_risk: float (0-1)
                - sepsis_label: int (0/1)
                - predicted_codes: list[str] (ICD-10 codes)
                - prediction_timeline: list[dict] (predicted events)
            patient: dict with patient context:
                - features: dict (current vitals/labs)
                - history: list[dict] (past events with hours)
                - meds: list[str] (current medications)
                - allergies: list[str]
                - conditions: list[str]

        Returns:
            Dict mapping ViolationType to list of violations
        """
        results = {vt: [] for vt in ViolationType}

        # Type validation
        features = patient.get("features", {})
        results[ViolationType.TYPE] = self.type_validator.validate(prediction, features)

        # Hierarchy validation
        codes = prediction.get("predicted_codes", [])
        if codes:
            results[ViolationType.HIERARCHY] = self.hierarchy_validator.validate(codes)

        # Temporal validation
        pred_timeline = prediction.get("prediction_timeline", [])
        history = patient.get("history", [])
        if pred_timeline or history:
            results[ViolationType.TEMPORAL] = self.temporal_validator.validate(
                pred_timeline, history
            )

        # Treatment validation
        meds = patient.get("meds", [])
        if meds:
            patient_state = {
                "allergies": patient.get("allergies", []),
                "conditions": patient.get("conditions", []),
                **features,
            }
            results[ViolationType.TREATMENT] = self.treatment_validator.validate(
                meds, patient_state
            )

        return results

    def count_violations(self, results: dict[ViolationType, list[Violation]]) -> dict:
        """Summarize violation counts."""
        return {vt.value: len(vs) for vt, vs in results.items()}

    def has_violations(self, results: dict[ViolationType, list[Violation]]) -> bool:
        return any(len(vs) > 0 for vs in results.values())

"""FHIR Output Adapter: wrap model output as FHIR R4 Bundle.

Resources generated:
- RiskAssessment: sepsis risk score
- Observation: derived features (SOFA score, concepts)
- DetectedIssue: DAG violations
- ClinicalImpression: reasoning trace
"""
import json
from datetime import datetime
from typing import Optional


class FHIROutputAdapter:
    """Convert model predictions to FHIR R4 Bundle."""

    def create_output_bundle(
        self,
        patient_id: str,
        risk_score: float,
        violations: list = None,
        concepts: list = None,
        reasoning_trace: str = "",
    ) -> dict:
        """Create FHIR Bundle with model outputs.

        Args:
            patient_id: FHIR Patient resource ID
            risk_score: sepsis risk (0-1)
            violations: list of Violation objects
            concepts: list of active concept names
            reasoning_trace: human-readable explanation
        Returns:
            FHIR R4 Bundle dict
        """
        entries = []
        now = datetime.utcnow().isoformat() + "Z"

        # RiskAssessment
        entries.append(self._risk_assessment(patient_id, risk_score, now))

        # DetectedIssue for each violation
        if violations:
            for i, v in enumerate(violations):
                entries.append(self._detected_issue(patient_id, v, i, now))

        # Observation for active concepts
        if concepts:
            entries.append(self._concepts_observation(patient_id, concepts, now))

        # ClinicalImpression for reasoning
        if reasoning_trace:
            entries.append(self._clinical_impression(
                patient_id, reasoning_trace, now
            ))

        return {
            "resourceType": "Bundle",
            "type": "collection",
            "timestamp": now,
            "entry": entries,
        }

    def _risk_assessment(self, patient_id: str, risk: float, now: str) -> dict:
        return {"resource": {
            "resourceType": "RiskAssessment",
            "status": "final",
            "subject": {"reference": f"Patient/{patient_id}"},
            "occurrenceDateTime": now,
            "prediction": [{
                "outcome": {"text": "Sepsis"},
                "probabilityDecimal": round(risk, 4),
                "qualitativeRisk": {"text": self._risk_level(risk)},
            }],
        }}

    def _detected_issue(self, patient_id: str, violation, idx: int, now: str) -> dict:
        return {"resource": {
            "resourceType": "DetectedIssue",
            "status": "final",
            "code": {"text": violation.type.value if hasattr(violation, 'type') else "unknown"},
            "severity": "high" if getattr(violation, 'severity', 0) > 0.7 else "moderate",
            "detail": str(violation),
            "identifiedDateTime": now,
            "patient": {"reference": f"Patient/{patient_id}"},
        }}

    def _concepts_observation(self, patient_id: str, concepts: list, now: str) -> dict:
        return {"resource": {
            "resourceType": "Observation",
            "status": "final",
            "code": {"coding": [{"code": "active-concepts", "display": "Active Clinical Concepts"}]},
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": now,
            "valueString": ", ".join(concepts),
        }}

    def _clinical_impression(self, patient_id: str, trace: str, now: str) -> dict:
        return {"resource": {
            "resourceType": "ClinicalImpression",
            "status": "completed",
            "subject": {"reference": f"Patient/{patient_id}"},
            "date": now,
            "summary": trace,
        }}

    @staticmethod
    def _risk_level(risk: float) -> str:
        if risk >= 0.8:
            return "high"
        if risk >= 0.5:
            return "moderate"
        if risk >= 0.2:
            return "low"
        return "negligible"

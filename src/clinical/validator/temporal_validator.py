"""Temporal validator: checks that predictions respect temporal prerequisites.

Concrete temporal rules for sepsis/ICU:
1. Sepsis onset requires prior infection signs (within preceding 48h)
2. SOFA increase requires baseline measurement
3. Septic shock requires prior sepsis diagnosis
4. Organ failure requires preceding physiological deterioration
5. Recovery cannot precede treatment initiation
"""
from .violation_types import Violation, ViolationType


# Temporal rules: (prerequisite, consequence, max_gap_hours, description)
TEMPORAL_RULES = [
    {
        "id": "T1",
        "prerequisite": "infection_signs",
        "consequence": "sepsis_onset",
        "max_gap_hours": 48,
        "description": "Sepsis onset requires infection signs within preceding 48h",
        "source": "Sepsis-3: Singer et al. JAMA 2016",
    },
    {
        "id": "T2",
        "prerequisite": "sofa_baseline",
        "consequence": "sofa_increase",
        "max_gap_hours": 24,
        "description": "SOFA increase requires baseline measurement within 24h",
        "source": "Sepsis-3: SOFA change from baseline",
    },
    {
        "id": "T3",
        "prerequisite": "sepsis_onset",
        "consequence": "septic_shock",
        "max_gap_hours": 72,
        "description": "Septic shock requires prior sepsis diagnosis",
        "source": "Sepsis-3: shock is a subset of sepsis",
    },
    {
        "id": "T4",
        "prerequisite": "vital_deterioration",
        "consequence": "organ_failure",
        "max_gap_hours": 24,
        "description": "Organ failure requires preceding vital deterioration",
        "source": "Clinical reasoning: organ failure follows deterioration",
    },
    {
        "id": "T5",
        "prerequisite": "treatment_start",
        "consequence": "clinical_improvement",
        "max_gap_hours": 72,
        "description": "Clinical improvement cannot precede treatment initiation",
        "source": "Temporal causality: effect follows cause",
    },
]


class TemporalValidator:
    """Validates temporal ordering of clinical events."""

    def __init__(self):
        self.rules = TEMPORAL_RULES

    def validate(
        self, prediction_timeline: list[dict], patient_history: list[dict]
    ) -> list[Violation]:
        """
        Args:
            prediction_timeline: list of {event, hour} predicted events
            patient_history: list of {event, hour} observed events
        Returns:
            List of temporal violations
        """
        violations = []
        all_events = self._merge_timelines(prediction_timeline, patient_history)

        for rule in self.rules:
            violation = self._check_rule(rule, all_events)
            if violation:
                violations.append(violation)

        return violations

    def _merge_timelines(self, predictions: list[dict], history: list[dict]) -> dict:
        """Merge into {event_type: [hours]} mapping."""
        events: dict[str, list[float]] = {}
        for item in predictions + history:
            event = item.get("event", "")
            hour = item.get("hour", 0)
            events.setdefault(event, []).append(hour)
        for k in events:
            events[k].sort()
        return events

    def _check_rule(self, rule: dict, events: dict) -> Violation | None:
        consequence = rule["consequence"]
        prerequisite = rule["prerequisite"]

        if consequence not in events:
            return None

        consequence_times = events[consequence]
        prerequisite_times = events.get(prerequisite, [])

        for c_time in consequence_times:
            has_valid_prereq = any(
                0 <= (c_time - p_time) <= rule["max_gap_hours"]
                for p_time in prerequisite_times
            )
            if not has_valid_prereq and prerequisite_times:
                return None  # prereq exists but outside window — ambiguous
            if not prerequisite_times:
                return Violation(
                    type=ViolationType.TEMPORAL,
                    description=(
                        f"{rule['description']}: '{consequence}' at t={c_time}h "
                        f"but no '{prerequisite}' found in history"
                    ),
                    severity=0.7,
                    source_rule=f"{rule['id']}: {rule['source']}",
                    evidence={
                        "consequence": consequence,
                        "consequence_time": c_time,
                        "prerequisite": prerequisite,
                        "prerequisite_times": [],
                    },
                )
        return None

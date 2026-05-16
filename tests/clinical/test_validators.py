"""Tests for DAG validators."""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.clinical.validator.violation_types import Violation, ViolationType
from src.clinical.validator.type_validator import TypeValidator
from src.clinical.validator.hierarchy_validator import HierarchyValidator
from src.clinical.validator.temporal_validator import TemporalValidator
from src.clinical.validator.treatment_validator import TreatmentValidator
from src.clinical.validator.dag_validator import DAGValidator


class TestTypeValidator:
    def test_no_violation_healthy(self):
        v = TypeValidator()
        prediction = {"sepsis_label": 0, "sepsis_risk": 0.1}
        features = {"wbc": 7.0, "temperature": 37.0, "heart_rate": 75}
        violations = v.validate(prediction, features)
        assert len(violations) == 0

    def test_violation_sepsis_no_infection(self):
        v = TypeValidator()
        prediction = {"sepsis_label": 1, "sepsis_risk": 0.9}
        features = {"wbc": 7.0, "temperature": 37.0, "heart_rate": 75,
                    "platelet": 250, "map": 80, "creatinine": 0.8}
        violations = v.validate(prediction, features)
        assert len(violations) >= 1
        assert any(viol.type == ViolationType.TYPE for viol in violations)

    def test_violation_sepsis_low_sofa(self):
        v = TypeValidator()
        prediction = {"sepsis_label": 1, "sepsis_risk": 0.9}
        features = {"wbc": 15.0, "temperature": 38.5, "heart_rate": 110,
                    "platelet": 250, "map": 80, "creatinine": 0.8, "bilirubin": 0.5}
        violations = v.validate(prediction, features)
        assert any("SOFA" in viol.description for viol in violations)


class TestHierarchyValidator:
    def test_no_violation(self):
        v = HierarchyValidator()
        violations = v.validate(["A41.9", "J18"])
        assert len(violations) == 0

    def test_mutex_violation(self):
        v = HierarchyValidator()
        violations = v.validate(["J96.0", "J96.1"])
        assert len(violations) >= 1
        assert violations[0].type == ViolationType.HIERARCHY
        assert "Mutually exclusive" in violations[0].description

    def test_redundancy_violation(self):
        v = HierarchyValidator()
        violations = v.validate(["R57", "R57.2"])
        assert any("ancestor" in viol.description for viol in violations)


class TestTemporalValidator:
    def test_no_violation(self):
        v = TemporalValidator()
        pred_timeline = [{"event": "sepsis_onset", "hour": 12}]
        history = [{"event": "infection_signs", "hour": 6}]
        violations = v.validate(pred_timeline, history)
        assert len(violations) == 0

    def test_violation_no_prerequisite(self):
        v = TemporalValidator()
        pred_timeline = [{"event": "sepsis_onset", "hour": 12}]
        history = []  # no infection signs at all
        violations = v.validate(pred_timeline, history)
        assert len(violations) >= 1
        assert violations[0].type == ViolationType.TEMPORAL

    def test_septic_shock_without_sepsis(self):
        v = TemporalValidator()
        pred_timeline = [{"event": "septic_shock", "hour": 24}]
        history = []  # no prior sepsis_onset
        violations = v.validate(pred_timeline, history)
        assert any("septic_shock" in v.description for v in violations)


class TestTreatmentValidator:
    def test_no_violation(self):
        v = TreatmentValidator()
        violations = v.validate(
            ["vancomycin", "norepinephrine"],
            {"allergies": [], "map": 55}
        )
        assert len(violations) == 0

    def test_allergy_violation(self):
        v = TreatmentValidator()
        violations = v.validate(
            ["piperacillin_tazobactam"],
            {"allergies": ["penicillin"], "map": 60}
        )
        assert len(violations) >= 1
        assert violations[0].severity == 1.0

    def test_drug_interaction_violation(self):
        v = TreatmentValidator()
        violations = v.validate(
            ["heparin", "enoxaparin"],
            {"allergies": [], "map": 60}
        )
        assert len(violations) >= 1
        assert any("INTERACTION" in v.description for v in violations)

    def test_vasopressor_without_hypotension(self):
        v = TreatmentValidator()
        violations = v.validate(
            ["norepinephrine"],
            {"allergies": [], "map": 90}
        )
        assert any("Vasopressor" in v.description for v in violations)


class TestDAGValidator:
    def test_aggregates_correctly(self):
        dag = DAGValidator()
        prediction = {
            "sepsis_label": 1,
            "sepsis_risk": 0.9,
            "predicted_codes": ["J96.0", "J96.1"],
            "prediction_timeline": [{"event": "sepsis_onset", "hour": 12}],
        }
        patient = {
            "features": {"wbc": 7.0, "temperature": 37.0, "heart_rate": 75,
                        "platelet": 250, "map": 80, "creatinine": 0.8},
            "history": [],
            "meds": ["heparin", "enoxaparin"],
            "allergies": [],
        }
        results = dag.validate_all(prediction, patient)
        assert ViolationType.TYPE in results
        assert ViolationType.HIERARCHY in results
        assert ViolationType.TEMPORAL in results
        assert ViolationType.TREATMENT in results
        assert dag.has_violations(results)

    def test_clean_patient_no_violations(self):
        dag = DAGValidator()
        prediction = {
            "sepsis_label": 0,
            "sepsis_risk": 0.1,
            "predicted_codes": [],
            "prediction_timeline": [],
        }
        patient = {
            "features": {"wbc": 7.0, "temperature": 37.0, "heart_rate": 75},
            "history": [],
            "meds": [],
            "allergies": [],
        }
        results = dag.validate_all(prediction, patient)
        assert not dag.has_violations(results)

    def test_deterministic(self):
        dag = DAGValidator()
        prediction = {
            "sepsis_label": 1, "sepsis_risk": 0.9,
            "predicted_codes": ["J96.0", "J96.1"],
            "prediction_timeline": [],
        }
        patient = {
            "features": {"wbc": 7.0, "temperature": 37.0},
            "history": [], "meds": [], "allergies": [],
        }
        r1 = dag.validate_all(prediction, patient)
        r2 = dag.validate_all(prediction, patient)
        c1 = dag.count_violations(r1)
        c2 = dag.count_violations(r2)
        assert c1 == c2

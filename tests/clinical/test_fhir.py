"""Tests for FHIR adapter."""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.clinical.fhir.fhir_input import FHIRInputAdapter
from src.clinical.fhir.fhir_output import FHIROutputAdapter


class TestFHIRInput:
    def test_parse_synthetic_bundle(self):
        adapter = FHIRInputAdapter()
        features = {"heart_rate": 95.0, "map": 70.0, "temperature": 37.5}
        bundle = FHIRInputAdapter.create_synthetic_bundle(features, "P001")

        result = adapter.parse_bundle(bundle)
        assert result["patient_id"] == "P001"
        assert result["features"]["heart_rate"] == 95.0
        assert result["features"]["map"] == 70.0

    def test_parse_empty_bundle(self):
        adapter = FHIRInputAdapter()
        bundle = {"resourceType": "Bundle", "type": "collection", "entry": []}
        result = adapter.parse_bundle(bundle)
        assert result["patient_id"] is None
        assert result["features"] == {}

    def test_parse_with_medications(self):
        adapter = FHIRInputAdapter()
        bundle = {
            "resourceType": "Bundle", "type": "collection",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "P002"}},
                {"resource": {
                    "resourceType": "MedicationStatement",
                    "medicationCodeableConcept": {
                        "coding": [{"display": "Vancomycin"}]
                    },
                }},
            ],
        }
        result = adapter.parse_bundle(bundle)
        assert "vancomycin" in result["meds"]


class TestFHIROutput:
    def test_output_bundle_structure(self):
        adapter = FHIROutputAdapter()
        bundle = adapter.create_output_bundle(
            patient_id="P001",
            risk_score=0.75,
            concepts=["tachycardia", "hypotension"],
            reasoning_trace="Elevated risk due to hemodynamic instability",
        )
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "collection"
        assert len(bundle["entry"]) >= 2

        # Check RiskAssessment
        risk_entry = bundle["entry"][0]["resource"]
        assert risk_entry["resourceType"] == "RiskAssessment"
        assert risk_entry["prediction"][0]["probabilityDecimal"] == 0.75

    def test_output_with_violations(self):
        from src.clinical.validator.violation_types import Violation, ViolationType
        adapter = FHIROutputAdapter()
        v = Violation(
            type=ViolationType.TYPE,
            description="Sepsis without infection",
            severity=0.8,
            source_rule="Sepsis-3",
        )
        bundle = adapter.create_output_bundle("P001", 0.9, violations=[v])
        issues = [e for e in bundle["entry"]
                  if e["resource"]["resourceType"] == "DetectedIssue"]
        assert len(issues) == 1
        assert issues[0]["resource"]["severity"] == "high"

    def test_roundtrip_consistency(self):
        """Input features → model → output bundle → parseable."""
        in_adapter = FHIRInputAdapter()
        out_adapter = FHIROutputAdapter()

        features = {"heart_rate": 110.0, "map": 55.0, "lactate": 4.0}
        in_bundle = FHIRInputAdapter.create_synthetic_bundle(features, "P003")
        parsed = in_adapter.parse_bundle(in_bundle)

        out_bundle = out_adapter.create_output_bundle(
            patient_id=parsed["patient_id"],
            risk_score=0.82,
            concepts=["tachycardia", "hypotension", "elevated_lactate"],
        )
        assert out_bundle["resourceType"] == "Bundle"
        assert len(out_bundle["entry"]) >= 1
        risk_res = out_bundle["entry"][0]["resource"]
        assert risk_res["subject"]["reference"] == "Patient/P003"

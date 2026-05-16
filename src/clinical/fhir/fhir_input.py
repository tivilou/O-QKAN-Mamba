"""FHIR Input Adapter: parse FHIR Bundle into model input format.

Handles: Patient, Observation, MedicationStatement, Condition resources.
Simplified parser for proof-of-concept; handles known synthetic format.
"""
import json
from datetime import datetime
from typing import Optional

import numpy as np


class FHIRInputAdapter:
    """Parse FHIR R4 Bundle into model-ready tensors."""

    OBSERVATION_CODE_MAP = {
        "8867-4": "heart_rate",
        "8480-6": "sbp",
        "8462-4": "dbp",
        "8478-0": "map",
        "9279-1": "resp_rate",
        "2708-6": "spo2",
        "8310-5": "temperature",
        "3150-0": "fio2",
        "6690-2": "wbc",
        "718-7": "hemoglobin",
        "777-3": "platelet",
        "2160-0": "creatinine",
        "1975-2": "bilirubin",
        "2524-7": "lactate",
        "2703-7": "pao2",
        "2019-8": "pco2",
        "2744-1": "ph",
        "2345-7": "glucose",
    }

    def parse_bundle(self, bundle: dict) -> dict:
        """Parse FHIR Bundle into structured patient data.

        Returns dict with: patient_id, features (dict), meds (list),
        conditions (list), allergies (list)
        """
        result = {
            "patient_id": None,
            "features": {},
            "meds": [],
            "conditions": [],
            "allergies": [],
            "observations": [],
        }

        entries = bundle.get("entry", [])
        for entry in entries:
            resource = entry.get("resource", {})
            rtype = resource.get("resourceType", "")

            if rtype == "Patient":
                result["patient_id"] = resource.get("id")
            elif rtype == "Observation":
                self._parse_observation(resource, result)
            elif rtype == "MedicationStatement":
                self._parse_medication(resource, result)
            elif rtype == "Condition":
                self._parse_condition(resource, result)
            elif rtype == "AllergyIntolerance":
                self._parse_allergy(resource, result)

        return result

    def _parse_observation(self, resource: dict, result: dict):
        coding = resource.get("code", {}).get("coding", [{}])
        code = coding[0].get("code", "") if coding else ""
        feature_name = self.OBSERVATION_CODE_MAP.get(code)
        if not feature_name:
            return

        value = resource.get("valueQuantity", {}).get("value")
        if value is None:
            return

        result["features"][feature_name] = float(value)
        result["observations"].append({
            "code": code,
            "feature": feature_name,
            "value": float(value),
            "time": resource.get("effectiveDateTime", ""),
        })

    def _parse_medication(self, resource: dict, result: dict):
        coding = resource.get("medicationCodeableConcept", {}).get("coding", [{}])
        display = coding[0].get("display", "") if coding else ""
        if display:
            result["meds"].append(display.lower().replace(" ", "_"))

    def _parse_condition(self, resource: dict, result: dict):
        coding = resource.get("code", {}).get("coding", [{}])
        code = coding[0].get("code", "") if coding else ""
        if code:
            result["conditions"].append(code)

    def _parse_allergy(self, resource: dict, result: dict):
        coding = resource.get("code", {}).get("coding", [{}])
        display = coding[0].get("display", "") if coding else ""
        if display:
            result["allergies"].append(display.lower())

    @staticmethod
    def create_synthetic_bundle(patient_features: dict, patient_id: str = "P001") -> dict:
        """Create a synthetic FHIR Bundle for testing."""
        entries = [
            {"resource": {"resourceType": "Patient", "id": patient_id}}
        ]
        code_map_inv = {v: k for k, v in FHIRInputAdapter.OBSERVATION_CODE_MAP.items()}
        for feat, value in patient_features.items():
            code = code_map_inv.get(feat)
            if code and value is not None:
                entries.append({
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"coding": [{"code": code, "display": feat}]},
                        "valueQuantity": {"value": value, "unit": ""},
                        "effectiveDateTime": "2024-01-01T12:00:00Z",
                    }
                })
        return {"resourceType": "Bundle", "type": "collection", "entry": entries}

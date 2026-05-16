"""Integration tests for the full ClinicalReasoningAgent."""
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.qkan.experimental.o_qkan_mamba_block import OQKANMambaBlock
from src.clinical.agent.clinical_agent import ClinicalReasoningAgent
from src.clinical.fhir.fhir_input import FHIRInputAdapter


@pytest.fixture
def agent():
    model = OQKANMambaBlock(
        d_model=18, latent_dim=8, d_ontology=16, reps=3,
        use_mamba=False, device="cpu",
    )
    return ClinicalReasoningAgent(model, d_ontology=16)


@pytest.fixture
def septic_bundle():
    features = {
        "heart_rate": 120.0, "sbp": 80.0, "dbp": 45.0, "map": 55.0,
        "resp_rate": 28.0, "spo2": 88.0, "temperature": 39.2,
        "fio2": 0.6, "wbc": 18.0, "hemoglobin": 8.5,
        "platelet": 70.0, "creatinine": 2.8, "bilirubin": 3.0,
        "lactate": 5.0, "pao2": 65.0, "pco2": 32.0, "ph": 7.28, "glucose": 180.0,
    }
    return FHIRInputAdapter.create_synthetic_bundle(features, "SEPTIC-001")


@pytest.fixture
def healthy_bundle():
    features = {
        "heart_rate": 72.0, "sbp": 120.0, "dbp": 75.0, "map": 85.0,
        "resp_rate": 14.0, "spo2": 98.0, "temperature": 36.8,
        "fio2": 0.21, "wbc": 7.0, "hemoglobin": 14.0,
        "platelet": 250.0, "creatinine": 0.9, "bilirubin": 0.5,
        "lactate": 1.0, "pao2": 95.0, "pco2": 40.0, "ph": 7.40, "glucose": 100.0,
    }
    return FHIRInputAdapter.create_synthetic_bundle(features, "HEALTHY-001")


class TestFullAgent:
    def test_agent_processes_fhir_input(self, agent, septic_bundle):
        output = agent.process(septic_bundle)
        assert output["resourceType"] == "Bundle"
        assert len(output["entry"]) >= 1

    def test_agent_output_is_valid_fhir(self, agent, healthy_bundle):
        output = agent.process(healthy_bundle)
        assert output["resourceType"] == "Bundle"
        assert output["type"] == "collection"
        risk_entry = output["entry"][0]["resource"]
        assert risk_entry["resourceType"] == "RiskAssessment"
        assert "prediction" in risk_entry
        prob = risk_entry["prediction"][0]["probabilityDecimal"]
        assert 0.0 <= prob <= 1.0

    def test_agent_handles_conflict_case(self, agent, septic_bundle):
        output = agent.process(septic_bundle)
        # Should have reasoning trace
        impressions = [e for e in output["entry"]
                      if e["resource"]["resourceType"] == "ClinicalImpression"]
        if impressions:
            assert "Risk=" in impressions[0]["resource"]["summary"]

    def test_agent_responds_to_counterfactual(self, agent, septic_bundle):
        result = agent.counterfactual(
            septic_bundle, {"scale_factor": 0.0}
        )
        assert "original_risk" in result
        assert "counterfactual_risk" in result
        assert "delta" in result
        assert result["direction"] in ["increases_risk", "decreases_risk", "no_effect"]

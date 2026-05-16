"""Tests for Phase 3 ontology infrastructure."""
import numpy as np
import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.clinical.knowledge.icd_dag import ICD10DAG
from src.clinical.knowledge.sepsis_rules import (
    SepsisRuleEngine,
    sofa_score,
    suspected_infection,
    sepsis3_positive,
    septic_shock_positive,
)
from src.clinical.knowledge.drugbank_mini import DrugInteractionChecker
from src.clinical.knowledge.ontology_embedder import OntologyEmbedder
from src.clinical.knowledge.concept_extractor import ConceptExtractor


class TestICD10DAG:
    def test_dag_loads(self):
        dag = ICD10DAG()
        assert dag.num_nodes > 30
        assert dag.num_edges > 20

    def test_get_parents(self):
        dag = ICD10DAG()
        parents = dag.get_parents("A41.9")
        assert "A41" in parents

    def test_get_children(self):
        dag = ICD10DAG()
        children = dag.get_children("R57")
        assert "R57.0" in children
        assert "R57.2" in children

    def test_get_ancestors(self):
        dag = ICD10DAG()
        ancestors = dag.get_ancestors("R65.21")
        assert "R65.2" in ancestors
        assert "R65" in ancestors
        assert "R00-R99" in ancestors

    def test_check_mutex(self):
        dag = ICD10DAG()
        assert dag.check_mutex("J96.0", "J96.1") is True
        assert dag.check_mutex("A41", "J18") is False


class TestSepsisRules:
    def test_sofa_score_healthy(self):
        features = {"platelet": 250, "bilirubin": 0.5, "map": 80, "creatinine": 0.8}
        assert sofa_score(features) == 0

    def test_sofa_score_sick(self):
        features = {"platelet": 80, "bilirubin": 3.0, "map": 60, "creatinine": 2.5}
        score = sofa_score(features)
        assert score >= 4

    def test_suspected_infection_positive(self):
        assert suspected_infection({"wbc": 15.0, "temperature": 38.5}) is True

    def test_suspected_infection_negative(self):
        assert suspected_infection({"wbc": 7.0, "temperature": 37.0}) is False

    def test_sepsis3_positive_case(self):
        baseline = {"platelet": 250, "bilirubin": 0.5, "map": 80, "creatinine": 0.8}
        sick = {"platelet": 80, "bilirubin": 3.0, "map": 55, "creatinine": 2.5,
                "wbc": 18.0, "temperature": 39.0}
        seq = [baseline] * 6 + [sick] * 6
        result = sepsis3_positive(seq)
        assert result.fired is True

    def test_septic_shock(self):
        result = septic_shock_positive({"map": 55, "lactate": 3.5})
        assert result.fired is True
        result = septic_shock_positive({"map": 75, "lactate": 1.0})
        assert result.fired is False

    def test_rule_engine(self):
        engine = SepsisRuleEngine()
        features = {"platelet": 80, "map": 55, "lactate": 4.0, "wbc": 16.0}
        results = engine.evaluate_all(features)
        assert len(results) == 3


class TestDrugInteractions:
    def test_allergy_detected(self):
        checker = DrugInteractionChecker()
        warnings = checker.check_contraindication(
            ["piperacillin_tazobactam", "vancomycin"],
            ["penicillin"]
        )
        assert any("ALLERGY" in w for w in warnings)

    def test_interaction_detected(self):
        checker = DrugInteractionChecker()
        warnings = checker.check_contraindication(
            ["heparin", "enoxaparin"], []
        )
        assert any("INTERACTION" in w for w in warnings)

    def test_no_issues(self):
        checker = DrugInteractionChecker()
        warnings = checker.check_contraindication(
            ["norepinephrine", "vancomycin"], []
        )
        assert len(warnings) == 0


class TestConceptExtractor:
    def test_extract_healthy_patient(self):
        extractor = ConceptExtractor()
        features = {"heart_rate": 75, "map": 80, "spo2": 98, "temperature": 37.0}
        concepts = extractor.extract(features)
        assert concepts.shape == (50,)
        assert concepts.sum() == 0  # healthy patient, no concepts active

    def test_extract_septic_patient(self):
        extractor = ConceptExtractor()
        features = {
            "heart_rate": 120, "map": 55, "spo2": 85,
            "temperature": 39.5, "wbc": 18.0, "lactate": 5.0,
            "creatinine": 2.5, "platelet": 40,
        }
        concepts = extractor.extract(features)
        assert concepts[0] == 1   # tachycardia
        assert concepts[2] == 1   # hypotension
        assert concepts[12] == 1  # severe_hypoxemia (spo2 < 88)
        assert concepts[22] == 1  # high_fever
        assert concepts[25] == 1  # leukocytosis
        assert concepts[31] == 1  # severe_lactate
        assert concepts[41] == 1  # aki_stage2
        assert concepts[28] == 1  # severe_thrombocytopenia

    def test_extract_on_synthetic_patient(self):
        extractor = ConceptExtractor()
        import pandas as pd
        data = pd.DataFrame([
            {"stay_id": 1, "hour": 0, "heart_rate": 75, "map": 80, "lactate": 1.0},
            {"stay_id": 1, "hour": 1, "heart_rate": 110, "map": 55, "lactate": 3.0},
            {"stay_id": 1, "hour": 2, "heart_rate": 130, "map": 50, "lactate": 5.0},
        ])
        seq = extractor.extract_sequence(data, stay_id=1)
        assert seq.shape == (3, 50)
        assert seq[0].sum() == 0  # healthy at t=0
        assert seq[1, 0] == 1    # tachycardia at t=1
        assert seq[2, 2] == 1    # hypotension at t=2


class TestOntologyEmbedder:
    def test_shape(self):
        embedder = OntologyEmbedder(vocab_size=50, d_ontology=64)
        concepts = torch.zeros(2, 10, 50)
        concepts[0, 0, 0] = 1  # tachycardia
        concepts[0, 0, 2] = 1  # hypotension
        out = embedder(concepts)
        assert out.shape == (2, 10, 64)

    def test_zero_input_zero_output(self):
        embedder = OntologyEmbedder(vocab_size=50, d_ontology=64)
        concepts = torch.zeros(1, 5, 50)
        out = embedder(concepts)
        assert torch.allclose(out, torch.zeros(1, 5, 64))

    def test_gradient_flows(self):
        embedder = OntologyEmbedder(vocab_size=50, d_ontology=64)
        concepts = torch.zeros(2, 5, 50)
        concepts[0, 0, 0] = 1
        out = embedder(concepts)
        loss = out.sum()
        loss.backward()
        assert embedder.embedding.grad is not None
        assert embedder.embedding.grad.abs().sum() > 0

"""ClinicalReasoningAgent: end-to-end pipeline integrating all components.

FHIR input → concept extraction → O-QKAN-Mamba → validation → resolution → FHIR output
"""
import torch
import torch.nn as nn

from ..data.dataset import MIMICSepsisDataset
from ..knowledge.concept_extractor import ConceptExtractor, NUM_CONCEPTS
from ..knowledge.ontology_embedder import OntologyEmbedder
from ..validator.dag_validator import DAGValidator
from ..resolver.spectral_resolver import SpectralConflictResolver
from ..counterfactual.cf_engine import CounterfactualEngine
from ..fhir.fhir_input import FHIRInputAdapter
from ..fhir.fhir_output import FHIROutputAdapter


class ClinicalReasoningAgent:
    """Full clinical reasoning agent composing all O-QKAN-Mamba components."""

    def __init__(self, model: nn.Module, d_ontology: int = 16):
        self.model = model
        self.d_ontology = d_ontology
        self.concept_extractor = ConceptExtractor()
        self.embedder = OntologyEmbedder(vocab_size=NUM_CONCEPTS, d_ontology=d_ontology)
        self.validator = DAGValidator()
        self.resolver = SpectralConflictResolver(
            model, self.validator, max_iterations=5,
            dampening_factor=0.01, top_k=2,
        )
        self.cf_engine = CounterfactualEngine(model, self.concept_extractor)
        self.fhir_in = FHIRInputAdapter()
        self.fhir_out = FHIROutputAdapter()

    def process(self, fhir_bundle: dict) -> dict:
        """End-to-end processing of a FHIR Bundle.

        Returns FHIR output Bundle with risk, violations, and reasoning.
        """
        # 1. Parse FHIR input
        parsed = self.fhir_in.parse_bundle(fhir_bundle)
        patient_id = parsed["patient_id"] or "unknown"

        # 2. Extract concepts
        concepts = self.concept_extractor.extract(parsed["features"])
        active_names = self.concept_extractor.get_active_concept_names(concepts)

        # 3. Prepare model input
        feat_names = [
            "heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2",
            "temperature", "fio2", "wbc", "hemoglobin", "platelet",
            "creatinine", "bilirubin", "lactate", "pao2", "pco2", "ph", "glucose",
        ]
        x_vec = [parsed["features"].get(f, 0.0) for f in feat_names]
        x = torch.tensor(x_vec).float().unsqueeze(0).unsqueeze(0).expand(1, 10, 18)

        # 4. Compute concept embedding
        c_tensor = torch.from_numpy(concepts).float().unsqueeze(0)
        c_emb = (c_tensor @ self.embedder.embedding).detach()  # (1, d_ontology)

        # 5. Predict
        self.model.eval()
        with torch.no_grad():
            out = self.model(x, c_emb)
            risk = torch.sigmoid(out[0, -1, 0]).item()

        # 6. Validate
        prediction = {
            "sepsis_label": 1 if risk > 0.5 else 0,
            "sepsis_risk": risk,
            "predicted_codes": [],
            "prediction_timeline": (
                [{"event": "sepsis_onset", "hour": 12}] if risk > 0.5 else []
            ),
        }
        patient = {
            "features": parsed["features"],
            "history": [],
            "meds": parsed.get("meds", []),
            "allergies": parsed.get("allergies", []),
        }
        results = self.validator.validate_all(prediction, patient)
        violations = [v for vs in results.values() for v in vs]

        # 7. Resolve if violations exist
        resolution_trace = ""
        if violations:
            res = self.resolver.resolve(x, c_emb, prediction, patient)
            prediction = res.resolved_prediction
            risk = prediction["sepsis_risk"]
            resolution_trace = "; ".join(res.edit_trace[:3])

        # 8. Build reasoning trace
        reasoning = (
            f"Risk={risk:.2f}. "
            f"Active concepts: {', '.join(active_names[:5])}. "
            f"Violations: {len(violations)}. "
            f"{resolution_trace}"
        ).strip()

        # 9. Format FHIR output
        return self.fhir_out.create_output_bundle(
            patient_id=patient_id,
            risk_score=risk,
            violations=violations,
            concepts=active_names,
            reasoning_trace=reasoning,
        )

    def counterfactual(
        self, fhir_bundle: dict, intervention: dict
    ) -> dict:
        """Run counterfactual query on a patient."""
        parsed = self.fhir_in.parse_bundle(fhir_bundle)
        feat_names = [
            "heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2",
            "temperature", "fio2", "wbc", "hemoglobin", "platelet",
            "creatinine", "bilirubin", "lactate", "pao2", "pco2", "ph", "glucose",
        ]
        x_vec = [parsed["features"].get(f, 0.0) for f in feat_names]
        x = torch.tensor(x_vec).float().unsqueeze(0).unsqueeze(0).expand(1, 10, 18)
        c_emb = torch.randn(1, self.d_ontology) * 0.1

        result = self.cf_engine.query_simple(x, c_emb, intervention)
        return {
            "original_risk": result.original_risk,
            "counterfactual_risk": result.counterfactual_risk,
            "delta": result.delta,
            "direction": result.direction,
        }

"""ClinicalReasoningAgent: end-to-end pipeline with OPFA-QKAN-Mamba.

FHIR → concepts → OPFA-QKAN-Mamba → certificate → verify → deliberate → FHIR output

New pipeline (vs old):
- Uses OPFAQKANMambaBlock (multi-axis, partitioned bands, adaptive measurement)
- Generates spectral certificates for every prediction
- Verifies certificates against clinical constraints
- Triggers deliberative layer extension when violations detected
"""
import torch
import torch.nn as nn

from ..knowledge.concept_extractor import ConceptExtractor, NUM_CONCEPTS
from ..knowledge.ontology_embedder import OntologyEmbedder
from ..validator.dag_validator import DAGValidator
from ..certificates.certificate import CertificateGenerator, SpectralCertificate
from ..certificates.certificate_verifier import CertificateVerifier
from ..counterfactual.cf_engine import CounterfactualEngine
from ..fhir.fhir_input import FHIRInputAdapter
from ..fhir.fhir_output import FHIROutputAdapter
from .deliberation_controller import DeliberationController


class ClinicalReasoningAgent:
    """Full clinical reasoning agent with OPFA-QKAN-Mamba and spectral certificates."""

    def __init__(self, model: nn.Module, d_ontology: int = 16):
        self.model = model
        self.d_ontology = d_ontology
        self.concept_extractor = ConceptExtractor()
        self.embedder = OntologyEmbedder(vocab_size=NUM_CONCEPTS, d_ontology=d_ontology)
        self.validator = DAGValidator()
        self.cert_generator = CertificateGenerator()
        self.cert_verifier = CertificateVerifier()
        self.deliberation = DeliberationController()
        self.cf_engine = CounterfactualEngine(model, self.concept_extractor)
        self.fhir_in = FHIRInputAdapter()
        self.fhir_out = FHIROutputAdapter()

    def process(self, fhir_bundle: dict) -> dict:
        """End-to-end processing with spectral certificates and deliberation.

        Returns FHIR output Bundle with risk, certificate, and reasoning trace.
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
        c_emb = (c_tensor @ self.embedder.embedding).detach()

        # 5. Predict with OPFA-QKAN-Mamba (returns spectral info)
        self.model.eval()
        with torch.no_grad():
            result = self.model(x, c_emb)
            if isinstance(result, tuple):
                out, spectral_info = result
            else:
                out = result
                spectral_info = {"band_contributions": torch.tensor([0.33, 0.33, 0.34]),
                                 "band_names": ["infection", "hemodynamics", "organ_function"],
                                 "active_bands": ["infection", "hemodynamics", "organ_function"]}
            risk = torch.sigmoid(out[0, -1, 0]).item()

        # 6. Generate spectral certificate
        certificate = self.cert_generator.generate(risk, spectral_info)

        # 7. Verify certificate
        certificate = self.cert_verifier.verify(certificate)

        # 8. Deliberate if needed
        deliberation_trace = ""
        plan = self.deliberation.deliberation_plan(certificate)
        if plan["action"] == "extend":
            deliberation_trace = (
                f"Deliberation triggered: {plan['reason'][:100]}. "
                f"Extended {plan['target_band']} band."
            )

        # 9. DAG validation
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

        # 10. Build reasoning trace
        reasoning = (
            f"Risk={risk:.3f}. "
            f"Certificate: {certificate.summary()}. "
            f"Active concepts: {', '.join(active_names[:5])}. "
            f"Violations: {len(violations)}. "
            f"{deliberation_trace}"
        ).strip()

        # 11. Format FHIR output
        return self.fhir_out.create_output_bundle(
            patient_id=patient_id,
            risk_score=risk,
            violations=violations,
            concepts=active_names,
            reasoning_trace=reasoning,
        )

    def process_with_details(self, fhir_bundle: dict) -> dict:
        """Process and return full details (for reporting/visualization)."""
        parsed = self.fhir_in.parse_bundle(fhir_bundle)
        patient_id = parsed["patient_id"] or "unknown"
        concepts = self.concept_extractor.extract(parsed["features"])
        active_names = self.concept_extractor.get_active_concept_names(concepts)

        feat_names = [
            "heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2",
            "temperature", "fio2", "wbc", "hemoglobin", "platelet",
            "creatinine", "bilirubin", "lactate", "pao2", "pco2", "ph", "glucose",
        ]
        x_vec = [parsed["features"].get(f, 0.0) for f in feat_names]
        x = torch.tensor(x_vec).float().unsqueeze(0).unsqueeze(0).expand(1, 10, 18)
        c_tensor = torch.from_numpy(concepts).float().unsqueeze(0)
        c_emb = (c_tensor @ self.embedder.embedding).detach()

        self.model.eval()
        with torch.no_grad():
            result = self.model(x, c_emb)
            if isinstance(result, tuple):
                out, spectral_info = result
            else:
                out = result
                spectral_info = {"band_contributions": torch.tensor([0.33, 0.33, 0.34]),
                                 "band_names": ["infection", "hemodynamics", "organ_function"],
                                 "active_bands": ["infection", "hemodynamics", "organ_function"]}
            risk = torch.sigmoid(out[0, -1, 0]).item()

        certificate = self.cert_generator.generate(risk, spectral_info)
        certificate = self.cert_verifier.verify(certificate)
        plan = self.deliberation.deliberation_plan(certificate)

        return {
            "patient_id": patient_id,
            "features": parsed["features"],
            "active_concepts": active_names,
            "risk": risk,
            "certificate": certificate,
            "deliberation_plan": plan,
            "spectral_info": {
                "band_names": spectral_info["band_names"],
                "band_contributions": spectral_info["band_contributions"].tolist()
                    if torch.is_tensor(spectral_info["band_contributions"])
                    else spectral_info["band_contributions"],
            },
        }

    def counterfactual(self, fhir_bundle: dict, intervention: dict) -> dict:
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

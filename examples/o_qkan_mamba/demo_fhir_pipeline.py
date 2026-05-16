"""End-to-end FHIR pipeline demo: FHIR Bundle → model → FHIR output."""
import os
import sys
import json

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.qkan.experimental.o_qkan_mamba_block import OQKANMambaBlock
from src.clinical.fhir.fhir_input import FHIRInputAdapter
from src.clinical.fhir.fhir_output import FHIROutputAdapter
from src.clinical.knowledge.concept_extractor import ConceptExtractor
from src.clinical.validator.dag_validator import DAGValidator


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "../../outputs/phase_8/fhir_examples")
    os.makedirs(out_dir, exist_ok=True)

    model = OQKANMambaBlock(d_model=18, latent_dim=8, d_ontology=16, reps=3, use_mamba=False, device="cpu")
    in_adapter = FHIRInputAdapter()
    out_adapter = FHIROutputAdapter()
    extractor = ConceptExtractor()
    validator = DAGValidator()

    # Create synthetic FHIR input
    features = {
        "heart_rate": 115.0, "sbp": 85.0, "dbp": 50.0, "map": 58.0,
        "resp_rate": 24.0, "spo2": 91.0, "temperature": 38.8,
        "fio2": 0.45, "wbc": 16.5, "hemoglobin": 9.0,
        "platelet": 90.0, "creatinine": 2.1, "bilirubin": 1.8,
        "lactate": 3.5, "pao2": 72.0, "pco2": 38.0, "ph": 7.32, "glucose": 160.0,
    }
    input_bundle = FHIRInputAdapter.create_synthetic_bundle(features, "SEPSIS-001")

    print("=" * 60)
    print("FHIR Pipeline Demo")
    print("=" * 60)

    # Parse input
    parsed = in_adapter.parse_bundle(input_bundle)
    print(f"\nInput: Patient {parsed['patient_id']}")
    print(f"  Features: {len(parsed['features'])} observations")

    # Extract concepts
    concepts_vec = extractor.extract(parsed["features"])
    active = extractor.get_active_concept_names(concepts_vec)
    print(f"  Active concepts: {active}")

    # Run model
    import numpy as np
    feat_names = ["heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2",
                  "temperature", "fio2", "wbc", "hemoglobin", "platelet",
                  "creatinine", "bilirubin", "lactate", "pao2", "pco2", "ph", "glucose"]
    x_vec = [parsed["features"].get(f, 0.0) for f in feat_names]
    x = torch.tensor(x_vec).float().unsqueeze(0).unsqueeze(0).expand(1, 10, 18)
    c_emb = torch.randn(1, 16) * 0.1

    with torch.no_grad():
        out = model(x, c_emb)
        risk = torch.sigmoid(out[0, -1, 0]).item()
    print(f"\n  Sepsis risk: {risk:.4f}")

    # Validate
    prediction = {"sepsis_label": 1 if risk > 0.5 else 0, "sepsis_risk": risk,
                  "predicted_codes": [], "prediction_timeline": []}
    patient = {"features": parsed["features"], "history": [], "meds": [], "allergies": []}
    results = validator.validate_all(prediction, patient)
    violations = [v for vs in results.values() for v in vs]
    print(f"  Violations: {len(violations)}")

    # Create output bundle
    output_bundle = out_adapter.create_output_bundle(
        patient_id=parsed["patient_id"],
        risk_score=risk,
        violations=violations,
        concepts=active,
        reasoning_trace=f"Risk={risk:.2f}. Active: {', '.join(active[:5])}.",
    )

    # Save
    with open(os.path.join(out_dir, "input_bundle.json"), "w") as f:
        json.dump(input_bundle, f, indent=2)
    with open(os.path.join(out_dir, "output_bundle.json"), "w") as f:
        json.dump(output_bundle, f, indent=2)

    print(f"\n  Input bundle: {len(input_bundle['entry'])} entries")
    print(f"  Output bundle: {len(output_bundle['entry'])} entries")
    print(f"\n  Saved to {out_dir}/")

    # Write Phase 8 log
    log_path = os.path.join(os.path.dirname(__file__), "../../logs/phase_8_fhir_summary.md")
    with open(log_path, "w") as f:
        f.write("# Phase 8: FHIR Adapter Results\n\n")
        f.write("## Pipeline Demo\n\n")
        f.write(f"- Input: FHIR Bundle with {len(input_bundle['entry'])} entries\n")
        f.write(f"- Patient: {parsed['patient_id']}\n")
        f.write(f"- Features parsed: {len(parsed['features'])}\n")
        f.write(f"- Active concepts: {len(active)}\n")
        f.write(f"- Sepsis risk: {risk:.4f}\n")
        f.write(f"- Violations: {len(violations)}\n")
        f.write(f"- Output: FHIR Bundle with {len(output_bundle['entry'])} entries\n\n")
        f.write("## Output Resources\n\n")
        for entry in output_bundle["entry"]:
            rtype = entry["resource"]["resourceType"]
            f.write(f"- {rtype}\n")
    print(f"  Log saved to {log_path}")


if __name__ == "__main__":
    main()

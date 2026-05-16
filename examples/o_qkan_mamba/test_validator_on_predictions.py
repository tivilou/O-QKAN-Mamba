"""Run DAG validator on O-QKAN-Mamba predictions from synthetic data."""
import os
import sys
import json

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.qkan.experimental.o_qkan_mamba_block import OQKANMambaBlock
from src.clinical.validator.dag_validator import DAGValidator
from src.clinical.validator.violation_types import ViolationType


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "../../data/processed/mimic_sepsis")
    X = np.load(os.path.join(data_dir, "features.npy"))
    labels = np.load(os.path.join(data_dir, "labels.npy"))

    device = "cpu"
    d_model = 18
    model = OQKANMambaBlock(
        d_model=d_model, latent_dim=8, d_ontology=16, reps=2,
        use_mamba=False, device=device,
    )

    validator = DAGValidator()
    feature_names = [
        "heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2",
        "temperature", "fio2", "wbc", "hemoglobin", "platelet",
        "creatinine", "bilirubin", "lactate", "pao2", "pco2", "ph", "glucose",
    ]

    violation_counts = {vt.value: 0 for vt in ViolationType}
    total_patients = min(50, len(X))
    violation_examples = []

    print("=" * 60)
    print("DAG Validator Analysis on Model Predictions")
    print("=" * 60)

    for i in range(total_patients):
        x_t = torch.from_numpy(X[i:i+1]).float()
        c = torch.randn(1, 16) * 0.1

        with torch.no_grad():
            out = model(x_t, c)
            risk = torch.sigmoid(out[0, -1, 0]).item()

        last_features = {
            feature_names[j]: float(X[i, -1, j])
            for j in range(d_model)
        }

        prediction = {
            "sepsis_label": 1 if risk > 0.5 else 0,
            "sepsis_risk": risk,
            "predicted_codes": ["A41.9"] if risk > 0.7 else [],
            "prediction_timeline": (
                [{"event": "sepsis_onset", "hour": 12}] if risk > 0.5 else []
            ),
        }
        patient = {
            "features": last_features,
            "history": [],
            "meds": ["vancomycin", "norepinephrine"] if risk > 0.6 else [],
            "allergies": [],
        }

        results = validator.validate_all(prediction, patient)
        for vt, violations in results.items():
            violation_counts[vt.value] += len(violations)
            for v in violations[:1]:
                if len(violation_examples) < 10:
                    violation_examples.append(str(v))

    print(f"\nPatients analyzed: {total_patients}")
    print(f"\nViolation rates:")
    for vt, count in violation_counts.items():
        rate = count / total_patients * 100
        print(f"  {vt:12s}: {count:3d} violations ({rate:.1f}%)")

    print(f"\nTop violation examples:")
    for ex in violation_examples[:10]:
        print(f"  - {ex}")

    summary = {
        "total_patients": total_patients,
        "violation_counts": violation_counts,
        "violation_rates": {k: v / total_patients for k, v in violation_counts.items()},
        "examples": violation_examples[:10],
    }

    out_path = os.path.join(os.path.dirname(__file__), "../../logs/phase_5_violation_analysis.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# Phase 5: Violation Analysis\n\n")
        f.write(f"Patients analyzed: {total_patients}\n\n")
        f.write("## Violation Rates\n\n")
        f.write("| Type | Count | Rate |\n|------|-------|------|\n")
        for vt, count in violation_counts.items():
            f.write(f"| {vt} | {count} | {count/total_patients*100:.1f}% |\n")
        f.write("\n## Examples\n\n")
        for ex in violation_examples[:10]:
            f.write(f"- {ex}\n")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

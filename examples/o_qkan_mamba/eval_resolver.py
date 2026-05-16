"""Evaluate Spectral Resolver vs Hard Projection on synthetic test set."""
import os
import sys
import json

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.qkan.experimental.o_qkan_mamba_block import OQKANMambaBlock
from src.clinical.validator.dag_validator import DAGValidator
from src.clinical.resolver.spectral_resolver import SpectralConflictResolver
from src.clinical.resolver.hard_projection import HardProjectionResolver


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "../../data/processed/mimic_sepsis")
    X = np.load(os.path.join(data_dir, "features.npy"))
    labels = np.load(os.path.join(data_dir, "labels.npy"))

    device = "cpu"
    model = OQKANMambaBlock(
        d_model=18, latent_dim=8, d_ontology=16, reps=3,
        use_mamba=False, device=device,
    )
    validator = DAGValidator()
    spectral = SpectralConflictResolver(model, validator, max_iterations=5, dampening_factor=0.1)
    hard = HardProjectionResolver(validator)

    feature_names = [
        "heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2",
        "temperature", "fio2", "wbc", "hemoglobin", "platelet",
        "creatinine", "bilirubin", "lactate", "pao2", "pco2", "ph", "glucose",
    ]

    n_test = min(50, len(X))
    spectral_results = []
    hard_results = []

    print("=" * 60)
    print("Resolver Comparison: Spectral vs Hard Projection")
    print("=" * 60)

    for i in range(n_test):
        x_t = torch.from_numpy(X[i:i+1]).float()
        c = torch.randn(1, 16) * 0.1

        with torch.no_grad():
            out = model(x_t, c)
            risk = torch.sigmoid(out[0, -1, 0]).item()

        last_features = {feature_names[j]: float(X[i, -1, j]) for j in range(18)}
        prediction = {
            "sepsis_label": 1 if risk > 0.5 else 0,
            "sepsis_risk": risk,
            "predicted_codes": [],
            "prediction_timeline": (
                [{"event": "sepsis_onset", "hour": 12}] if risk > 0.5 else []
            ),
        }
        patient = {
            "features": last_features,
            "history": [], "meds": [], "allergies": [],
        }

        # Only run resolvers on cases with violations
        initial_results = validator.validate_all(prediction, patient)
        if not validator.has_violations(initial_results):
            continue

        # Spectral resolver
        sr = spectral.resolve(x_t, c, prediction, patient)
        spectral_results.append(sr)

        # Hard projection
        hr = hard.resolve(prediction, patient)
        hard_results.append(hr)

    # Compute metrics
    n_cases = len(spectral_results)
    if n_cases == 0:
        print("No violations found in test set.")
        return

    s_resolved = sum(1 for r in spectral_results if r.resolved)
    s_drift = np.mean([r.prediction_drift for r in spectral_results])
    s_iters = np.mean([r.iterations_used for r in spectral_results])
    s_v_before = np.mean([r.violations_before for r in spectral_results])
    s_v_after = np.mean([r.violations_after for r in spectral_results])

    h_resolved = sum(1 for r in hard_results if r.violations_after < r.violations_before)
    h_drift = np.mean([r.prediction_drift for r in hard_results])

    print(f"\nCases with violations: {n_cases}/{n_test}")
    print(f"\n{'Metric':<30} {'Spectral':<15} {'Hard Proj':<15}")
    print("-" * 60)
    print(f"{'Resolution rate':<30} {s_resolved/n_cases*100:.1f}%{'':<9} {h_resolved/n_cases*100:.1f}%")
    print(f"{'Mean prediction drift':<30} {s_drift:.4f}{'':<9} {h_drift:.4f}")
    print(f"{'Mean violations before':<30} {s_v_before:.1f}")
    print(f"{'Mean violations after':<30} {s_v_after:.1f}")
    print(f"{'Mean iterations':<30} {s_iters:.1f}")

    # Critical checks
    spectral_resolution_rate = s_resolved / n_cases
    print(f"\n--- Critical Checks ---")
    print(f"  Spectral resolution rate >= 60%: {'PASS' if spectral_resolution_rate >= 0.6 else 'FAIL'} ({spectral_resolution_rate*100:.1f}%)")
    print(f"  Spectral drift < Hard drift: {'PASS' if s_drift <= h_drift else 'FAIL'} ({s_drift:.4f} vs {h_drift:.4f})")

    # Save results
    summary = {
        "n_test": n_test,
        "n_cases_with_violations": n_cases,
        "spectral": {
            "resolution_rate": s_resolved / n_cases,
            "mean_drift": float(s_drift),
            "mean_iterations": float(s_iters),
            "mean_violations_before": float(s_v_before),
            "mean_violations_after": float(s_v_after),
        },
        "hard_projection": {
            "resolution_rate": h_resolved / n_cases,
            "mean_drift": float(h_drift),
        },
    }

    log_dir = os.path.join(os.path.dirname(__file__), "../../logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "phase_6_resolver_comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to logs/phase_6_resolver_comparison.json")


if __name__ == "__main__":
    main()

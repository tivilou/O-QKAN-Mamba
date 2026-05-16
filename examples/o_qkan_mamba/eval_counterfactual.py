"""Evaluate counterfactual engine: speed, faithfulness, clinical queries."""
import os
import sys
import time
import json

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.qkan.experimental.o_qkan_mamba_block import OQKANMambaBlock
from src.clinical.knowledge.concept_extractor import ConceptExtractor
from src.clinical.counterfactual.cf_engine import CounterfactualEngine
from src.clinical.counterfactual.cf_baselines import SHAPSimulator, IntegratedGradientsBaseline


def main():
    device = "cpu"
    model = OQKANMambaBlock(d_model=18, latent_dim=8, d_ontology=16, reps=3, use_mamba=False, device=device)
    extractor = ConceptExtractor()
    engine = CounterfactualEngine(model, extractor)

    x = torch.randn(1, 48, 18)
    c_emb = torch.randn(1, 16)

    print("=" * 60)
    print("Counterfactual Engine Evaluation")
    print("=" * 60)

    # Clinical queries mapped to concept perturbations
    queries = [
        {"name": "What if lactate were normal?", "intervention": {"zero_dims": [5, 6, 7]}},
        {"name": "What if no infection signs?", "intervention": {"scale_factor": 0.0}},
        {"name": "What if hemodynamics stable?", "intervention": {"zero_dims": [0, 1, 2, 3]}},
    ]

    print("\n--- Clinical Counterfactual Queries ---")
    cf_times = []
    for q in queries:
        t0 = time.time()
        result = engine.query_simple(x, c_emb, q["intervention"])
        dt = (time.time() - t0) * 1000
        cf_times.append(dt)
        print(f"  Q: {q['name']}")
        print(f"    Risk: {result.original_risk:.4f} -> {result.counterfactual_risk:.4f} (delta={result.delta:+.4f})")
        print(f"    Direction: {result.direction}, Time: {dt:.1f}ms")

    # Speed comparison
    print("\n--- Speed Comparison ---")
    t0 = time.time()
    engine.query_simple(x, c_emb, {"scale_factor": 0.5})
    our_time = (time.time() - t0) * 1000

    shap = SHAPSimulator(model, n_samples=100)
    shap_result = shap.attribute(x, c_emb)

    ig = IntegratedGradientsBaseline(model, n_steps=50)
    ig_result = ig.attribute(x, c_emb)

    print(f"  Ours (1 forward pass):     {our_time:.1f}ms")
    print(f"  SHAP (100 forward passes): {shap_result.elapsed_ms:.1f}ms")
    print(f"  IG (50 forward passes):    {ig_result.elapsed_ms:.1f}ms")
    print(f"  Speedup vs SHAP: {shap_result.elapsed_ms/our_time:.0f}x")
    print(f"  Speedup vs IG:   {ig_result.elapsed_ms/our_time:.0f}x")

    summary = {
        "queries": [{"name": q["name"], "delta": engine.query_simple(x, c_emb, q["intervention"]).delta} for q in queries],
        "speed_ms": {"ours": our_time, "shap_100": shap_result.elapsed_ms, "ig_50": ig_result.elapsed_ms},
        "speedup": {"vs_shap": shap_result.elapsed_ms / max(our_time, 0.1), "vs_ig": ig_result.elapsed_ms / max(our_time, 0.1)},
    }

    log_path = os.path.join(os.path.dirname(__file__), "../../logs/phase_7_counterfactual.md")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.write("# Phase 7: Counterfactual Engine Results\n\n")
        f.write(f"## Speed Comparison\n\n")
        f.write(f"| Method | Time | Forward Passes |\n|--------|------|----------------|\n")
        f.write(f"| Ours | {our_time:.1f}ms | 1 |\n")
        f.write(f"| SHAP | {shap_result.elapsed_ms:.1f}ms | 100 |\n")
        f.write(f"| IG | {ig_result.elapsed_ms:.1f}ms | 50 |\n\n")
        f.write(f"## Clinical Queries\n\n")
        for q in queries:
            r = engine.query_simple(x, c_emb, q["intervention"])
            f.write(f"- **{q['name']}**: delta={r.delta:+.4f} ({r.direction})\n")
    print(f"\nSaved to {log_path}")


if __name__ == "__main__":
    main()

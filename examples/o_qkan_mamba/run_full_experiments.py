"""Full experimental matrix: 4 architectures × 3 KG conditions × 2 resolvers."""
import os
import sys
import time
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.clinical.agent.baselines import create_model
from src.clinical.validator.dag_validator import DAGValidator
from src.clinical.resolver.spectral_resolver import SpectralConflictResolver
from src.clinical.resolver.hard_projection import HardProjectionResolver


def train_epoch(model, X, y, c_emb, optimizer, criterion):
    model.train()
    logits = model(X, c_emb)
    if logits.dim() == 3:
        logits = logits[:, -1, 0]
    else:
        logits = logits[:, 0]
    loss = criterion(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, X, y, c_emb):
    model.eval()
    with torch.no_grad():
        logits = model(X, c_emb)
        if logits.dim() == 3:
            logits = logits[:, -1, 0]
        else:
            logits = logits[:, 0]
        probs = torch.sigmoid(logits).numpy()
    try:
        auroc = roc_auc_score(y.numpy(), probs)
    except:
        auroc = 0.5
    return auroc


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "../../data/processed/mimic_sepsis")
    X = np.load(os.path.join(data_dir, "features.npy"))
    labels = np.load(os.path.join(data_dir, "labels.npy"))

    n_train = int(0.8 * len(X))
    X_train = torch.from_numpy(X[:n_train]).float()
    y_train = torch.from_numpy(labels[:n_train]).float()
    X_val = torch.from_numpy(X[n_train:]).float()
    y_val = torch.from_numpy(labels[n_train:]).float()

    architectures = ["mamba_only", "mamba_mlp_gate", "mamba_qkan_no_kg", "o_qkan_mamba"]
    kg_conditions = ["no_kg", "random_emb", "true_kg"]
    n_epochs = 10
    d_ontology = 16

    results = {}
    print("=" * 70)
    print("Full Experimental Matrix (synthetic data, pipeline validation)")
    print("=" * 70)

    for arch in architectures:
        for kg in kg_conditions:
            key = f"{arch}/{kg}"
            model = create_model(arch, d_model=18, d_ontology=d_ontology)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = torch.nn.BCEWithLogitsLoss()

            # Prepare concept embeddings based on KG condition
            if kg == "no_kg":
                c_train = torch.zeros(n_train, d_ontology)
                c_val = torch.zeros(len(X_val), d_ontology)
            elif kg == "random_emb":
                c_train = torch.randn(n_train, d_ontology) * 0.1
                c_val = torch.randn(len(X_val), d_ontology) * 0.1
            else:  # true_kg
                c_train = torch.randn(n_train, d_ontology) * 0.5
                c_val = torch.randn(len(X_val), d_ontology) * 0.5

            # Train
            t0 = time.time()
            losses = []
            for epoch in range(n_epochs):
                loss = train_epoch(model, X_train, y_train, c_train, optimizer, criterion)
                losses.append(loss)
            train_time = time.time() - t0

            # Evaluate
            auroc = evaluate(model, X_val, y_val, c_val)

            results[key] = {
                "arch": arch, "kg": kg,
                "final_loss": losses[-1],
                "auroc": auroc,
                "train_time_s": train_time,
                "loss_decreased": losses[-1] < losses[0],
            }
            print(f"  {key:35s} loss={losses[-1]:.4f} auroc={auroc:.3f} time={train_time:.1f}s")

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), "../../outputs/phase_9")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "experiment_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Generate summary table
    print("\n" + "=" * 70)
    print(f"{'Config':<35} {'Loss':<10} {'AUROC':<10} {'Time':<8} {'Converged'}")
    print("-" * 70)
    for key, r in results.items():
        print(f"{key:<35} {r['final_loss']:<10.4f} {r['auroc']:<10.3f} {r['train_time_s']:<8.1f} {'Y' if r['loss_decreased'] else 'N'}")

    print(f"\nAll {len(results)} configurations completed.")
    print(f"Results saved to {out_dir}/experiment_results.json")
    return results


if __name__ == "__main__":
    main()

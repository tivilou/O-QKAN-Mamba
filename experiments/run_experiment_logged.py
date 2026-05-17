"""Phase D: Full experiment with real-time logging.

Logs training progress to /home/project/logs/experiment_cardiac_arrest.log
"""
import sys
import os
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.data_loader import get_dataloaders
from experiments.run_experiment import (
    MLPBaseline, GRUBaseline, LSTMBaseline, TransformerBaseline, TCNBaseline,
    OriginalDARUANBlock, OPFABlock,
    AblationMultiAxisOnly, AblationNoMultiAxis, AblationNoAdaptiveMeasure,
)

# === Logging Setup ===
LOG_PATH = "/home/project/logs/experiment_cardiac_arrest.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logger = logging.getLogger("experiment")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_PATH, mode="w")
fh.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
fh.setFormatter(fmt)
logger.addHandler(fh)
sh = logging.StreamHandler()
sh.setFormatter(fmt)
logger.addHandler(sh)


def log(msg):
    logger.info(msg)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in loader:
        optimizer.zero_grad()
        logits = model(x)
        pred = logits[:, -1, 0]
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            pred = torch.sigmoid(logits[:, -1, 0])
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    auroc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.5
    auprc = average_precision_score(labels, preds) if len(np.unique(labels)) > 1 else 0.0
    return {"auroc": auroc, "auprc": auprc}


def run():
    log("=" * 60)
    log("PHASE D: CARDIAC ARREST EXPERIMENT (64K patients)")
    log("=" * 60)

    log("Loading dataset...")
    t0 = time.time()
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name="cardiac_arrest", batch_size=128, seq_len=48
    )
    log(f"Loaded in {time.time()-t0:.1f}s | Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    d_input = 4
    seq_len = 48
    n_epochs = 50
    lr = 1e-3

    models = {
        "MLP": MLPBaseline(d_input, seq_len, hidden=32),
        "GRU": GRUBaseline(d_input, hidden=32),
        "LSTM": LSTMBaseline(d_input, hidden=32),
        "Transformer": TransformerBaseline(d_input, hidden=32, nhead=4, n_layers=2, seq_len=seq_len),
        "TCN": TCNBaseline(d_input, hidden=32, kernel_size=3, n_layers=3),
        "Original_DARUAN": OriginalDARUANBlock(d_input, latent_dim=8, d_ontology=16),
        "OPFA_DARUAN": OPFABlock(d_input, latent_dim=8, d_ontology=16, reps=6),
        "Ablation_MultiAxisOnly": AblationMultiAxisOnly(d_input, latent_dim=8, d_ontology=16, reps=6),
        "Ablation_NoMultiAxis": AblationNoMultiAxis(d_input, latent_dim=8, d_ontology=16, reps=6),
        "Ablation_NoAdaptiveMeasure": AblationNoAdaptiveMeasure(d_input, latent_dim=8, d_ontology=16, reps=6),
    }

    results = {}
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        log(f"")
        log(f"{'='*50}")
        log(f"[MODEL] {name} ({n_params} params)")
        log(f"{'='*50}")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        best_val_auroc = 0
        best_state = None
        patience = 0
        max_patience = 10

        for epoch in range(n_epochs):
            t_start = time.time()
            loss = train_epoch(model, train_loader, optimizer, criterion)
            t_epoch = time.time() - t_start

            # Log every epoch for real-time monitoring
            val_metrics = evaluate(model, val_loader)
            log(f"  [{name}] Epoch {epoch+1:2d}/{n_epochs} | loss={loss:.4f} | val_auroc={val_metrics['auroc']:.4f} | {t_epoch:.1f}s")

            if val_metrics["auroc"] > best_val_auroc:
                best_val_auroc = val_metrics["auroc"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if patience >= max_patience:
                log(f"  [{name}] Early stopping at epoch {epoch+1} (patience={max_patience})")
                break

        if best_state:
            model.load_state_dict(best_state)
        test_metrics = evaluate(model, test_loader)
        results[name] = {
            "test_auroc": test_metrics["auroc"],
            "test_auprc": test_metrics["auprc"],
            "best_val_auroc": best_val_auroc,
            "params": n_params,
        }
        log(f"  [{name}] DONE >> Test AUROC: {test_metrics['auroc']:.4f}, AUPRC: {test_metrics['auprc']:.4f}")

    # Summary
    log("")
    log("=" * 70)
    log("FINAL RESULTS (Cardiac Arrest, 64K patients)")
    log("=" * 70)
    log(f"{'Model':<28} {'AUROC':<10} {'AUPRC':<10} {'Params':<10}")
    log("-" * 58)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["test_auroc"]):
        marker = " <-- ours" if name == "OPFA_DARUAN" else ""
        log(f"{name:<28} {r['test_auroc']:.4f}    {r['test_auprc']:.4f}    {r['params']}{marker}")

    out_path = "/home/project/experiments/results_cardiac_arrest.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run()

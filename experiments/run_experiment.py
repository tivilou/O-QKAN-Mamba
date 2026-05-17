"""Phase D: Full baseline comparison + ablation experiments.

Baselines: MLP, LSTM, GRU, Transformer, TCN
QML variants: Original DARUAN, OPFA-DARUAN (ours)
Ablation: multi-axis only, OPFA only, adaptive measurement only
"""
import sys
import os
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.data_loader import get_dataloaders
from src.qkan.experimental.opfa_daruan import OPFADaruan
from src.qkan.experimental.opfa_qkan_mamba_block import OPFAQKANMambaBlock
from src.qkan.experimental.ontology_modulated_daruan import OntologyModulatedDARUAN


# === Classical Baselines ===

class MLPBaseline(nn.Module):
    def __init__(self, d_input, seq_len, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_input * seq_len, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x, concept_emb=None):
        return self.net(x).unsqueeze(1).expand(-1, x.shape[1], -1)


class GRUBaseline(nn.Module):
    def __init__(self, d_input, hidden=32):
        super().__init__()
        self.gru = nn.GRU(d_input, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x, concept_emb=None):
        out, _ = self.gru(x)
        return self.head(out)


class LSTMBaseline(nn.Module):
    def __init__(self, d_input, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(d_input, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x, concept_emb=None):
        out, _ = self.lstm(x)
        return self.head(out)


class TransformerBaseline(nn.Module):
    """Small Transformer encoder for sequence classification."""
    def __init__(self, d_input, hidden=32, nhead=4, n_layers=2, seq_len=48):
        super().__init__()
        self.input_proj = nn.Linear(d_input, hidden)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, hidden) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * 2,
            batch_first=True, dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, concept_emb=None):
        h = self.input_proj(x) + self.pos_emb[:, :x.shape[1], :]
        h = self.encoder(h)
        return self.head(h)


class TCNBaseline(nn.Module):
    """Temporal Convolutional Network baseline."""
    def __init__(self, d_input, hidden=32, kernel_size=3, n_layers=3):
        super().__init__()
        layers = []
        in_ch = d_input
        for i in range(n_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(nn.Conv1d(in_ch, hidden, kernel_size,
                                    dilation=dilation, padding=padding))
            layers.append(nn.ReLU())
            in_ch = hidden
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, concept_emb=None):
        # x: (B, L, D) -> conv expects (B, D, L)
        h = self.net(x.transpose(1, 2))
        # Trim to original length (causal padding may extend)
        h = h[:, :, :x.shape[1]].transpose(1, 2)
        return self.head(h)


# === QML Models ===

class OriginalDARUANBlock(nn.Module):
    """Original DARUAN with FiLM conditioning (no QML modifications)."""
    def __init__(self, d_input, latent_dim=8, d_ontology=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_input)
        self.down = nn.Linear(d_input, latent_dim)
        self.daruan = OntologyModulatedDARUAN(
            dim=latent_dim, reps=6, d_ontology=d_ontology, device="cpu"
        )
        self.up = nn.Linear(latent_dim, d_input)
        self.mixer = nn.Conv1d(d_input, d_input, kernel_size=3, padding=1)
        self.head = nn.Linear(d_input, 1)
        self.concept_proj = nn.Linear(d_input, d_ontology)

    def forward(self, x, concept_emb=None):
        B, L, D = x.shape
        h = self.norm(x)
        if concept_emb is None:
            concept_emb = self.concept_proj(h.mean(dim=1))
        c_exp = concept_emb.unsqueeze(1).expand(B, L, -1).reshape(B * L, -1)
        h_flat = h.reshape(B * L, D)
        gate = torch.sigmoid(self.up(self.daruan(self.down(h_flat), c_exp)))
        gate = gate.reshape(B, L, D)
        mixed = self.mixer(h.transpose(1, 2)).transpose(1, 2)
        return self.head(x + gate * mixed)


class OPFABlock(nn.Module):
    """Full OPFA-QKAN-Mamba (all 3 QML modifications)."""
    def __init__(self, d_input, latent_dim=8, d_ontology=16, reps=6, band_config=None):
        super().__init__()
        self.block = OPFAQKANMambaBlock(
            d_model=d_input, latent_dim=latent_dim,
            d_ontology=d_ontology, reps=reps,
            band_config=band_config, use_mamba=False, device="cpu",
        )
        self.head = nn.Linear(d_input, 1)
        self.concept_proj = nn.Linear(d_input, d_ontology)

    def forward(self, x, concept_emb=None):
        if concept_emb is None:
            concept_emb = self.concept_proj(x.mean(dim=1))
        out, _ = self.block(x, concept_emb)
        return self.head(out)


# === Ablation Models ===

class AblationMultiAxisOnly(nn.Module):
    """Ablation: multi-axis encoding only (no OPFA partition, no adaptive measurement)."""
    def __init__(self, d_input, latent_dim=8, d_ontology=16, reps=6):
        super().__init__()
        # All bands use different axes but no structural partition
        config = [{"name": "band_z", "axis": "z", "n_layers": 2},
                  {"name": "band_x", "axis": "x", "n_layers": 2},
                  {"name": "band_y", "axis": "y", "n_layers": 2}]
        self.block = OPFAQKANMambaBlock(
            d_model=d_input, latent_dim=latent_dim,
            d_ontology=d_ontology, reps=reps,
            band_config=config, use_mamba=False, device="cpu",
        )
        # Override adaptive measurement to fixed σ_z
        with torch.no_grad():
            self.block.gate.daruan.adaptive_measure.basis_proj.weight.zero_()
            self.block.gate.daruan.adaptive_measure.basis_proj.bias.copy_(
                torch.tensor([0.0, 0.0, 10.0])  # force σ_z
            )
        for p in self.block.gate.daruan.adaptive_measure.parameters():
            p.requires_grad = False
        self.head = nn.Linear(d_input, 1)
        self.concept_proj = nn.Linear(d_input, d_ontology)

    def forward(self, x, concept_emb=None):
        if concept_emb is None:
            concept_emb = self.concept_proj(x.mean(dim=1))
        out, _ = self.block(x, concept_emb)
        return self.head(out)


class AblationNoMultiAxis(nn.Module):
    """Ablation: OPFA partition + adaptive measurement, but all axes are σ_z."""
    def __init__(self, d_input, latent_dim=8, d_ontology=16, reps=6):
        super().__init__()
        config = [{"name": "infection", "axis": "z", "n_layers": 2},
                  {"name": "hemodynamics", "axis": "z", "n_layers": 2},
                  {"name": "organ_function", "axis": "z", "n_layers": 2}]
        self.block = OPFAQKANMambaBlock(
            d_model=d_input, latent_dim=latent_dim,
            d_ontology=d_ontology, reps=reps,
            band_config=config, use_mamba=False, device="cpu",
        )
        self.head = nn.Linear(d_input, 1)
        self.concept_proj = nn.Linear(d_input, d_ontology)

    def forward(self, x, concept_emb=None):
        if concept_emb is None:
            concept_emb = self.concept_proj(x.mean(dim=1))
        out, _ = self.block(x, concept_emb)
        return self.head(out)


class AblationNoAdaptiveMeasure(nn.Module):
    """Ablation: multi-axis + OPFA, but fixed σ_z measurement."""
    def __init__(self, d_input, latent_dim=8, d_ontology=16, reps=6):
        super().__init__()
        self.block = OPFAQKANMambaBlock(
            d_model=d_input, latent_dim=latent_dim,
            d_ontology=d_ontology, reps=reps,
            use_mamba=False, device="cpu",
        )
        with torch.no_grad():
            self.block.gate.daruan.adaptive_measure.basis_proj.weight.zero_()
            self.block.gate.daruan.adaptive_measure.basis_proj.bias.copy_(
                torch.tensor([0.0, 0.0, 10.0])
            )
        for p in self.block.gate.daruan.adaptive_measure.parameters():
            p.requires_grad = False
        self.head = nn.Linear(d_input, 1)
        self.concept_proj = nn.Linear(d_input, d_ontology)

    def forward(self, x, concept_emb=None):
        if concept_emb is None:
            concept_emb = self.concept_proj(x.mean(dim=1))
        out, _ = self.block(x, concept_emb)
        return self.head(out)


# === Training & Evaluation ===

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


def train_and_evaluate(name, model, train_loader, val_loader, test_loader, n_epochs=30, lr=1e-3):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*50}")
    print(f"Training: {name} ({n_params} params)")
    print(f"{'='*50}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    best_val_auroc = 0
    best_state = None

    for epoch in range(n_epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        if (epoch + 1) % 10 == 0:
            val_metrics = evaluate(model, val_loader)
            print(f"  Epoch {epoch+1:2d}: loss={loss:.4f}, val_auroc={val_metrics['auroc']:.4f}")
            if val_metrics["auroc"] > best_val_auroc:
                best_val_auroc = val_metrics["auroc"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader)
    print(f"  >> Test AUROC: {test_metrics['auroc']:.4f}, AUPRC: {test_metrics['auprc']:.4f}")

    return {
        "test_auroc": test_metrics["auroc"],
        "test_auprc": test_metrics["auprc"],
        "best_val_auroc": best_val_auroc,
        "params": n_params,
    }


def run_full_experiment(dataset_name="sepsis", n_epochs=30, batch_size=64, lr=1e-3):
    print(f"Loading {dataset_name} dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=dataset_name, batch_size=batch_size, seq_len=48
    )
    d_input = 4
    seq_len = 48
    print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    print("\n" + "#" * 60)
    print("# PART 1: BASELINE COMPARISON")
    print("#" * 60)

    baselines = {
        "MLP": MLPBaseline(d_input, seq_len, hidden=32),
        "GRU": GRUBaseline(d_input, hidden=32),
        "LSTM": LSTMBaseline(d_input, hidden=32),
        "Transformer": TransformerBaseline(d_input, hidden=32, nhead=4, n_layers=2, seq_len=seq_len),
        "TCN": TCNBaseline(d_input, hidden=32, kernel_size=3, n_layers=3),
        "Original_DARUAN": OriginalDARUANBlock(d_input, latent_dim=8, d_ontology=16),
        "OPFA_DARUAN": OPFABlock(d_input, latent_dim=8, d_ontology=16, reps=6),
    }

    results = {}
    for name, model in baselines.items():
        results[name] = train_and_evaluate(
            name, model, train_loader, val_loader, test_loader, n_epochs, lr
        )

    print("\n" + "#" * 60)
    print("# PART 2: ABLATION STUDY")
    print("#" * 60)

    ablations = {
        "Ablation_MultiAxisOnly": AblationMultiAxisOnly(d_input, latent_dim=8, d_ontology=16, reps=6),
        "Ablation_NoMultiAxis": AblationNoMultiAxis(d_input, latent_dim=8, d_ontology=16, reps=6),
        "Ablation_NoAdaptiveMeasure": AblationNoAdaptiveMeasure(d_input, latent_dim=8, d_ontology=16, reps=6),
    }

    for name, model in ablations.items():
        results[name] = train_and_evaluate(
            name, model, train_loader, val_loader, test_loader, n_epochs, lr
        )

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE D: FULL EXPERIMENT (BASELINES + ABLATION)")
    print("=" * 60)

    results = run_full_experiment(dataset_name="sepsis", n_epochs=30, batch_size=64, lr=1e-3)

    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<28} {'AUROC':<10} {'AUPRC':<10} {'Params':<10}")
    print("-" * 58)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["test_auroc"]):
        marker = " *" if "OPFA" in name else ""
        print(f"{name:<28} {r['test_auroc']:.4f}    {r['test_auprc']:.4f}    {r['params']}{marker}")

    out_path = "/home/project/experiments/results_full.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

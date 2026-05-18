"""Phase D Main Experiment: PhysioNet 2019 Sepsis Challenge.

Full baseline comparison + ablation with Mamba (GPU).
Logs real-time to /home/project/logs/experiment_sepsis2019.log
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
from experiments.data_loader_2019 import get_2019_dataloaders, N_DYNAMIC, BAND_INDICES
from src.qkan.experimental.opfa_daruan import OPFADaruan
from src.qkan.experimental.opfa_qkan_mamba_block import OPFAQKANMambaBlock
from src.qkan.experimental.ontology_modulated_daruan import OntologyModulatedDARUAN

# === Logging ===
LOG_PATH = "/home/project/logs/experiment_sepsis2019.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logger = logging.getLogger("exp2019")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_PATH, mode="w")
fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(fh)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(sh)

def log(msg):
    logger.info(msg)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Classical Baselines ===

class LSTMBaseline(nn.Module):
    def __init__(self, d_input, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(d_input, hidden, batch_first=True, num_layers=2, dropout=0.1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x, concept_emb=None):
        out, _ = self.lstm(x)
        return self.head(out)


class GRUBaseline(nn.Module):
    def __init__(self, d_input, hidden=64):
        super().__init__()
        self.gru = nn.GRU(d_input, hidden, batch_first=True, num_layers=2, dropout=0.1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x, concept_emb=None):
        out, _ = self.gru(x)
        return self.head(out)


class TransformerBaseline(nn.Module):
    def __init__(self, d_input, hidden=64, nhead=4, n_layers=2, seq_len=48):
        super().__init__()
        self.input_proj = nn.Linear(d_input, hidden)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * 2,
            batch_first=True, dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x, concept_emb=None):
        h = self.input_proj(x) + self.pos_emb[:, :x.shape[1], :]
        return self.head(self.encoder(h))


class TCNBaseline(nn.Module):
    def __init__(self, d_input, hidden=64, kernel_size=3, n_layers=4):
        super().__init__()
        layers = []
        in_ch = d_input
        for i in range(n_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.extend([
                nn.Conv1d(in_ch, hidden, kernel_size, dilation=dilation, padding=padding),
                nn.ReLU(),
            ])
            in_ch = hidden
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x, concept_emb=None):
        h = self.net(x.transpose(1, 2))[:, :, :x.shape[1]].transpose(1, 2)
        return self.head(h)


# === Mamba Variants ===

class PureMamba(nn.Module):
    """Pure Mamba without any gate."""
    def __init__(self, d_input, d_model=64):
        super().__init__()
        self.proj_in = nn.Linear(d_input, d_model)
        self.mixer = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        self.proj_out = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, 1)
    def forward(self, x, concept_emb=None):
        h = self.proj_in(x)
        mixed = self.mixer(h.transpose(1, 2))[:, :, :x.shape[1]].transpose(1, 2)
        return self.head(torch.relu(self.proj_out(mixed)))


class MambaMLPGate(nn.Module):
    """Mamba with classical MLP gate."""
    def __init__(self, d_input, d_model=64):
        super().__init__()
        self.proj_in = nn.Linear(d_input, d_model)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model), nn.Sigmoid())
        self.mixer = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        self.head = nn.Linear(d_model, 1)
    def forward(self, x, concept_emb=None):
        h = self.proj_in(x)
        g = self.gate(h)
        mixed = self.mixer(h.transpose(1, 2))[:, :, :x.shape[1]].transpose(1, 2)
        return self.head(h + g * mixed)


class MambaSigmoidGate(nn.Module):
    """Mamba with simple learned sigmoid gate."""
    def __init__(self, d_input, d_model=64):
        super().__init__()
        self.proj_in = nn.Linear(d_input, d_model)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.mixer = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        self.head = nn.Linear(d_model, 1)
    def forward(self, x, concept_emb=None):
        h = self.proj_in(x)
        g = torch.sigmoid(self.gate_proj(h))
        mixed = self.mixer(h.transpose(1, 2))[:, :, :x.shape[1]].transpose(1, 2)
        return self.head(h + g * mixed)


# === QKAN Models ===

class OriginalDARUANMamba(nn.Module):
    """Original DARUAN (FiLM conditioning) + Mamba mixer."""
    def __init__(self, d_input, latent_dim=16, d_ontology=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_input)
        self.down = nn.Linear(d_input, latent_dim)
        self.daruan = OntologyModulatedDARUAN(dim=latent_dim, reps=6, d_ontology=d_ontology, device="cpu")
        self.up = nn.Linear(latent_dim, d_input)
        self.mixer = nn.Conv1d(d_input, d_input, kernel_size=4, padding=3, groups=1)
        self.head = nn.Linear(d_input, 1)
        self.concept_proj = nn.Linear(d_input, d_ontology)

    def forward(self, x, concept_emb=None):
        B, L, D = x.shape
        h = self.norm(x)
        if concept_emb is None:
            concept_emb = self.concept_proj(h.mean(dim=1))
        c_exp = concept_emb.unsqueeze(1).expand(B, L, -1).reshape(B * L, -1)
        gate = torch.sigmoid(self.up(self.daruan(self.down(h.reshape(B * L, D)), c_exp)))
        gate = gate.reshape(B, L, D)
        mixed = self.mixer(h.transpose(1, 2))[:, :, :L].transpose(1, 2)
        return self.head(x + gate * mixed)


class OPFAQKANMamba(nn.Module):
    """Full OPFA-QKAN-Mamba (our proposed model)."""
    def __init__(self, d_input, latent_dim=16, d_ontology=16, reps=6, use_mamba=True):
        super().__init__()
        self.block = OPFAQKANMambaBlock(
            d_model=d_input, latent_dim=latent_dim,
            d_ontology=d_ontology, reps=reps,
            use_mamba=use_mamba, device="cpu",
        )
        self.head = nn.Linear(d_input, 1)
        self.concept_proj = nn.Linear(d_input, d_ontology)

    def forward(self, x, concept_emb=None):
        if concept_emb is None:
            concept_emb = self.concept_proj(x.mean(dim=1))
        out, _ = self.block(x, concept_emb)
        return self.head(out)


# === Ablation Models ===

class AblationNoMultiAxis(nn.Module):
    """Ablation: all axes σ_z (no multi-axis encoding)."""
    def __init__(self, d_input, latent_dim=16, d_ontology=16, reps=6, use_mamba=True):
        super().__init__()
        config = [{"name": "infection", "axis": "z", "n_layers": 2},
                  {"name": "hemodynamics", "axis": "z", "n_layers": 2},
                  {"name": "organ_function", "axis": "z", "n_layers": 2}]
        self.block = OPFAQKANMambaBlock(
            d_model=d_input, latent_dim=latent_dim, d_ontology=d_ontology,
            reps=reps, band_config=config, use_mamba=use_mamba, device="cpu",
        )
        self.head = nn.Linear(d_input, 1)
        self.concept_proj = nn.Linear(d_input, d_ontology)

    def forward(self, x, concept_emb=None):
        if concept_emb is None:
            concept_emb = self.concept_proj(x.mean(dim=1))
        out, _ = self.block(x, concept_emb)
        return self.head(out)


class AblationNoPartition(nn.Module):
    """Ablation: no frequency partition (single band, all layers shared)."""
    def __init__(self, d_input, latent_dim=16, d_ontology=16, reps=6, use_mamba=True):
        super().__init__()
        config = [{"name": "unified", "axis": "z", "n_layers": 6}]
        self.block = OPFAQKANMambaBlock(
            d_model=d_input, latent_dim=latent_dim, d_ontology=d_ontology,
            reps=reps, band_config=config, use_mamba=use_mamba, device="cpu",
        )
        self.head = nn.Linear(d_input, 1)
        self.concept_proj = nn.Linear(d_input, d_ontology)

    def forward(self, x, concept_emb=None):
        if concept_emb is None:
            concept_emb = self.concept_proj(x.mean(dim=1))
        out, _ = self.block(x, concept_emb)
        return self.head(out)


class AblationNoAdaptiveMeasure(nn.Module):
    """Ablation: fixed σ_z measurement (no adaptive)."""
    def __init__(self, d_input, latent_dim=16, d_ontology=16, reps=6, use_mamba=True):
        super().__init__()
        self.block = OPFAQKANMambaBlock(
            d_model=d_input, latent_dim=latent_dim, d_ontology=d_ontology,
            reps=reps, use_mamba=use_mamba, device="cpu",
        )
        with torch.no_grad():
            self.block.gate.daruan.adaptive_measure.basis_proj.weight.zero_()
            self.block.gate.daruan.adaptive_measure.basis_proj.bias.copy_(
                torch.tensor([0.0, 0.0, 10.0]))
        for p in self.block.gate.daruan.adaptive_measure.parameters():
            p.requires_grad = False
        self.head = nn.Linear(d_input, 1)
        self.concept_proj = nn.Linear(d_input, d_ontology)

    def forward(self, x, concept_emb=None):
        if concept_emb is None:
            concept_emb = self.concept_proj(x.mean(dim=1))
        out, _ = self.block(x, concept_emb)
        return self.head(out)


class AblationNoMamba(nn.Module):
    """Ablation: OPFA-QKAN with Conv1d instead of Mamba."""
    def __init__(self, d_input, latent_dim=16, d_ontology=16, reps=6):
        super().__init__()
        self.block = OPFAQKANMambaBlock(
            d_model=d_input, latent_dim=latent_dim, d_ontology=d_ontology,
            reps=reps, use_mamba=False, device="cpu",
        )
        self.head = nn.Linear(d_input, 1)
        self.concept_proj = nn.Linear(d_input, d_ontology)

    def forward(self, x, concept_emb=None):
        if concept_emb is None:
            concept_emb = self.concept_proj(x.mean(dim=1))
        out, _ = self.block(x, concept_emb)
        return self.head(out)


# === Training & Evaluation ===

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        pred = logits[:, -1, 0]
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = torch.sigmoid(logits[:, -1, 0])
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.numpy())
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    auroc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.5
    auprc = average_precision_score(labels, preds) if len(np.unique(labels)) > 1 else 0.0
    return {"auroc": auroc, "auprc": auprc}


def run():
    log("=" * 60)
    log(f"PHASE D: PhysioNet 2019 Sepsis Challenge (DEVICE={DEVICE})")
    log("=" * 60)

    use_mamba = DEVICE.type == "cuda"
    log(f"use_mamba={use_mamba}")

    log("Loading dataset (40K patients)...")
    t0 = time.time()
    train_loader, val_loader, test_loader = get_2019_dataloaders(
        batch_size=128, seq_len=48, max_patients=None
    )
    log(f"Loaded in {time.time()-t0:.1f}s")

    d_input = N_DYNAMIC  # 34
    n_epochs = 50
    lr = 1e-3

    models = {
        "LSTM": LSTMBaseline(d_input, hidden=64),
        "GRU": GRUBaseline(d_input, hidden=64),
        "Transformer": TransformerBaseline(d_input, hidden=64, nhead=4, n_layers=2, seq_len=48),
        "TCN": TCNBaseline(d_input, hidden=64, kernel_size=3, n_layers=4),
        "Mamba_Pure": PureMamba(d_input, d_model=64),
        "Mamba_MLP_Gate": MambaMLPGate(d_input, d_model=64),
        "Mamba_Sigmoid_Gate": MambaSigmoidGate(d_input, d_model=64),
        "Original_DARUAN_Mamba": OriginalDARUANMamba(d_input, latent_dim=16, d_ontology=16),
        "OPFA_QKAN_Mamba": OPFAQKANMamba(d_input, latent_dim=16, d_ontology=16, reps=6, use_mamba=use_mamba),
        "Ablation_NoMultiAxis": AblationNoMultiAxis(d_input, latent_dim=16, d_ontology=16, reps=6, use_mamba=use_mamba),
        "Ablation_NoPartition": AblationNoPartition(d_input, latent_dim=16, d_ontology=16, reps=6, use_mamba=use_mamba),
        "Ablation_NoAdaptiveMeasure": AblationNoAdaptiveMeasure(d_input, latent_dim=16, d_ontology=16, reps=6, use_mamba=use_mamba),
        "Ablation_NoMamba": AblationNoMamba(d_input, latent_dim=16, d_ontology=16, reps=6),
    }

    results = {}
    for name, model in models.items():
        model = model.to(DEVICE)
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
            loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            t_epoch = time.time() - t_start
            val_metrics = evaluate(model, val_loader, DEVICE)
            log(f"  [{name}] Epoch {epoch+1:2d}/{n_epochs} | loss={loss:.4f} | val_auroc={val_metrics['auroc']:.4f} | {t_epoch:.1f}s")

            if val_metrics["auroc"] > best_val_auroc:
                best_val_auroc = val_metrics["auroc"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= max_patience:
                log(f"  [{name}] Early stopping at epoch {epoch+1}")
                break

        if best_state:
            model.load_state_dict(best_state)
            model = model.to(DEVICE)
        test_metrics = evaluate(model, test_loader, DEVICE)
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
    log("FINAL RESULTS (PhysioNet 2019 Sepsis Challenge)")
    log("=" * 70)
    log(f"{'Model':<28} {'AUROC':<10} {'AUPRC':<10} {'Params':<10}")
    log("-" * 58)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["test_auroc"]):
        marker = " <-- ours" if name == "OPFA_QKAN_Mamba" else ""
        log(f"{name:<28} {r['test_auroc']:.4f}    {r['test_auprc']:.4f}    {r['params']}{marker}")

    out_path = "/home/project/experiments/results_sepsis2019.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run()

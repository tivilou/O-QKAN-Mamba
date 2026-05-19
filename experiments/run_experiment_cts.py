"""Phase D: External validation on Clinical Time Series datasets.

Runs all baselines on eICU Sepsis (3.3K), eICU Cardiac Arrest (64K), MIMIC GIB (2.6K).
Logs to /home/project/logs/experiment_clinical_ts.log
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
from experiments.data_loader import get_dataloaders, ICUSepsisDataset, ICUCardiacArrestDataset
from src.qkan.experimental.opfa_qkan_mamba_block import OPFAQKANMambaBlock
from src.qkan.experimental.ontology_modulated_daruan import OntologyModulatedDARUAN

LOG_PATH = "/home/project/logs/experiment_clinical_ts.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logger = logging.getLogger("exp_cts")
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


# === Models (4-dim input) ===

class LSTMBaseline(nn.Module):
    def __init__(self, d_input, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(d_input, hidden, batch_first=True, num_layers=2, dropout=0.1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x, c=None):
        out, _ = self.lstm(x)
        return self.head(out)

class GRUBaseline(nn.Module):
    def __init__(self, d_input, hidden=32):
        super().__init__()
        self.gru = nn.GRU(d_input, hidden, batch_first=True, num_layers=2, dropout=0.1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x, c=None):
        out, _ = self.gru(x)
        return self.head(out)

class TransformerBaseline(nn.Module):
    def __init__(self, d_input, hidden=32, nhead=4, n_layers=2, seq_len=48):
        super().__init__()
        self.proj = nn.Linear(d_input, hidden)
        self.pos = nn.Parameter(torch.randn(1, seq_len, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead,
            dim_feedforward=hidden*2, batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x, c=None):
        return self.head(self.enc(self.proj(x) + self.pos[:, :x.shape[1], :]))


class TCNBaseline(nn.Module):
    def __init__(self, d_input, hidden=32, kernel_size=3, n_layers=3):
        super().__init__()
        layers = []
        in_ch = d_input
        for i in range(n_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.extend([nn.Conv1d(in_ch, hidden, kernel_size, dilation=dilation, padding=padding), nn.ReLU()])
            in_ch = hidden
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x, c=None):
        h = self.net(x.transpose(1, 2))[:, :, :x.shape[1]].transpose(1, 2)
        return self.head(h)


class PureMamba(nn.Module):
    def __init__(self, d_input, d_model=32):
        super().__init__()
        self.proj_in = nn.Linear(d_input, d_model)
        self.mixer = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        self.proj_out = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, 1)
    def forward(self, x, c=None):
        h = self.proj_in(x)
        mixed = self.mixer(h.transpose(1, 2))[:, :, :x.shape[1]].transpose(1, 2)
        return self.head(torch.relu(self.proj_out(mixed)))


class MambaMLPGate(nn.Module):
    def __init__(self, d_input, d_model=32):
        super().__init__()
        self.proj_in = nn.Linear(d_input, d_model)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model), nn.Sigmoid())
        self.mixer = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        self.head = nn.Linear(d_model, 1)
    def forward(self, x, c=None):
        h = self.proj_in(x)
        g = self.gate(h)
        mixed = self.mixer(h.transpose(1, 2))[:, :, :x.shape[1]].transpose(1, 2)
        return self.head(h + g * mixed)


class MambaSigmoidGate(nn.Module):
    def __init__(self, d_input, d_model=32):
        super().__init__()
        self.proj_in = nn.Linear(d_input, d_model)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.mixer = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        self.head = nn.Linear(d_model, 1)
    def forward(self, x, c=None):
        h = self.proj_in(x)
        g = torch.sigmoid(self.gate_proj(h))
        mixed = self.mixer(h.transpose(1, 2))[:, :, :x.shape[1]].transpose(1, 2)
        return self.head(h + g * mixed)


class OriginalDARUANMamba(nn.Module):
    def __init__(self, d_input, latent_dim=8, d_ontology=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_input)
        self.down = nn.Linear(d_input, latent_dim)
        self.daruan = OntologyModulatedDARUAN(dim=latent_dim, reps=6, d_ontology=d_ontology, device="cpu")
        self.up = nn.Linear(latent_dim, d_input)
        self.mixer = nn.Conv1d(d_input, d_input, kernel_size=4, padding=3)
        self.head = nn.Linear(d_input, 1)
        self.concept_proj = nn.Linear(d_input, d_ontology)
    def forward(self, x, c=None):
        B, L, D = x.shape
        h = self.norm(x)
        if c is None:
            c = self.concept_proj(h.mean(dim=1))
        c_exp = c.unsqueeze(1).expand(B, L, -1).reshape(B * L, -1)
        gate = torch.sigmoid(self.up(self.daruan(self.down(h.reshape(B*L, D)), c_exp))).reshape(B, L, D)
        mixed = self.mixer(h.transpose(1, 2))[:, :, :L].transpose(1, 2)
        return self.head(x + gate * mixed)


class OPFAQKANMamba(nn.Module):
    def __init__(self, d_input, latent_dim=8, d_ontology=16, reps=6, use_mamba=True):
        super().__init__()
        self.block = OPFAQKANMambaBlock(
            d_model=d_input, latent_dim=latent_dim, d_ontology=d_ontology,
            reps=reps, use_mamba=use_mamba, device="cpu")
        self.head = nn.Linear(d_input, 1)
        self.concept_proj = nn.Linear(d_input, d_ontology)
    def forward(self, x, c=None):
        if c is None:
            c = self.concept_proj(x.mean(dim=1))
        out, _ = self.block(x, c)
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


def build_models(d_input, use_mamba):
    return {
        "LSTM": LSTMBaseline(d_input, hidden=32),
        "GRU": GRUBaseline(d_input, hidden=32),
        "Transformer": TransformerBaseline(d_input, hidden=32, nhead=4, n_layers=2, seq_len=48),
        "TCN": TCNBaseline(d_input, hidden=32, kernel_size=3, n_layers=3),
        "Mamba_Pure": PureMamba(d_input, d_model=32),
        "Mamba_MLP_Gate": MambaMLPGate(d_input, d_model=32),
        "Mamba_Sigmoid_Gate": MambaSigmoidGate(d_input, d_model=32),
        "Original_DARUAN_Mamba": OriginalDARUANMamba(d_input, latent_dim=8, d_ontology=16),
        "OPFA_QKAN_Mamba": OPFAQKANMamba(d_input, latent_dim=8, d_ontology=16, reps=6, use_mamba=use_mamba),
    }


def run_on_dataset(dataset_name, n_epochs=50, batch_size=128, lr=1e-3):
    log(f"\n{'#'*60}")
    log(f"# Dataset: {dataset_name}")
    log(f"{'#'*60}")

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=dataset_name, batch_size=batch_size, seq_len=48
    )
    log(f"  Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")

    d_input = 4
    use_mamba = DEVICE.type == "cuda"
    models = build_models(d_input, use_mamba)
    results = {}

    for name, model in models.items():
        model = model.to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        log(f"\n  [{name}] ({n_params} params) training...")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        best_val_auroc = 0
        best_state = None
        patience = 0

        for epoch in range(n_epochs):
            loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_m = evaluate(model, val_loader, DEVICE)
            if (epoch + 1) % 10 == 0:
                log(f"    Epoch {epoch+1:2d} | loss={loss:.4f} | val_auroc={val_m['auroc']:.4f}")
            if val_m["auroc"] > best_val_auroc:
                best_val_auroc = val_m["auroc"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= 10:
                log(f"    Early stop at epoch {epoch+1}")
                break

        if best_state:
            model.load_state_dict(best_state)
            model = model.to(DEVICE)
        test_m = evaluate(model, test_loader, DEVICE)
        results[name] = {"test_auroc": test_m["auroc"], "test_auprc": test_m["auprc"], "params": n_params}
        log(f"  [{name}] DONE >> AUROC={test_m['auroc']:.4f}, AUPRC={test_m['auprc']:.4f}")

    return results


def run():
    log("=" * 60)
    log(f"PHASE D: Clinical Time Series External Validation (DEVICE={DEVICE})")
    log("=" * 60)

    all_results = {}
    for ds in ["sepsis", "cardiac_arrest", "gib"]:
        all_results[ds] = run_on_dataset(ds)

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY: ALL DATASETS")
    log("=" * 70)
    for ds, res in all_results.items():
        log(f"\n  [{ds}]")
        log(f"  {'Model':<28} {'AUROC':<10} {'AUPRC':<10} {'Params'}")
        log(f"  {'-'*56}")
        for name, r in sorted(res.items(), key=lambda x: -x[1]["test_auroc"]):
            marker = " <--" if "OPFA" in name else ""
            log(f"  {name:<28} {r['test_auroc']:.4f}    {r['test_auprc']:.4f}    {r['params']}{marker}")

    out_path = "/home/project/experiments/results_clinical_ts.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run()

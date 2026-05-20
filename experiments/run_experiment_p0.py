"""P0 Experiments: Gate × Mixer matrix + OPFA ablation.

Experiment 1: Gate comparison (fixed Conv1d mixer)
Experiment 2: Mixer comparison (fixed OPFA gate)
Experiment 3: OPFA component ablation (fixed Conv1d mixer)

All on PhysioNet 2019 v1.2 (per-hour, window=24, no leakage).
"""
import sys, os, time, json, logging, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.data_loader_2019_v2 import get_2019_dataloaders_v2, N_FEATURES
from src.qkan.experimental.opfa_qkan_mamba_block import OPFAQKANMambaBlock
from src.qkan.experimental.ontology_modulated_daruan import OntologyModulatedDARUAN
from src.qkan.experimental.opfa_daruan import OPFADaruan

LOG_PATH = "/home/project/logs/experiment_p0.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logger = logging.getLogger("p0")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_PATH, mode="w")
fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(fh)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(sh)
def log(msg): logger.info(msg)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_INPUT = N_FEATURES  # 34
WINDOW = 24
LATENT_DIM = 16
D_ONTOLOGY = 16
REPS = 6


# ============================================================
# MIXERS
# ============================================================

class IdentityMixer(nn.Module):
    def forward(self, x): return x

class Conv1dMixer(nn.Module):
    def __init__(self, d): super().__init__(); self.conv = nn.Conv1d(d, d, kernel_size=4, padding=3)
    def forward(self, x): return self.conv(x.transpose(1,2))[:, :, :x.shape[1]].transpose(1,2)

class CausalAttentionMixer(nn.Module):
    def __init__(self, d, nhead=2):
        super().__init__()
        hidden = ((d + nhead - 1) // nhead) * nhead
        self.proj_in = nn.Linear(d, hidden)
        self.attn = nn.MultiheadAttention(hidden, nhead, batch_first=True, dropout=0.1)
        self.proj_out = nn.Linear(hidden, d)
    def forward(self, x):
        h = self.proj_in(x)
        mask = nn.Transformer.generate_square_subsequent_mask(h.shape[1], device=h.device)
        out, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        return self.proj_out(out)

class GRUMixer(nn.Module):
    def __init__(self, d): super().__init__(); self.gru = nn.GRU(d, d, batch_first=True)
    def forward(self, x): out, _ = self.gru(x); return out

class MambaMixer(nn.Module):
    def __init__(self, d):
        super().__init__()
        try:
            from mamba_ssm import Mamba
            self.mamba = Mamba(d_model=d, d_state=16, d_conv=4, expand=1)
        except ImportError:
            self.mamba = None; self.conv = nn.Conv1d(d, d, kernel_size=4, padding=3)
    def forward(self, x):
        if self.mamba: return self.mamba(x)
        return self.conv(x.transpose(1,2))[:, :, :x.shape[1]].transpose(1,2)


# ============================================================
# GATES
# ============================================================

class NoGate(nn.Module):
    def __init__(self, d): super().__init__()
    def forward(self, x, c=None): return torch.ones_like(x)

class SigmoidGate(nn.Module):
    def __init__(self, d): super().__init__(); self.proj = nn.Linear(d, d)
    def forward(self, x, c=None):
        B, L, D = x.shape
        return torch.sigmoid(self.proj(x))

class MLPGate(nn.Module):
    def __init__(self, d, hidden=None):
        super().__init__()
        hidden = hidden or d * 2
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
    def forward(self, x, c=None):
        return torch.sigmoid(self.net(x))


class OriginalDARUANGate(nn.Module):
    def __init__(self, d, latent_dim, d_ontology):
        super().__init__()
        self.down = nn.Linear(d, latent_dim)
        self.daruan = OntologyModulatedDARUAN(dim=latent_dim, reps=REPS, d_ontology=d_ontology, device="cpu")
        self.up = nn.Linear(latent_dim, d)
        self.concept_proj = nn.Linear(d, d_ontology)
    def forward(self, x, c=None):
        B, L, D = x.shape
        if c is None: c = self.concept_proj(x.mean(dim=1))
        c_exp = c.unsqueeze(1).expand(B, L, -1).reshape(B*L, -1)
        h = self.down(x.reshape(B*L, D))
        h = self.daruan(h, c_exp)
        return torch.sigmoid(self.up(h)).reshape(B, L, D)


class OPFAGate(nn.Module):
    def __init__(self, d, latent_dim, d_ontology, reps=6):
        super().__init__()
        self.down = nn.Linear(d, latent_dim)
        self.opfa = OPFADaruan(dim=latent_dim, reps=reps, d_ontology=d_ontology, device="cpu")
        self.up = nn.Linear(latent_dim, d)
        self.concept_proj = nn.Linear(d, d_ontology)
    def forward(self, x, c=None):
        B, L, D = x.shape
        if c is None: c = self.concept_proj(x.mean(dim=1))
        c_exp = c.unsqueeze(1).expand(B, L, -1).reshape(B*L, -1)
        h = self.down(x.reshape(B*L, D))
        h, _ = self.opfa(h, c_exp)
        return torch.sigmoid(self.up(h)).reshape(B, L, D)


class OPFAGateNoMultiAxis(nn.Module):
    """Ablation: all bands use R_z (no multi-axis encoding)."""
    def __init__(self, d, latent_dim, d_ontology, reps=6):
        super().__init__()
        self.down = nn.Linear(d, latent_dim)
        band_config = [
            {"name": "infection", "axis": "z", "n_layers": 2},
            {"name": "hemodynamics", "axis": "z", "n_layers": 2},
            {"name": "organ_function", "axis": "z", "n_layers": 2},
        ]
        self.opfa = OPFADaruan(dim=latent_dim, reps=reps, d_ontology=d_ontology, band_config=band_config, device="cpu")
        self.up = nn.Linear(latent_dim, d)
        self.concept_proj = nn.Linear(d, d_ontology)
    def forward(self, x, c=None):
        B, L, D = x.shape
        if c is None: c = self.concept_proj(x.mean(dim=1))
        c_exp = c.unsqueeze(1).expand(B, L, -1).reshape(B*L, -1)
        h = self.down(x.reshape(B*L, D))
        h, _ = self.opfa(h, c_exp)
        return torch.sigmoid(self.up(h)).reshape(B, L, D)


class OPFAGateNoPartition(nn.Module):
    """Ablation: single band (no frequency partitioning)."""
    def __init__(self, d, latent_dim, d_ontology, reps=6):
        super().__init__()
        self.down = nn.Linear(d, latent_dim)
        band_config = [{"name": "unified", "axis": "z", "n_layers": reps}]
        self.opfa = OPFADaruan(dim=latent_dim, reps=reps, d_ontology=d_ontology, band_config=band_config, device="cpu")
        self.up = nn.Linear(latent_dim, d)
        self.concept_proj = nn.Linear(d, d_ontology)
    def forward(self, x, c=None):
        B, L, D = x.shape
        if c is None: c = self.concept_proj(x.mean(dim=1))
        c_exp = c.unsqueeze(1).expand(B, L, -1).reshape(B*L, -1)
        h = self.down(x.reshape(B*L, D))
        h, _ = self.opfa(h, c_exp)
        return torch.sigmoid(self.up(h)).reshape(B, L, D)


class OPFAGateNoAdaptiveMeasure(nn.Module):
    """Ablation: fixed sigma_z measurement (no adaptive measurement)."""
    def __init__(self, d, latent_dim, d_ontology, reps=6):
        super().__init__()
        self.down = nn.Linear(d, latent_dim)
        self.opfa = OPFADaruan(dim=latent_dim, reps=reps, d_ontology=d_ontology, device="cpu")
        self.up = nn.Linear(latent_dim, d)
        self.concept_proj = nn.Linear(d, d_ontology)
        self.fixed_measure = True
    def forward(self, x, c=None):
        B, L, D = x.shape
        if c is None: c = self.concept_proj(x.mean(dim=1))
        c_exp = c.unsqueeze(1).expand(B, L, -1).reshape(B*L, -1)
        h = self.down(x.reshape(B*L, D))
        h = self.opfa.forward_fixed_measure(h, c_exp)
        return torch.sigmoid(self.up(h)).reshape(B, L, D)


# ============================================================
# UNIFIED MODEL
# ============================================================

class GatedMixerModel(nn.Module):
    """Generic model: y = x + gate(x) * mixer(x), then classify last step."""
    def __init__(self, d_input, gate, mixer):
        super().__init__()
        self.norm = nn.LayerNorm(d_input)
        self.gate = gate
        self.mixer = mixer
        self.head = nn.Linear(d_input, 1)

    def forward(self, x):
        h = self.norm(x)
        g = self.gate(h)
        m = self.mixer(h)
        out = x + g * m
        return self.head(out[:, -1, :]).squeeze(-1)


# ============================================================
# TRAINING
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return total_loss / n


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = torch.sigmoid(model(xb))
            preds.extend(pred.cpu().numpy())
            labels.extend(yb.numpy())
    p, l = np.array(preds), np.array(labels)
    auroc = roc_auc_score(l, p) if len(np.unique(l)) > 1 else 0.5
    auprc = average_precision_score(l, p) if len(np.unique(l)) > 1 else 0.0
    return {"auroc": auroc, "auprc": auprc}


def train_and_eval(name, model, train_loader, val_loader, test_loader, max_epochs=30, patience=7):
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  [{name}] params={n_params}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pos_weight = torch.tensor([10.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_val, best_state, pat = 0, None, 0
    for epoch in range(max_epochs):
        t0 = time.time()
        loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_m = evaluate(model, val_loader, DEVICE)
        t_ep = time.time() - t0
        if (epoch+1) % 5 == 0 or epoch == 0:
            log(f"    Ep {epoch+1:2d} | loss={loss:.4f} | val={val_m['auroc']:.4f} | {t_ep:.0f}s")
        if val_m["auroc"] > best_val:
            best_val = val_m["auroc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= patience: break
    if best_state:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)
    test_m = evaluate(model, test_loader, DEVICE)
    log(f"    >> AUROC={test_m['auroc']:.4f}, AUPRC={test_m['auprc']:.4f}")
    return {"test_auroc": test_m["auroc"], "test_auprc": test_m["auprc"],
            "best_val": best_val, "params": n_params}


# ============================================================
# MAIN
# ============================================================

def run():
    log("=" * 60)
    log(f"P0 EXPERIMENTS (DEVICE={DEVICE})")
    log("=" * 60)

    log("\nLoading PhysioNet 2019 v1.2 data...")
    t0 = time.time()
    train_loader, val_loader, test_loader = get_2019_dataloaders_v2(
        batch_size=256, window=WINDOW, max_patients=None)
    log(f"Loaded in {time.time()-t0:.0f}s")

    D = D_INPUT
    all_results = {}

    # ----------------------------------------------------------
    # Experiment 1: Gate comparison (fixed Conv1d mixer)
    # ----------------------------------------------------------
    log("\n" + "=" * 60)
    log("EXP 1: Gate Comparison (Conv1d mixer fixed)")
    log("=" * 60)

    exp1_configs = {
        "NoGate+Conv1d": (NoGate(D), Conv1dMixer(D)),
        "Sigmoid+Conv1d": (SigmoidGate(D), Conv1dMixer(D)),
        "MLP+Conv1d": (MLPGate(D), Conv1dMixer(D)),
        "OrigDARUAN+Conv1d": (OriginalDARUANGate(D, LATENT_DIM, D_ONTOLOGY), Conv1dMixer(D)),
        "OPFA+Conv1d": (OPFAGate(D, LATENT_DIM, D_ONTOLOGY, REPS), Conv1dMixer(D)),
    }
    for name, (gate, mixer) in exp1_configs.items():
        model = GatedMixerModel(D, gate, mixer)
        all_results[f"exp1_{name}"] = train_and_eval(name, model, train_loader, val_loader, test_loader)

    # ----------------------------------------------------------
    # Experiment 2: Mixer comparison (fixed OPFA gate)
    # ----------------------------------------------------------
    log("\n" + "=" * 60)
    log("EXP 2: Mixer Comparison (OPFA gate fixed)")
    log("=" * 60)

    exp2_configs = {
        "OPFA+Identity": (OPFAGate(D, LATENT_DIM, D_ONTOLOGY, REPS), IdentityMixer()),
        "OPFA+Conv1d": None,  # already in exp1
        "OPFA+Attention": (OPFAGate(D, LATENT_DIM, D_ONTOLOGY, REPS), CausalAttentionMixer(D)),
        "OPFA+GRU": (OPFAGate(D, LATENT_DIM, D_ONTOLOGY, REPS), GRUMixer(D)),
        "OPFA+Mamba": (OPFAGate(D, LATENT_DIM, D_ONTOLOGY, REPS), MambaMixer(D)),
    }
    for name, cfg in exp2_configs.items():
        if cfg is None:
            all_results[f"exp2_{name}"] = all_results["exp1_OPFA+Conv1d"]
            log(f"  [{name}] (reuse from exp1)")
            continue
        gate, mixer = cfg
        model = GatedMixerModel(D, gate, mixer)
        all_results[f"exp2_{name}"] = train_and_eval(name, model, train_loader, val_loader, test_loader)

    # ----------------------------------------------------------
    # Experiment 3: OPFA ablation (fixed Conv1d mixer)
    # ----------------------------------------------------------
    log("\n" + "=" * 60)
    log("EXP 3: OPFA Ablation (Conv1d mixer fixed)")
    log("=" * 60)

    exp3_configs = {
        "OPFA_Full": None,  # reuse from exp1
        "OPFA_NoMultiAxis": (OPFAGateNoMultiAxis(D, LATENT_DIM, D_ONTOLOGY, REPS), Conv1dMixer(D)),
        "OPFA_NoPartition": (OPFAGateNoPartition(D, LATENT_DIM, D_ONTOLOGY, REPS), Conv1dMixer(D)),
        "OPFA_NoAdaptMeasure": (OPFAGateNoAdaptiveMeasure(D, LATENT_DIM, D_ONTOLOGY, REPS), Conv1dMixer(D)),
    }
    for name, cfg in exp3_configs.items():
        if cfg is None:
            all_results[f"exp3_{name}"] = all_results["exp1_OPFA+Conv1d"]
            log(f"  [{name}] (reuse from exp1)")
            continue
        gate, mixer = cfg
        model = GatedMixerModel(D, gate, mixer)
        all_results[f"exp3_{name}"] = train_and_eval(name, model, train_loader, val_loader, test_loader)

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)

    for exp_name in ["EXP 1: Gate Comparison", "EXP 2: Mixer Comparison", "EXP 3: OPFA Ablation"]:
        prefix = exp_name.split(":")[0].lower().replace(" ", "")
        log(f"\n  {exp_name}")
        log(f"  {'Config':<28} {'AUROC':<10} {'AUPRC':<10} {'Params'}")
        log(f"  {'-'*56}")
        for k, v in sorted(all_results.items(), key=lambda x: -x[1]["test_auroc"]):
            if k.startswith(prefix):
                short = k.split("_", 1)[1]
                log(f"  {short:<28} {v['test_auroc']:.4f}    {v['test_auprc']:.4f}    {v['params']}")

    out_path = "/home/project/experiments/results_p0.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run()

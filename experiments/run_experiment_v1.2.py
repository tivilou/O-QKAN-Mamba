"""Phase D v1.2: PhysioNet 2019 with correct per-hour prediction task.

No data leakage: at each time step t, model sees only past data [t-W+1:t+1]
and predicts SepsisLabel[t].
"""
import sys, os, time, json, logging, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.data_loader_2019_v2 import get_2019_dataloaders_v2, N_FEATURES
from src.qkan.experimental.opfa_qkan_mamba_block import OPFAQKANMambaBlock
from src.qkan.experimental.ontology_modulated_daruan import OntologyModulatedDARUAN

LOG_PATH = "/home/project/logs/experiment_v1.2_perhour.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logger = logging.getLogger("v12")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_PATH, mode="w")
fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(fh)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(sh)
def log(msg): logger.info(msg)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Models ===

class LSTMModel(nn.Module):
    def __init__(self, d_input, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(d_input, hidden, batch_first=True, num_layers=2, dropout=0.1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x, c=None):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self, d_input, hidden=64):
        super().__init__()
        self.gru = nn.GRU(d_input, hidden, batch_first=True, num_layers=2, dropout=0.1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x, c=None):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class TransformerModel(nn.Module):
    def __init__(self, d_input, hidden=64, nhead=4, n_layers=2, seq_len=24):
        super().__init__()
        self.proj = nn.Linear(d_input, hidden)
        self.pos = nn.Parameter(torch.randn(1, seq_len, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead,
            dim_feedforward=hidden*2, batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x, c=None):
        h = self.enc(self.proj(x) + self.pos[:, :x.shape[1], :])
        return self.head(h[:, -1, :]).squeeze(-1)


class TCNModel(nn.Module):
    def __init__(self, d_input, hidden=64, kernel_size=3, n_layers=4):
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
        h = self.net(x.transpose(1, 2))[:, :, -1]
        return self.head(h).squeeze(-1)


class OriginalDARUANModel(nn.Module):
    def __init__(self, d_input, latent_dim=16, d_ontology=16):
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
        out = x + gate * mixed
        return self.head(out[:, -1, :]).squeeze(-1)


class OPFAQKANMambaModel(nn.Module):
    def __init__(self, d_input, latent_dim=16, d_ontology=16, reps=6, use_mamba=True):
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
        return self.head(out[:, -1, :]).squeeze(-1)


# === Training ===

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = torch.sigmoid(logits)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.numpy())
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    auroc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.5
    auprc = average_precision_score(labels, preds) if len(np.unique(labels)) > 1 else 0.0
    return {"auroc": auroc, "auprc": auprc}


def run():
    log("=" * 60)
    log(f"v1.2: Per-Hour Prediction (No Leakage) DEVICE={DEVICE}")
    log("=" * 60)

    use_mamba = DEVICE.type == "cuda"
    window = 24

    log(f"Window={window}, Loading full dataset...")
    t0 = time.time()
    train_loader, val_loader, test_loader = get_2019_dataloaders_v2(
        batch_size=256, window=window, max_patients=None)
    log(f"Loaded in {time.time()-t0:.1f}s")

    d_input = N_FEATURES
    models = {
        "LSTM": LSTMModel(d_input, hidden=64),
        "GRU": GRUModel(d_input, hidden=64),
        "Transformer": TransformerModel(d_input, hidden=64, nhead=4, n_layers=2, seq_len=window),
        "TCN": TCNModel(d_input, hidden=64, kernel_size=3, n_layers=4),
        "Original_DARUAN": OriginalDARUANModel(d_input, latent_dim=16, d_ontology=16),
        "OPFA_QKAN_Mamba": OPFAQKANMambaModel(d_input, latent_dim=16, d_ontology=16, reps=6, use_mamba=use_mamba),
    }

    results = {}
    for name, model in models.items():
        model = model.to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        log(f"")
        log(f"[{name}] ({n_params} params)")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        pos_weight = torch.tensor([10.0]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        best_val = 0
        best_state = None
        patience = 0

        for epoch in range(30):
            t_start = time.time()
            loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_m = evaluate(model, val_loader, DEVICE)
            t_ep = time.time() - t_start
            log(f"  Epoch {epoch+1:2d} | loss={loss:.4f} | val_auroc={val_m['auroc']:.4f} | {t_ep:.1f}s")
            if val_m["auroc"] > best_val:
                best_val = val_m["auroc"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= 7:
                log(f"  Early stop at epoch {epoch+1}")
                break

        if best_state:
            model.load_state_dict(best_state)
            model = model.to(DEVICE)
        test_m = evaluate(model, test_loader, DEVICE)
        results[name] = {"test_auroc": test_m["auroc"], "test_auprc": test_m["auprc"],
                         "best_val_auroc": best_val, "params": n_params}
        log(f"  >> Test AUROC={test_m['auroc']:.4f}, AUPRC={test_m['auprc']:.4f}")

    log("")
    log("=" * 60)
    log("FINAL RESULTS (v1.2 Per-Hour, No Leakage)")
    log("=" * 60)
    log(f"{'Model':<24} {'AUROC':<10} {'AUPRC':<10} {'Params'}")
    log("-" * 54)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["test_auroc"]):
        marker = " <-- ours" if "OPFA" in name else ""
        log(f"{name:<24} {r['test_auroc']:.4f}    {r['test_auprc']:.4f}    {r['params']}{marker}")

    out_path = "/home/project/experiments/results_v1.2_perhour.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run()

"""Data loader for PhysioNet 2019 Sepsis Challenge.

40,336 patients, 34 clinical features + SepsisLabel, hourly records.
Handles NaN imputation, feature normalization, and frequency band mapping.
"""
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "challenge-2019", "training"
)

# Feature groups mapped to frequency bands
INFECTION_FEATURES = ["Temp", "WBC", "Lactate", "Fibrinogen", "BaseExcess", "BUN"]
HEMODYNAMICS_FEATURES = ["HR", "MAP", "SBP", "DBP", "Resp", "O2Sat", "EtCO2"]
ORGAN_FEATURES = ["Creatinine", "Bilirubin_direct", "Bilirubin_total",
                  "Platelets", "AST", "Alkalinephos"]
OTHER_LAB_FEATURES = ["HCO3", "FiO2", "pH", "PaCO2", "SaO2", "Calcium",
                      "Chloride", "Glucose", "Magnesium", "Phosphate",
                      "Potassium", "TroponinI", "Hct", "Hgb", "PTT"]
STATIC_FEATURES = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]

ALL_DYNAMIC_FEATURES = (INFECTION_FEATURES + HEMODYNAMICS_FEATURES +
                        ORGAN_FEATURES + OTHER_LAB_FEATURES)
N_DYNAMIC = len(ALL_DYNAMIC_FEATURES)  # 34

BAND_INDICES = {
    "infection": [ALL_DYNAMIC_FEATURES.index(f) for f in INFECTION_FEATURES],
    "hemodynamics": [ALL_DYNAMIC_FEATURES.index(f) for f in HEMODYNAMICS_FEATURES],
    "organ_function": [ALL_DYNAMIC_FEATURES.index(f) for f in ORGAN_FEATURES],
}


def load_all_patients(data_dir=None, max_patients=None):
    """Load all .psv files into a list of (features, label) tuples."""
    data_dir = data_dir or DATA_DIR
    files = sorted(
        glob.glob(os.path.join(data_dir, "training_setA", "*.psv")) +
        glob.glob(os.path.join(data_dir, "training_setB", "*.psv"))
    )
    if max_patients:
        files = files[:max_patients]

    patients = []
    for f in files:
        df = pd.read_csv(f, sep="|")
        features = df[ALL_DYNAMIC_FEATURES].values.astype(np.float32)
        label = int(df["SepsisLabel"].max())
        patients.append((features, label))
    return patients


class Sepsis2019Dataset(Dataset):
    """PhysioNet 2019 Sepsis Challenge dataset.

    Handles NaN imputation (forward fill + zero), normalization,
    and padding/truncation to fixed sequence length.
    """

    def __init__(self, patients, seq_len=48, normalize_stats=None):
        self.seq_len = seq_len
        self.patients = patients
        self.n_features = N_DYNAMIC

        # Compute normalization stats from this split or use provided
        if normalize_stats is None:
            all_vals = []
            for feat, _ in patients:
                all_vals.append(feat)
            all_vals = np.concatenate(all_vals, axis=0)
            self.mean = np.nanmean(all_vals, axis=0)
            self.std = np.nanstd(all_vals, axis=0)
            self.std[self.std < 1e-6] = 1.0
        else:
            self.mean, self.std = normalize_stats

    def get_normalize_stats(self):
        return (self.mean, self.std)

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        features, label = self.patients[idx]

        # Forward fill NaN, then fill remaining with 0
        features = features.copy()
        df = pd.DataFrame(features)
        df = df.ffill().bfill().fillna(0.0)
        features = df.values.astype(np.float32)

        # Normalize
        features = (features - self.mean) / self.std
        # Replace any remaining inf/nan with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Pad or truncate
        L = features.shape[0]
        if L >= self.seq_len:
            x = features[:self.seq_len]
        else:
            x = np.zeros((self.seq_len, self.n_features), dtype=np.float32)
            x[:L] = features

        return torch.from_numpy(x), torch.tensor(label, dtype=torch.float32)


def get_2019_dataloaders(batch_size=128, seq_len=48, max_patients=None, val_ratio=0.1, test_ratio=0.1):
    """Load PhysioNet 2019 data and return train/val/test dataloaders."""
    print("  Loading .psv files...")
    patients = load_all_patients(max_patients=max_patients)
    np.random.seed(42)
    indices = np.random.permutation(len(patients))

    n_test = int(len(patients) * test_ratio)
    n_val = int(len(patients) * val_ratio)
    n_train = len(patients) - n_test - n_val

    train_patients = [patients[i] for i in indices[:n_train]]
    val_patients = [patients[i] for i in indices[n_train:n_train + n_val]]
    test_patients = [patients[i] for i in indices[n_train + n_val:]]

    print(f"  Split: train={len(train_patients)}, val={len(val_patients)}, test={len(test_patients)}")

    train_ds = Sepsis2019Dataset(train_patients, seq_len=seq_len)
    stats = train_ds.get_normalize_stats()
    val_ds = Sepsis2019Dataset(val_patients, seq_len=seq_len, normalize_stats=stats)
    test_ds = Sepsis2019Dataset(test_patients, seq_len=seq_len, normalize_stats=stats)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

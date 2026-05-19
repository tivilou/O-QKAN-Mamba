"""PhysioNet 2019 data loader v2: per-hour prediction (no data leakage).

Task: At each hour t, using data from hours [max(0, t-W+1), t],
predict SepsisLabel at hour t.

This is the correct formulation: the model only sees past data,
and predicts whether the patient is currently in the sepsis window.
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

ALL_DYNAMIC_FEATURES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2",
    "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine",
    "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
    "PTT", "WBC", "Fibrinogen", "Platelets",
]
N_FEATURES = len(ALL_DYNAMIC_FEATURES)  # 34

BAND_INDICES = {
    "infection": [0, 1, 2, 3, 4, 5],      # Temp, WBC, Lactate, etc.
    "hemodynamics": [6, 7, 8, 9, 10, 11, 12],
    "organ_function": [13, 14, 15, 16, 17, 18],
}


def load_patient_files(data_dir=None, max_patients=None):
    """Load raw patient data as list of DataFrames."""
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
        patients.append(df)
    return patients


class Sepsis2019DatasetV2(Dataset):
    """Per-hour prediction dataset (correct formulation).

    For each patient, generates multiple samples: one per valid time step.
    At time t, the model sees data from [max(0, t-W+1) : t+1] and predicts
    SepsisLabel[t]. This ensures no future data leakage.
    """

    def __init__(self, patients, window=24, normalize_stats=None):
        self.window = window
        self.n_features = N_FEATURES

        # Compute normalization stats
        if normalize_stats is None:
            all_vals = []
            for df in patients:
                all_vals.append(df[ALL_DYNAMIC_FEATURES].values)
            all_vals = np.concatenate(all_vals, axis=0)
            self.mean = np.nanmean(all_vals, axis=0).astype(np.float32)
            self.std = np.nanstd(all_vals, axis=0).astype(np.float32)
            self.std[self.std < 1e-6] = 1.0
        else:
            self.mean, self.std = normalize_stats

        # Build index: (patient_idx, time_step)
        self.samples = []
        self.patient_data = []
        for p_idx, df in enumerate(patients):
            features = df[ALL_DYNAMIC_FEATURES].values.astype(np.float32)
            # Forward fill + backward fill + zero fill
            feat_df = pd.DataFrame(features, columns=ALL_DYNAMIC_FEATURES)
            feat_df = feat_df.ffill().bfill().fillna(0.0)
            features = feat_df.values.astype(np.float32)
            labels = df["SepsisLabel"].values.astype(np.float32)
            self.patient_data.append((features, labels))
            for t in range(len(df)):
                self.samples.append((p_idx, t))

    def get_normalize_stats(self):
        return (self.mean, self.std)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p_idx, t = self.samples[idx]
        features, labels = self.patient_data[p_idx]

        # Extract window [max(0, t-W+1) : t+1]
        start = max(0, t - self.window + 1)
        window_data = features[start:t + 1]

        # Normalize
        window_data = (window_data - self.mean) / self.std
        window_data = np.nan_to_num(window_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Pad to fixed window size (left-pad with zeros)
        x = np.zeros((self.window, self.n_features), dtype=np.float32)
        L = window_data.shape[0]
        x[self.window - L:] = window_data

        label = labels[t]
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.float32)


def get_2019_dataloaders_v2(batch_size=256, window=24, max_patients=None,
                            val_ratio=0.1, test_ratio=0.1):
    """Load PhysioNet 2019 data with per-hour prediction (no leakage)."""
    print("  Loading .psv files...")
    patients = load_patient_files(max_patients=max_patients)
    np.random.seed(42)
    indices = np.random.permutation(len(patients))

    n_test = int(len(patients) * test_ratio)
    n_val = int(len(patients) * val_ratio)
    n_train = len(patients) - n_test - n_val

    train_patients = [patients[i] for i in indices[:n_train]]
    val_patients = [patients[i] for i in indices[n_train:n_train + n_val]]
    test_patients = [patients[i] for i in indices[n_train + n_val:]]

    pos_train = sum(1 for p in train_patients if p["SepsisLabel"].max() > 0)
    print(f"  Split: train={len(train_patients)} ({pos_train} pos), "
          f"val={len(val_patients)}, test={len(test_patients)}")

    train_ds = Sepsis2019DatasetV2(train_patients, window=window)
    stats = train_ds.get_normalize_stats()
    val_ds = Sepsis2019DatasetV2(val_patients, window=window, normalize_stats=stats)
    test_ds = Sepsis2019DatasetV2(test_patients, window=window, normalize_stats=stats)

    print(f"  Samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

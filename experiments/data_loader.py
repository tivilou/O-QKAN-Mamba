"""Data loader for eICU Sepsis and Cardiac Arrest datasets.

Adapts the Clinical Time Series datasets to our OPFA-QKAN-Mamba model format.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw",
    "clinical-time-series-datasets-for-trajectory-flow-matching-evaluation-"
    "icu-sepsis-icu-cardiac-arrest-and-icu-gib-cohorts-1.0.0"
)


class ICUSepsisDataset(Dataset):
    """eICU Sepsis dataset: 3362 patients, 24h ICU time series."""

    FEATURES = ["hr_normalized", "map_normalized", "norepi_inf_scaled", "apache"]
    TARGET = "ICU_MORT"

    def __init__(self, split: str = "train", seq_len: int = 48, data_dir: str = None):
        self.seq_len = seq_len
        data_dir = data_dir or DATA_DIR
        df = pd.read_csv(os.path.join(data_dir, "eICU_sepsis_physionet.csv"))
        df = df[df["label"] == split].copy()

        self.patients = []
        for hadm_id, group in df.groupby("HADM_ID"):
            group = group.sort_values("TIME_FROM_ADM")
            features = group[self.FEATURES].values.astype(np.float32)
            target = int(group[self.TARGET].iloc[0])
            self.patients.append((features, target))

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        features, target = self.patients[idx]
        # Pad or truncate to seq_len
        L = features.shape[0]
        if L >= self.seq_len:
            x = features[:self.seq_len]
        else:
            x = np.zeros((self.seq_len, features.shape[1]), dtype=np.float32)
            x[:L] = features
        return torch.from_numpy(x), torch.tensor(target, dtype=torch.float32)


class ICUCardiacArrestDataset(Dataset):
    """eICU Cardiac Arrest dataset: 64589 patients, richer features."""

    FEATURES = ["hr_normalized", "dbp_normalized", "rr_normalized", "age_normalized"]
    TARGET = "ICU_MORT"

    def __init__(self, split: str = "train", seq_len: int = 48, data_dir: str = None):
        self.seq_len = seq_len
        data_dir = data_dir or DATA_DIR
        df = pd.read_csv(os.path.join(data_dir, "eICU_cardiacArrest_physionet.csv"))
        df = df[df["label"] == split].copy()

        self.patients = []
        for hadm_id, group in df.groupby("HADM_ID"):
            group = group.sort_values("TIME_FROM_ADM")
            features = group[self.FEATURES].values.astype(np.float32)
            target = int(group[self.TARGET].iloc[0])
            self.patients.append((features, target))

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        features, target = self.patients[idx]
        L = features.shape[0]
        if L >= self.seq_len:
            x = features[:self.seq_len]
        else:
            x = np.zeros((self.seq_len, features.shape[1]), dtype=np.float32)
            x[:L] = features
        return torch.from_numpy(x), torch.tensor(target, dtype=torch.float32)


class MIMICGIBDataset(Dataset):
    """MIMIC GIB dataset: 2602 patients with gastrointestinal bleeding."""

    FEATURES = ["hr_normalized", "map_normalized", "pressor_gaussian", "bloodprod_gaussian"]
    TARGET = "HOSP_MORT"

    def __init__(self, split: str = "train", seq_len: int = 48, data_dir: str = None):
        self.seq_len = seq_len
        data_dir = data_dir or DATA_DIR
        df = pd.read_csv(os.path.join(data_dir, "MIMIC_gib_physionet.csv"))
        df = df[df["label"] == split].copy()

        self.patients = []
        for hadm_id, group in df.groupby("HADM_ID"):
            group = group.sort_values("TIME_FROM_ADM")
            features = group[self.FEATURES].values.astype(np.float32)
            target = int(group[self.TARGET].iloc[0])
            self.patients.append((features, target))

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        features, target = self.patients[idx]
        L = features.shape[0]
        if L >= self.seq_len:
            x = features[:self.seq_len]
        else:
            x = np.zeros((self.seq_len, features.shape[1]), dtype=np.float32)
            x[:L] = features
        return torch.from_numpy(x), torch.tensor(target, dtype=torch.float32)


def get_dataloaders(dataset_name="sepsis", batch_size=64, seq_len=48):
    """Get train/val/test dataloaders."""
    if dataset_name == "sepsis":
        cls = ICUSepsisDataset
    elif dataset_name == "cardiac_arrest":
        cls = ICUCardiacArrestDataset
    elif dataset_name == "gib":
        cls = MIMICGIBDataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    train_ds = cls("train", seq_len=seq_len)
    val_ds = cls("val", seq_len=seq_len)
    test_ds = cls("test", seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

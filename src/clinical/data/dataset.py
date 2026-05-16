"""PyTorch Dataset for MIMIC-IV sepsis prediction."""
import numpy as np
import torch
from torch.utils.data import Dataset


class MIMICSepsisDataset(Dataset):
    """Dataset for sepsis prediction from preprocessed EHR data.

    Returns dict with: features, mask, label, patient_id, times, concepts
    """

    def __init__(
        self,
        features: np.ndarray,
        masks: np.ndarray,
        labels: np.ndarray,
        stay_ids: np.ndarray,
        n_concepts: int = 64,
    ):
        """
        Args:
            features: (N, T, F) normalized feature array
            masks: (N, T, F) binary mask (1=observed, 0=missing)
            labels: (N,) binary labels
            stay_ids: (N,) patient/stay identifiers
            n_concepts: placeholder dimension for ontology concepts
        """
        self.features = torch.from_numpy(features).float()
        self.masks = torch.from_numpy(masks).float()
        self.labels = torch.from_numpy(labels).long()
        self.stay_ids = stay_ids
        self.n_concepts = n_concepts
        self.seq_len = features.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "mask": self.masks[idx],
            "label": self.labels[idx],
            "patient_id": self.stay_ids[idx],
            "times": torch.arange(self.seq_len, dtype=torch.float),
            "concepts": torch.zeros(self.seq_len, self.n_concepts),
        }

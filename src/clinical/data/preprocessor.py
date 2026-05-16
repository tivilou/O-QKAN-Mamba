"""EHR preprocessing: resampling, normalization, imputation."""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


def _load_config():
    config_path = Path(__file__).parent / "cohort_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


class EHRPreprocessor:
    """Preprocess raw EHR time-series into model-ready tensors."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or _load_config()
        self.feature_names = self._get_feature_names()
        self.stats: dict = {}

    def _get_feature_names(self) -> list[str]:
        cfg = self.config["features"]
        return cfg["vitals"] + cfg["labs"]

    def generate_synthetic_features(
        self, cohort: pd.DataFrame, seed: int = 42
    ) -> pd.DataFrame:
        """Generate synthetic hourly features for development."""
        rng = np.random.default_rng(seed)
        records = []
        feature_ranges = {
            "heart_rate": (60, 100, 15),
            "sbp": (100, 140, 20),
            "dbp": (60, 90, 12),
            "map": (65, 95, 12),
            "resp_rate": (12, 20, 4),
            "spo2": (94, 100, 2),
            "temperature": (36.5, 37.5, 0.5),
            "fio2": (0.21, 0.5, 0.1),
            "wbc": (4.5, 11.0, 3.0),
            "hemoglobin": (12.0, 16.0, 2.0),
            "platelet": (150, 400, 80),
            "creatinine": (0.6, 1.2, 0.4),
            "bilirubin": (0.2, 1.0, 0.5),
            "lactate": (0.5, 2.0, 0.8),
            "pao2": (80, 100, 15),
            "pco2": (35, 45, 5),
            "ph": (7.35, 7.45, 0.05),
            "glucose": (70, 140, 30),
        }

        for _, row in cohort.iterrows():
            n_hours = int(row["los_hours"])
            for h in range(n_hours):
                rec = {"stay_id": row["stay_id"], "hour": h}
                for feat in self.feature_names:
                    mean, high, std = feature_ranges.get(feat, (50, 100, 10))
                    val = rng.normal(mean + (high - mean) / 2, std)
                    if rng.random() < 0.15:
                        val = np.nan
                    rec[feat] = val
                records.append(rec)
        return pd.DataFrame(records)

    def resample_hourly(
        self, features_df: pd.DataFrame, cohort: pd.DataFrame, max_hours: int = 48
    ) -> pd.DataFrame:
        """Align features to 1-hour time grid, truncate to max_hours."""
        if "hour" in features_df.columns:
            features_df = features_df[features_df["hour"] < max_hours]
            return features_df
        return features_df

    def compute_statistics(self, features_df: pd.DataFrame) -> dict:
        """Compute per-feature mean and std for normalization."""
        stats = {}
        for feat in self.feature_names:
            if feat in features_df.columns:
                stats[feat] = {
                    "mean": float(features_df[feat].mean()),
                    "std": float(features_df[feat].std()),
                }
        self.stats = stats
        return stats

    def normalize(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Z-score normalization using precomputed statistics."""
        df = features_df.copy()
        for feat in self.feature_names:
            if feat in df.columns and feat in self.stats:
                mean = self.stats[feat]["mean"]
                std = self.stats[feat]["std"]
                if std > 0:
                    df[feat] = (df[feat] - mean) / std
        return df

    def mask_and_impute(self, features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Forward-fill imputation + binary mask for missing values."""
        mask_df = features_df[self.feature_names].notna().astype(float)
        filled_df = features_df.copy()
        for feat in self.feature_names:
            if feat in filled_df.columns:
                filled_df[feat] = filled_df.groupby("stay_id")[feat].ffill()
                filled_df[feat] = filled_df[feat].fillna(0.0)
        return filled_df, mask_df

    def to_tensors(
        self, features_df: pd.DataFrame, mask_df: pd.DataFrame, max_hours: int = 48
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert to fixed-size arrays [n_patients, max_hours, n_features]."""
        stay_ids = features_df["stay_id"].unique()
        n_patients = len(stay_ids)
        n_features = len(self.feature_names)

        X = np.zeros((n_patients, max_hours, n_features), dtype=np.float32)
        M = np.zeros((n_patients, max_hours, n_features), dtype=np.float32)

        for i, sid in enumerate(stay_ids):
            patient = features_df[features_df["stay_id"] == sid]
            patient_mask = mask_df.loc[patient.index]
            T = min(len(patient), max_hours)
            for j, feat in enumerate(self.feature_names):
                if feat in patient.columns:
                    X[i, :T, j] = patient[feat].values[:T]
                    M[i, :T, j] = patient_mask[feat].values[:T]
        return X, M

    def save_stats(self, path: str):
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=2)

    def load_stats(self, path: str):
        with open(path) as f:
            self.stats = json.load(f)

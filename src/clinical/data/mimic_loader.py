"""MIMIC-IV data loader with support for demo and synthetic modes."""
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


def _load_config():
    config_path = Path(__file__).parent / "cohort_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


class MIMICIVLoader:
    """Load and filter MIMIC-IV tables for sepsis cohort extraction."""

    def __init__(self, data_dir: str, mode: str = "auto"):
        """
        Args:
            data_dir: path to MIMIC-IV root (contains hosp/, icu/ subdirs)
            mode: "full", "demo", "synthetic", or "auto" (detect)
        """
        self.data_dir = Path(data_dir)
        self.config = _load_config()
        self.mode = self._detect_mode(mode)

    def _detect_mode(self, mode: str) -> str:
        if mode != "auto":
            return mode
        if (self.data_dir / "hosp").exists():
            return "full"
        if (self.data_dir / "mimiciv_demo").exists():
            return "demo"
        return "synthetic"

    def load_cohort(self) -> pd.DataFrame:
        if self.mode == "synthetic":
            return self._generate_synthetic_cohort()
        return self._load_real_cohort()

    def _load_real_cohort(self) -> pd.DataFrame:
        base = self.data_dir
        if self.mode == "demo":
            base = self.data_dir / "mimiciv_demo"

        patients = pd.read_csv(base / "hosp" / "patients.csv.gz")
        admissions = pd.read_csv(base / "hosp" / "admissions.csv.gz")
        icustays = pd.read_csv(base / "icu" / "icustays.csv.gz")

        cohort = icustays.merge(patients, on="subject_id")
        cohort = cohort.merge(admissions, on=["subject_id", "hadm_id"])

        cohort = self.filter_first_icu_stay(cohort)
        return cohort

    def filter_first_icu_stay(self, cohort: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config["cohort"]
        if "anchor_age" in cohort.columns:
            cohort = cohort[cohort["anchor_age"] >= cfg["min_age"]]

        cohort["intime"] = pd.to_datetime(cohort["intime"])
        cohort["outtime"] = pd.to_datetime(cohort["outtime"])
        cohort["los_hours"] = (
            (cohort["outtime"] - cohort["intime"]).dt.total_seconds() / 3600
        )
        cohort = cohort[cohort["los_hours"] >= cfg["min_stay_hours"]]
        cohort = cohort[cohort["los_hours"] <= cfg["max_stay_hours"]]

        if cfg["first_icu_stay_only"]:
            cohort = cohort.sort_values("intime").groupby("subject_id").first().reset_index()

        return cohort.reset_index(drop=True)

    def load_chartevents(
        self, stay_ids: list, itemids: Optional[list] = None
    ) -> pd.DataFrame:
        if self.mode == "synthetic":
            return pd.DataFrame(columns=["stay_id", "charttime", "itemid", "valuenum"])
        base = self.data_dir
        if self.mode == "demo":
            base = self.data_dir / "mimiciv_demo"
        path = base / "icu" / "chartevents.csv.gz"
        chunks = []
        for chunk in pd.read_csv(path, chunksize=100000):
            chunk = chunk[chunk["stay_id"].isin(stay_ids)]
            if itemids:
                chunk = chunk[chunk["itemid"].isin(itemids)]
            chunks.append(chunk[["stay_id", "charttime", "itemid", "valuenum"]])
        if not chunks:
            return pd.DataFrame(columns=["stay_id", "charttime", "itemid", "valuenum"])
        return pd.concat(chunks, ignore_index=True)

    def load_labevents(
        self, hadm_ids: list, itemids: Optional[list] = None
    ) -> pd.DataFrame:
        if self.mode == "synthetic":
            return pd.DataFrame(columns=["hadm_id", "charttime", "itemid", "valuenum"])
        base = self.data_dir
        if self.mode == "demo":
            base = self.data_dir / "mimiciv_demo"
        path = base / "hosp" / "labevents.csv.gz"
        chunks = []
        for chunk in pd.read_csv(path, chunksize=100000):
            chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]
            if itemids:
                chunk = chunk[chunk["itemid"].isin(itemids)]
            chunks.append(chunk[["hadm_id", "charttime", "itemid", "valuenum"]])
        if not chunks:
            return pd.DataFrame(columns=["hadm_id", "charttime", "itemid", "valuenum"])
        return pd.concat(chunks, ignore_index=True)

    def _generate_synthetic_cohort(self, n_patients: int = 200) -> pd.DataFrame:
        """Generate synthetic ICU cohort for development without MIMIC access."""
        rng = np.random.default_rng(42)
        records = []
        for i in range(n_patients):
            los_hours = rng.uniform(24, 168)
            intime = pd.Timestamp("2020-01-01") + pd.Timedelta(hours=rng.uniform(0, 8760))
            records.append({
                "subject_id": 10000 + i,
                "hadm_id": 20000 + i,
                "stay_id": 30000 + i,
                "intime": intime,
                "outtime": intime + pd.Timedelta(hours=los_hours),
                "los_hours": los_hours,
                "anchor_age": rng.integers(18, 90),
            })
        return pd.DataFrame(records)

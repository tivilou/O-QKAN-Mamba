"""Sepsis-3 labeling: SOFA increase >= 2 within 24h of suspected infection."""
import numpy as np
import pandas as pd

from .sofa import (
    compute_sofa_cardiovascular,
    compute_sofa_coagulation,
    compute_sofa_liver,
    compute_sofa_renal,
    compute_sofa_respiratory,
)


def compute_sofa_total(row: pd.Series) -> int:
    score = 0
    pf = row.get("pao2_fio2_ratio", np.nan)
    score += compute_sofa_respiratory(pf)
    score += compute_sofa_coagulation(row.get("platelet", np.nan))
    score += compute_sofa_liver(row.get("bilirubin", np.nan))
    score += compute_sofa_cardiovascular(row.get("map", np.nan))
    score += compute_sofa_renal(row.get("creatinine", np.nan))
    return score


def label_sepsis3(
    features_df: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """
    Label patients with Sepsis-3 onset.

    For synthetic mode: assigns sepsis based on clinical heuristics
    (elevated lactate + low MAP + high HR pattern).

    Args:
        features_df: hourly features with columns [stay_id, hour, ...]
        cohort: cohort dataframe with stay metadata

    Returns:
        DataFrame with columns [stay_id, label, onset_hour]
        label: 1 = sepsis, 0 = no sepsis
        onset_hour: hour of sepsis onset (NaN if no sepsis)
    """
    results = []
    for stay_id in cohort["stay_id"].unique():
        patient_data = features_df[features_df["stay_id"] == stay_id]
        if patient_data.empty:
            results.append({"stay_id": stay_id, "label": 0, "onset_hour": np.nan})
            continue

        label, onset = _detect_sepsis_onset(patient_data)
        results.append({"stay_id": stay_id, "label": label, "onset_hour": onset})

    return pd.DataFrame(results)


def _detect_sepsis_onset(patient_data: pd.DataFrame):
    """
    Detect sepsis onset using simplified Sepsis-3 criteria.

    For real MIMIC data: looks for SOFA increase >= 2 within 24h window.
    For synthetic data: uses surrogate markers (low MAP + high HR + elevated lactate).
    """
    sofa_cols = ["pao2_fio2_ratio", "platelet", "bilirubin", "map", "creatinine"]
    has_sofa_data = any(col in patient_data.columns for col in sofa_cols)

    if has_sofa_data:
        sofa_scores = patient_data.apply(compute_sofa_total, axis=1).values
        for i in range(24, len(sofa_scores)):
            baseline = min(sofa_scores[max(0, i - 24):i]) if i > 0 else 0
            if sofa_scores[i] - baseline >= 2:
                return 1, patient_data.iloc[i].get("hour", i)
        return 0, np.nan

    # Synthetic fallback: heuristic based on available vitals
    if "map" in patient_data.columns and "heart_rate" in patient_data.columns:
        for idx, row in patient_data.iterrows():
            hr = row.get("heart_rate", 80)
            map_val = row.get("map", 75)
            lactate = row.get("lactate", 1.0)
            if hr > 100 and map_val < 65 and lactate > 2.0:
                return 1, row.get("hour", 0)
    return 0, np.nan


def label_sepsis3_synthetic(
    cohort: pd.DataFrame, prevalence: float = 0.15, seed: int = 42
) -> pd.DataFrame:
    """Assign synthetic sepsis labels with target prevalence."""
    rng = np.random.default_rng(seed)
    n = len(cohort)
    n_sepsis = int(n * prevalence)
    labels = np.zeros(n, dtype=int)
    sepsis_idx = rng.choice(n, size=n_sepsis, replace=False)
    labels[sepsis_idx] = 1
    onset_hours = np.full(n, np.nan)
    for idx in sepsis_idx:
        los = cohort.iloc[idx].get("los_hours", 72)
        onset_hours[idx] = rng.uniform(6, min(los - 6, 48))

    return pd.DataFrame({
        "stay_id": cohort["stay_id"].values,
        "label": labels,
        "onset_hour": onset_hours,
    })

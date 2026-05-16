"""SOFA score calculation utilities.

Reusable across sepsis labeling (Phase 2) and DAG validation (Phase 5).
"""
import numpy as np
import pandas as pd


def compute_sofa_respiratory(pao2_fio2: float) -> int:
    if pd.isna(pao2_fio2):
        return 0
    if pao2_fio2 < 100:
        return 4
    if pao2_fio2 < 200:
        return 3
    if pao2_fio2 < 300:
        return 2
    if pao2_fio2 < 400:
        return 1
    return 0


def compute_sofa_coagulation(platelets: float) -> int:
    if pd.isna(platelets):
        return 0
    if platelets < 20:
        return 4
    if platelets < 50:
        return 3
    if platelets < 100:
        return 2
    if platelets < 150:
        return 1
    return 0


def compute_sofa_liver(bilirubin: float) -> int:
    if pd.isna(bilirubin):
        return 0
    if bilirubin >= 12.0:
        return 4
    if bilirubin >= 6.0:
        return 3
    if bilirubin >= 2.0:
        return 2
    if bilirubin >= 1.2:
        return 1
    return 0


def compute_sofa_cardiovascular(map_val: float) -> int:
    if pd.isna(map_val):
        return 0
    if map_val < 70:
        return 1
    return 0


def compute_sofa_renal(creatinine: float) -> int:
    if pd.isna(creatinine):
        return 0
    if creatinine >= 5.0:
        return 4
    if creatinine >= 3.5:
        return 3
    if creatinine >= 2.0:
        return 2
    if creatinine >= 1.2:
        return 1
    return 0

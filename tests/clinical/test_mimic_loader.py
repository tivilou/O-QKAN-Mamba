"""Tests for MIMIC-IV data pipeline (synthetic mode)."""
import numpy as np
import pandas as pd
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.clinical.data.mimic_loader import MIMICIVLoader
from src.clinical.data.sepsis_labels import label_sepsis3_synthetic
from src.clinical.data.preprocessor import EHRPreprocessor
from src.clinical.data.dataset import MIMICSepsisDataset


@pytest.fixture
def loader():
    return MIMICIVLoader("data/raw/mimiciv", mode="synthetic")


@pytest.fixture
def cohort(loader):
    return loader.load_cohort()


def test_synthetic_cohort_size(cohort):
    assert len(cohort) == 200


def test_cohort_columns(cohort):
    required = ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los_hours"]
    for col in required:
        assert col in cohort.columns


def test_cohort_age_filter(cohort):
    assert (cohort["anchor_age"] >= 18).all()


def test_cohort_los_range(cohort):
    assert (cohort["los_hours"] >= 24).all()
    assert (cohort["los_hours"] <= 240).all()


def test_sepsis_labels_distribution(cohort):
    labels = label_sepsis3_synthetic(cohort, prevalence=0.15)
    assert len(labels) == len(cohort)
    assert labels["label"].sum() > 0
    assert labels["label"].sum() < len(labels)
    prevalence = labels["label"].mean()
    assert 0.10 <= prevalence <= 0.20


def test_preprocessor_synthetic_features(cohort):
    prep = EHRPreprocessor()
    features = prep.generate_synthetic_features(cohort)
    assert "stay_id" in features.columns
    assert "hour" in features.columns
    assert "heart_rate" in features.columns
    assert len(features) > 0


def test_full_pipeline_to_dataset(cohort):
    prep = EHRPreprocessor()
    features = prep.generate_synthetic_features(cohort)
    features = prep.resample_hourly(features, cohort, max_hours=48)
    prep.compute_statistics(features)
    features = prep.normalize(features)
    filled, mask = prep.mask_and_impute(features)
    X, M = prep.to_tensors(filled, mask, max_hours=48)

    labels_df = label_sepsis3_synthetic(cohort, prevalence=0.15)
    stay_ids = filled["stay_id"].unique()
    labels = np.array([
        labels_df[labels_df["stay_id"] == sid]["label"].values[0]
        for sid in stay_ids
    ])

    dataset = MIMICSepsisDataset(X, M, labels, stay_ids)
    assert len(dataset) == len(stay_ids)

    sample = dataset[0]
    assert sample["features"].shape == (48, 18)
    assert sample["mask"].shape == (48, 18)
    assert sample["label"].item() in [0, 1]
    assert sample["concepts"].shape == (48, 64)

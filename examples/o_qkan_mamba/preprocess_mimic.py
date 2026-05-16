"""End-to-end MIMIC-IV preprocessing script."""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.clinical.data.mimic_loader import MIMICIVLoader
from src.clinical.data.sepsis_labels import label_sepsis3, label_sepsis3_synthetic
from src.clinical.data.preprocessor import EHRPreprocessor


def main():
    parser = argparse.ArgumentParser(description="Preprocess MIMIC-IV for sepsis prediction")
    parser.add_argument("--data_dir", type=str, default="data/raw/mimiciv")
    parser.add_argument("--output_dir", type=str, default="data/processed/mimic_sepsis")
    parser.add_argument("--max_hours", type=int, default=48)
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "full", "demo", "synthetic"])
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "../..", args.data_dir)
    output_dir = os.path.join(os.path.dirname(__file__), "../..", args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("MIMIC-IV Sepsis Preprocessing Pipeline")
    print("=" * 60)

    loader = MIMICIVLoader(data_dir, mode=args.mode)
    print(f"Mode: {loader.mode}")

    print("\n[1/5] Loading cohort...")
    cohort = loader.load_cohort()
    print(f"  Cohort size: {len(cohort)} patients")

    print("\n[2/5] Generating features...")
    preprocessor = EHRPreprocessor()
    if loader.mode == "synthetic":
        features_df = preprocessor.generate_synthetic_features(cohort)
    else:
        features_df = _load_real_features(loader, cohort, preprocessor)
    print(f"  Total records: {len(features_df)}")

    print("\n[3/5] Labeling sepsis...")
    if loader.mode == "synthetic":
        labels_df = label_sepsis3_synthetic(cohort, prevalence=0.15)
    else:
        features_df = preprocessor.resample_hourly(features_df, cohort, args.max_hours)
        labels_df = label_sepsis3(features_df, cohort)
    n_sepsis = labels_df["label"].sum()
    print(f"  Sepsis positive: {n_sepsis}/{len(labels_df)} ({100*n_sepsis/len(labels_df):.1f}%)")

    print("\n[4/5] Preprocessing...")
    features_df = preprocessor.resample_hourly(features_df, cohort, args.max_hours)
    stats = preprocessor.compute_statistics(features_df)
    features_df = preprocessor.normalize(features_df)
    filled_df, mask_df = preprocessor.mask_and_impute(features_df)

    X, M = preprocessor.to_tensors(filled_df, mask_df, max_hours=args.max_hours)
    stay_ids = filled_df["stay_id"].unique()
    labels = np.array([
        labels_df[labels_df["stay_id"] == sid]["label"].values[0]
        if sid in labels_df["stay_id"].values else 0
        for sid in stay_ids
    ])

    print(f"  Tensor shape: {X.shape}")
    print(f"  Labels: {labels.sum()} positive, {len(labels) - labels.sum()} negative")

    print("\n[5/5] Saving...")
    np.save(os.path.join(output_dir, "features.npy"), X)
    np.save(os.path.join(output_dir, "masks.npy"), M)
    np.save(os.path.join(output_dir, "labels.npy"), labels)
    np.save(os.path.join(output_dir, "stay_ids.npy"), stay_ids)
    preprocessor.save_stats(os.path.join(output_dir, "stats.json"))
    cohort.to_parquet(os.path.join(output_dir, "cohort.parquet"))
    labels_df.to_parquet(os.path.join(output_dir, "labels.parquet"))

    missing_rates = {}
    for i, feat in enumerate(preprocessor.feature_names):
        missing_rates[feat] = float(1.0 - M[:, :, i].mean())

    summary = {
        "mode": loader.mode,
        "n_patients": int(len(stay_ids)),
        "sepsis_prevalence": float(labels.mean()),
        "n_sepsis": int(labels.sum()),
        "tensor_shape": list(X.shape),
        "feature_names": preprocessor.feature_names,
        "mean_seq_length": float(M.sum(axis=(1, 2)).mean() / len(preprocessor.feature_names)),
        "missing_rates": missing_rates,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved to: {output_dir}")
    print(f"  Summary: {json.dumps(summary, indent=2)}")
    return summary


def _load_real_features(loader, cohort, preprocessor):
    """Load real MIMIC-IV features from chartevents/labevents."""
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), "../../src/clinical/data/cohort_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    all_itemids = []
    itemid_to_feat = {}
    for feat, ids in config["itemid_map"].items():
        all_itemids.extend(ids)
        for iid in ids:
            itemid_to_feat[iid] = feat

    stay_ids = cohort["stay_id"].tolist()
    hadm_ids = cohort["hadm_id"].tolist()

    chart = loader.load_chartevents(stay_ids, all_itemids)
    labs = loader.load_labevents(hadm_ids, all_itemids)

    records = []
    for _, row in cohort.iterrows():
        sid = row["stay_id"]
        intime = pd.to_datetime(row["intime"])
        patient_chart = chart[chart["stay_id"] == sid].copy()
        patient_chart["hour"] = (
            (pd.to_datetime(patient_chart["charttime"]) - intime).dt.total_seconds() / 3600
        ).astype(int)
        for _, r in patient_chart.iterrows():
            feat = itemid_to_feat.get(r["itemid"])
            if feat:
                records.append({"stay_id": sid, "hour": int(r["hour"]), feat: r["valuenum"]})

    if records:
        return pd.DataFrame(records).groupby(["stay_id", "hour"]).first().reset_index()
    return preprocessor.generate_synthetic_features(cohort)


if __name__ == "__main__":
    main()

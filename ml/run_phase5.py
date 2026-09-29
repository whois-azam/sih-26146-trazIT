"""
ml/run_phase5.py

Ties preprocessing.py + anomaly.py + clustering.py together against the
100k dataset, and validates against the roadmap's Phase 5 criteria:
  - score stability across separate inference batches (the bug fix)
  - clustering quality metrics pass sanity thresholds
"""

from __future__ import annotations

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, "../features")
sys.path.insert(0, ".")

from entity_features import build_features, ALL_FEATURE_COLUMNS, LOG_FEATURE_COLUMNS
from preprocessing import build_preprocessor
from anomaly import CalibratedAnomalyDetector, verify_score_stability
from clustering import fit_behavioral_clustering, evaluate_cluster_purity_against_scenarios


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)[:k]
    return float(y_true[order].mean())


def main():
    raw_dir = "../data/raw"
    proc_dir = "../data/processed"
    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)

    print("Loading data + building features...")
    df = pd.read_csv(os.path.join(raw_dir, "bitcoin_dataset.csv"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    feats = build_features(df)

    with open(os.path.join(raw_dir, "ground_truth.json")) as f:
        ground_truth = json.load(f)
    feats["is_risky"] = feats["entity_id"].map(lambda a: ground_truth.get(a, {}).get("is_risky", False))
    feats["scenario_type"] = feats["entity_id"].map(lambda a: ground_truth.get(a, {}).get("scenario_type"))

    log_columns = [c for c in ALL_FEATURE_COLUMNS if c in LOG_FEATURE_COLUMNS or c.startswith(
        ("tx_count_last", "amount_sum_last", "recent_vs_baseline_ratio")) or c == "incoming_outgoing_ratio"]
    other_columns = [c for c in ALL_FEATURE_COLUMNS if c not in log_columns]
    print(f"log-transformed columns: {len(log_columns)}, other columns: {len(other_columns)}")

    print("\nSplitting entities 70/15/15 (stratified by is_risky)...")
    train_df, temp_df = train_test_split(feats, test_size=0.30, stratify=feats["is_risky"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["is_risky"], random_state=42)
    print(f"train={len(train_df)} ({train_df['is_risky'].sum()} risky), "
          f"val={len(val_df)} ({val_df['is_risky'].sum()} risky), "
          f"test={len(test_df)} ({test_df['is_risky'].sum()} risky)")

    print("\nFitting preprocessor on TRAIN only...")
    preprocessor, out_cols = build_preprocessor(log_columns, other_columns)
    X_train = preprocessor.fit_transform(train_df[log_columns + other_columns])
    X_val = preprocessor.transform(val_df[log_columns + other_columns])
    X_test = preprocessor.transform(test_df[log_columns + other_columns])
    X_full = preprocessor.transform(feats[log_columns + other_columns])  # for clustering on full population

    print("\n=== STAGE 2: Anomaly Detection (frozen-reference calibration) ===")
    detector = CalibratedAnomalyDetector(seed=42)
    t0 = time.time()
    detector.fit(X_train)
    print(f"fit seconds: {round(time.time()-t0, 2)}")

    print("Verifying score stability (the bug fix) on a sample of test rows...")
    stability = verify_score_stability(detector, X_test[:50])
    print(json.dumps(stability, indent=2))

    val_scores = detector.score(X_val)["calibrated_percentile"]
    test_scores = detector.score(X_test)["calibrated_percentile"]

    print("\nSelecting threshold on VAL (best F1), evaluating on TEST...")
    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.80, 0.999, 0.005):
        preds = (val_scores >= thresh).astype(int)
        f1 = f1_score(val_df["is_risky"].astype(int), preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    test_preds = (test_scores >= best_thresh).astype(int)
    y_test = test_df["is_risky"].astype(int).values

    anomaly_eval = {
        "threshold_selected_on_val": round(float(best_thresh), 4),
        "val_best_f1": round(float(best_f1), 4),
        "test_precision": round(float(precision_score(y_test, test_preds, zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_test, test_preds, zero_division=0)), 4),
        "test_f1": round(float(f1_score(y_test, test_preds, zero_division=0)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, test_scores)), 4),
        "test_pr_auc": round(float(average_precision_score(y_test, test_scores)), 4),
        "precision_at_50": round(precision_at_k(y_test, test_scores, min(50, len(y_test))), 4),
        "precision_at_100": round(precision_at_k(y_test, test_scores, min(100, len(y_test))), 4),
        "precision_at_500": round(precision_at_k(y_test, test_scores, min(500, len(y_test))), 4),
        "note": "Metrics measure fit to THIS generator's own definition of anomalous (synthetic ground truth), not real-world detection capability -- see audit Section 16.",
    }
    print(json.dumps(anomaly_eval, indent=2))

    print("\n=== STAGE 3a: Behavioral Clustering ===")
    t0 = time.time()
    cluster_result = fit_behavioral_clustering(X_full, min_samples=10)
    print(f"clustering seconds: {round(time.time()-t0, 2)}, algorithm used: {cluster_result['algorithm']}, "
          f"collapsed_from_dbscan: {cluster_result['collapsed']}")
    print(json.dumps(cluster_result["evaluation"], indent=2))

    purity = evaluate_cluster_purity_against_scenarios(cluster_result["labels"], feats["scenario_type"].tolist())
    print("Cluster purity vs. scenario ground truth:", json.dumps(purity, indent=2))

    print("\nSaving artifacts...")
    detector.save(os.path.join(artifacts_dir, "isolation_forest.joblib"), os.path.join(artifacts_dir, "score_reference.npy"))
    joblib.dump(preprocessor, os.path.join(artifacts_dir, "preprocessor.joblib"))
    with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
        json.dump({"log_columns": log_columns, "other_columns": other_columns}, f, indent=2)

    os.makedirs(proc_dir, exist_ok=True)
    with open(os.path.join(proc_dir, "ml_evaluation.json"), "w") as f:
        json.dump({
            "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
            "score_stability_check": stability,
            "anomaly_detection_evaluation": anomaly_eval,
            "clustering_evaluation": cluster_result["evaluation"],
            "clustering_algorithm_used": cluster_result["algorithm"],
            "clustering_fell_back_from_dbscan": cluster_result["collapsed"],
            "cluster_purity_vs_scenario_ground_truth": purity,
        }, f, indent=2)

    feats_out = feats[["entity_id"]].copy()
    feats_out["behavioral_cluster"] = cluster_result["labels"]
    feats_out.to_csv(os.path.join(proc_dir, "behavioral_clusters.csv"), index=False)

    print(f"\nWritten: {artifacts_dir}/, {proc_dir}/ml_evaluation.json, {proc_dir}/behavioral_clusters.csv")


if __name__ == "__main__":
    main()

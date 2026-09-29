"""
scoring/run_phase6.py

Ties risk.py + explainability/evidence.py together against the full
pipeline (Phases 1-5 outputs), and validates:
  - does the COMBINED score actually beat the anomaly score alone at
    ranking risky entities? (never assumed -- measured)
  - does explanation generation actually avoid padding on a normal entity?
"""

from __future__ import annotations

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, "../features")
sys.path.insert(0, "../ml")
sys.path.insert(0, "../graph")
sys.path.insert(0, "../explainability")
sys.path.insert(0, ".")

from entity_features import build_features, ALL_FEATURE_COLUMNS, LOG_FEATURE_COLUMNS
from anomaly import CalibratedAnomalyDetector
from build import build_graph, wallet_projection
from risk import (
    compute_flagged_neighbor_fraction, compute_risk_scores, derive_thresholds, assign_risk_level,
    calibrate_weights, calibrate_weights_logistic, recombine_risk_score, WEIGHTS,
)
from evidence import compute_feature_baselines, generate_explanation


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)[:k]
    return float(y_true[order].mean())


def main():
    raw_dir = "../data/raw"
    proc_dir = "../data/processed"
    ml_artifacts = "../ml/artifacts"

    print("Loading data + rebuilding features (must match Phase 5 exactly)...")
    df = pd.read_csv(os.path.join(raw_dir, "bitcoin_dataset.csv"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    feats = build_features(df)

    with open(os.path.join(raw_dir, "ground_truth.json")) as f:
        ground_truth = json.load(f)
    feats["is_risky"] = feats["entity_id"].map(lambda a: ground_truth.get(a, {}).get("is_risky", False))

    # reproduce the EXACT same split as Phase 5 (same random_state) so
    # thresholds/baselines are calibrated on the same train population the
    # anomaly detector was fit on -- not a different, inconsistent slice.
    train_df, temp_df = train_test_split(feats, test_size=0.30, stratify=feats["is_risky"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["is_risky"], random_state=42)
    print(f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)} (reproduced from Phase 5 seed)")

    print("\nLoading Phase 5 artifacts (detector, preprocessor)...")
    detector = CalibratedAnomalyDetector.load(
        os.path.join(ml_artifacts, "isolation_forest.joblib"),
        os.path.join(ml_artifacts, "score_reference.npy"),
    )
    preprocessor = joblib.load(os.path.join(ml_artifacts, "preprocessor.joblib"))
    with open(os.path.join(ml_artifacts, "feature_columns.json")) as f:
        cols = json.load(f)
    log_columns, other_columns = cols["log_columns"], cols["other_columns"]

    print("Scoring FULL population through frozen-reference calibration...")
    X_full = preprocessor.transform(feats[log_columns + other_columns])
    anomaly_percentile = detector.score(X_full)["calibrated_percentile"]
    anomaly_by_entity = dict(zip(feats["entity_id"], anomaly_percentile))

    print("\nLoading Phase 3 graph centrality + rebuilding wallet-projection graph...")
    with open(os.path.join(proc_dir, "graph_centrality.json")) as f:
        centrality = json.load(f)
    degree_centrality = centrality["degree_centrality"]

    direct_corr = pd.read_csv(os.path.join(proc_dir, "correlations_direct.csv"))
    fallback_corr = pd.read_csv(os.path.join(proc_dir, "correlations_fallback.csv"))
    t0 = time.time()
    G = build_graph(df, direct_corr, fallback_corr)
    H = wallet_projection(G)
    print(f"graph rebuilt in {round(time.time()-t0, 1)}s")

    print("\nLoading Phase 5 behavioral clusters...")
    clusters_df = pd.read_csv(os.path.join(proc_dir, "behavioral_clusters.csv"))
    cluster_labels = dict(zip(clusters_df["entity_id"], clusters_df["behavioral_cluster"]))

    print("Computing flagged-neighbor fraction from the real graph object...")
    flagged_neighbor_fraction = compute_flagged_neighbor_fraction(
        feats["entity_id"].tolist(), anomaly_by_entity, H,
    )

    centrality_reference = np.array(list(degree_centrality.values()))

    print("\n=== Computing risk-score COMPONENTS (weight-independent) ===")
    risk_df = compute_risk_scores(
        entity_ids=feats["entity_id"].tolist(),
        anomaly_percentile=anomaly_percentile,
        degree_centrality=degree_centrality,
        cluster_labels=cluster_labels,
        flagged_neighbor_fraction=flagged_neighbor_fraction,
        centrality_reference=centrality_reference,
        weights=WEIGHTS,  # placeholder combination just to get components; recombined below
    )
    risk_df = risk_df.merge(feats[["entity_id", "is_risky"]], on="entity_id")

    train_ids = set(train_df["entity_id"])
    train_mask = risk_df["entity_id"].isin(train_ids)

    print("\n=== Calibrating weights: comparing naive-correlation vs. logistic regression ===")
    naive_calibrated = calibrate_weights(risk_df[train_mask], risk_df.loc[train_mask, "is_risky"].astype(int).values)
    logistic_calibrated, logistic_info = calibrate_weights_logistic(
        risk_df[train_mask], risk_df.loc[train_mask, "is_risky"].astype(int).values
    )
    print("Default (hypothesis) weights:", WEIGHTS)
    print("Naive-correlation weights:   ", naive_calibrated, "(measured worse than default -- kept only for comparison)")
    print("Logistic-regression weights: ", logistic_calibrated)
    print("Logistic raw coefficients/intercept:", logistic_info)

    risk_df["risk_score"] = recombine_risk_score(risk_df, logistic_calibrated)

    train_risk_scores = risk_df.loc[train_mask, "risk_score"].values
    thresholds = derive_thresholds(train_risk_scores)
    risk_df["risk_level"] = risk_df["risk_score"].apply(lambda s: assign_risk_level(s, thresholds))
    print("\nThresholds derived from TRAIN population:", json.dumps(thresholds, indent=2))
    print(risk_df["risk_level"].value_counts().to_dict())

    print("\n=== Validating on TEST: calibrated-combined vs. anomaly-alone vs. default-weight-combined ===")
    test_ids = set(test_df["entity_id"])
    test_mask = risk_df["entity_id"].isin(test_ids)
    y_test = risk_df.loc[test_mask, "is_risky"].astype(int).values
    calibrated_scores = risk_df.loc[test_mask, "risk_score"].values
    anomaly_only_scores = risk_df.loc[test_mask, "anomaly_component"].values
    default_weight_scores = recombine_risk_score(risk_df[test_mask], WEIGHTS)

    comparison = {
        "logistic_combined_pr_auc": round(float(average_precision_score(y_test, calibrated_scores)), 4),
        "anomaly_only_pr_auc": round(float(average_precision_score(y_test, anomaly_only_scores)), 4),
        "default_weight_combined_pr_auc": round(float(average_precision_score(y_test, default_weight_scores)), 4),
        "naive_correlation_combined_pr_auc": round(float(average_precision_score(y_test, recombine_risk_score(risk_df[test_mask], naive_calibrated))), 4),
        "logistic_combined_precision_at_50": round(precision_at_k(y_test, calibrated_scores, 50), 4),
        "anomaly_only_precision_at_50": round(precision_at_k(y_test, anomaly_only_scores, 50), 4),
        "default_weight_precision_at_50": round(precision_at_k(y_test, default_weight_scores, 50), 4),
        "logistic_combined_precision_at_100": round(precision_at_k(y_test, calibrated_scores, 100), 4),
        "anomaly_only_precision_at_100": round(precision_at_k(y_test, anomaly_only_scores, 100), 4),
        "default_weight_precision_at_100": round(precision_at_k(y_test, default_weight_scores, 100), 4),
    }
    print(json.dumps(comparison, indent=2))

    print("\n=== Generating explanations: top 10 highest-risk entities ===")
    baselines = compute_feature_baselines(train_df, ALL_FEATURE_COLUMNS)
    feats_indexed = feats.set_index("entity_id")

    def n_neighbors(entity_id):
        return len(list(H.neighbors(entity_id))) if entity_id in H else 0

    def n_flagged(entity_id):
        frac = flagged_neighbor_fraction.get(entity_id, 0.0)
        return round(frac * n_neighbors(entity_id))

    cluster_sizes = clusters_df["behavioral_cluster"].value_counts().to_dict()

    top10 = risk_df.sort_values("risk_score", ascending=False).head(10)
    explanations = []
    for _, row in top10.iterrows():
        eid = row["entity_id"]
        cl = cluster_labels.get(eid, -1)
        exp = generate_explanation(
            entity_id=eid, risk_score=row["risk_score"], risk_level=row["risk_level"],
            entity_features_row=feats_indexed.loc[eid], baselines=baselines,
            anomaly_percentile=row["anomaly_component"], centrality_percentile=row["graph_centrality_component"],
            n_flagged_neighbors=n_flagged(eid), n_total_neighbors=n_neighbors(eid),
            is_cluster_outlier=(cl == -1), cluster_size=cluster_sizes.get(cl) if cl != -1 else None,
        )
        explanations.append(exp.to_dict())

    print(f"Top entity ({top10.iloc[0]['entity_id'][:16]}...): "
          f"{len(explanations[0]['evidence'])} evidence items, risk_score={explanations[0]['risk_score']}")
    for item in explanations[0]["evidence"]:
        print(f"  [{item['category']}] {item['statement']}")

    print("\n=== Sanity check: does a LOW-scoring normal entity get padded evidence? ===")
    lowest = risk_df.sort_values("risk_score", ascending=True).iloc[0]
    eid = lowest["entity_id"]
    cl = cluster_labels.get(eid, -1)
    low_exp = generate_explanation(
        entity_id=eid, risk_score=lowest["risk_score"], risk_level=lowest["risk_level"],
        entity_features_row=feats_indexed.loc[eid], baselines=baselines,
        anomaly_percentile=lowest["anomaly_component"], centrality_percentile=lowest["graph_centrality_component"],
        n_flagged_neighbors=n_flagged(eid), n_total_neighbors=n_neighbors(eid),
        is_cluster_outlier=(cl == -1), cluster_size=cluster_sizes.get(cl) if cl != -1 else None,
    )
    print(f"Lowest-risk entity: {len(low_exp.items)} evidence items (should be few/zero, NOT padded to match top entity's count)")
    for item in low_exp.items:
        print(f"  [{item.category}] {item.statement}")

    print("\nSaving outputs...")
    risk_df.to_csv(os.path.join(proc_dir, "risk_scores.csv"), index=False)
    with open(os.path.join(proc_dir, "risk_evaluation.json"), "w") as f:
        json.dump({
            "default_hypothesis_weights": WEIGHTS,
            "naive_correlation_weights_worse_kept_for_comparison": naive_calibrated,
            "logistic_regression_weights_used": logistic_calibrated,
            "logistic_regression_info": logistic_info,
            "thresholds": thresholds,
            "risk_level_counts": risk_df["risk_level"].value_counts().to_dict(),
            "ranking_comparison": comparison,
        }, f, indent=2)
    with open(os.path.join(proc_dir, "sample_explanations.json"), "w") as f:
        json.dump({"top_10_highest_risk": explanations, "lowest_risk_sanity_check": low_exp.to_dict()}, f, indent=2)

    print(f"\nWritten: {proc_dir}/risk_scores.csv, risk_evaluation.json, sample_explanations.json")


if __name__ == "__main__":
    main()

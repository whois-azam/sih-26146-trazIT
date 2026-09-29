"""
innovations/run_phase9.py

Runs and validates all four Phase 9 innovations against the real pipeline
outputs. The drift check specifically includes a NEGATIVE and POSITIVE
control -- comparing the reference against a population that should NOT
show drift, and one that deliberately SHOULD -- because a drift detector
that only ever says "no drift" hasn't actually been tested.
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "../features")
sys.path.insert(0, "../ml")
sys.path.insert(0, ".")

from entity_features import build_features
from divergence import compute_full_divergence_report
from temporal_rescoring import flag_behavior_change
from alert_chains import build_chain_alerts
from drift_check import check_drift

DATA_RAW = "../data/raw"
DATA_PROC = "../data/processed"
ML_ARTIFACTS = "../ml/artifacts"


def main():
    print("Loading data...")
    risk_df = pd.read_csv(os.path.join(DATA_PROC, "risk_scores.csv"))
    tx_df = pd.read_csv(os.path.join(DATA_RAW, "bitcoin_dataset.csv"))
    tx_df["timestamp"] = pd.to_datetime(tx_df["timestamp"], utc=True)
    feats = build_features(tx_df)

    with open(os.path.join(DATA_RAW, "ground_truth.json")) as f:
        ground_truth = json.load(f)
    feats["is_risky"] = feats["entity_id"].map(lambda a: ground_truth.get(a, {}).get("is_risky", False))

    clusters_df = pd.read_csv(os.path.join(DATA_PROC, "behavioral_clusters.csv"))
    cluster_labels = dict(zip(clusters_df["entity_id"], clusters_df["behavioral_cluster"]))
    with open(os.path.join(DATA_PROC, "graph_communities.json")) as f:
        communities = json.load(f)

    print("\n=== Innovation #1: Bidirectional cluster/community divergence ===")
    div_report = compute_full_divergence_report(cluster_labels, communities, risk_df)
    print(f"Type A (graph-connected, behaviorally different): {div_report['type_a_count']} entities, "
          f"{div_report['type_a_high_risk_count']} at HIGH/CRITICAL")
    print(f"Type B (behaviorally similar, graph-disconnected): {div_report['type_b_count']} entities, "
          f"{div_report['type_b_high_risk_count']} at HIGH/CRITICAL")

    print("\n=== Innovation #2: Temporal behavior-change flagging ===")
    behavior_changes = flag_behavior_change(feats, ratio_threshold=3.0)
    print(f"{len(behavior_changes)} entities flagged for sudden behavior change (>=3x own baseline)")
    if len(behavior_changes):
        # cross-check: are behavior-change flags correlated with is_risky more than base rate?
        flagged_ids = set(behavior_changes["entity_id"])
        flagged_risky_rate = feats[feats["entity_id"].isin(flagged_ids)]["is_risky"].mean()
        base_rate = feats["is_risky"].mean()
        print(f"Risky rate among flagged: {flagged_risky_rate:.4f} vs population base rate: {base_rate:.4f} "
              f"({flagged_risky_rate/base_rate:.1f}x enrichment)")
        print(behavior_changes.head(5).to_string())

    print("\n=== Innovation #3: Chain/ring-level alerts ===")
    chain_alerts = build_chain_alerts(communities, risk_df, min_members_flagged=2)
    print(f"{len(chain_alerts)} chain alerts generated (communities with >=2 HIGH/CRITICAL members)")
    if len(chain_alerts):
        print(chain_alerts.drop(columns=["member_entity_ids"]).head(5).to_string())

    print("\n=== Innovation #4: Drift check (with negative + positive control) ===")
    preprocessor = joblib.load(os.path.join(ML_ARTIFACTS, "preprocessor.joblib"))
    model = joblib.load(os.path.join(ML_ARTIFACTS, "isolation_forest.joblib"))
    with open(os.path.join(ML_ARTIFACTS, "feature_columns.json")) as f:
        cols = json.load(f)
    reference = np.load(os.path.join(ML_ARTIFACTS, "score_reference.npy"))

    all_cols = cols["log_columns"] + cols["other_columns"]

    # NEGATIVE CONTROL: a random subsample of ALL entities -- should show NO drift
    normal_sample = feats.sample(n=2000, random_state=1)
    X_normal = preprocessor.transform(normal_sample[all_cols])
    normal_scores = -model.decision_function(X_normal)
    result_negative = check_drift(reference, normal_scores)
    print("Negative control (random population sample) -- expect NO drift:")
    print(f"  drift_detected={result_negative['drift_detected']}, p={result_negative['p_value']}, "
          f"ks={result_negative['ks_statistic']}")

    # POSITIVE CONTROL: ONLY risky entities -- should show drift (deliberately shifted population)
    risky_only = feats[feats["is_risky"]]
    X_risky = preprocessor.transform(risky_only[all_cols])
    risky_scores = -model.decision_function(X_risky)
    result_positive = check_drift(reference, risky_scores)
    print("Positive control (risky-only subpopulation) -- expect DRIFT:")
    print(f"  drift_detected={result_positive['drift_detected']}, p={result_positive['p_value']}, "
          f"ks={result_positive['ks_statistic']}")

    print("\nSaving outputs...")
    os.makedirs(DATA_PROC, exist_ok=True)
    div_report["type_a_entities"].to_csv(os.path.join(DATA_PROC, "divergence_type_a.csv"), index=False)
    div_report["type_b_entities"].to_csv(os.path.join(DATA_PROC, "divergence_type_b.csv"), index=False)
    behavior_changes.to_csv(os.path.join(DATA_PROC, "behavior_change_alerts.csv"), index=False)
    chain_alerts.to_csv(os.path.join(DATA_PROC, "chain_alerts.csv"), index=False)

    with open(os.path.join(DATA_PROC, "innovation_summary.json"), "w") as f:
        json.dump({
            "divergence_type_a_count": div_report["type_a_count"],
            "divergence_type_a_high_risk": div_report["type_a_high_risk_count"],
            "divergence_type_b_count": div_report["type_b_count"],
            "divergence_type_b_high_risk": div_report["type_b_high_risk_count"],
            "behavior_change_flags": len(behavior_changes),
            "behavior_change_risky_enrichment": round(float(flagged_risky_rate / base_rate), 2) if len(behavior_changes) else None,
            "chain_alerts": len(chain_alerts),
            "drift_check_negative_control": result_negative,
            "drift_check_positive_control": result_positive,
        }, f, indent=2)

    print(f"\nWritten: {DATA_PROC}/divergence_type_a.csv, divergence_type_b.csv, "
          f"behavior_change_alerts.csv, chain_alerts.csv, innovation_summary.json")


if __name__ == "__main__":
    main()

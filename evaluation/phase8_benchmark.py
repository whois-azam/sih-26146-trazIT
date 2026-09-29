"""
evaluation/benchmark.py

Phase 8 -- FRESH, real timing measurements for the full pipeline at 100k-row
scale (audit Section 15), run now rather than copy-pasted from earlier
console output during development.
"""

from __future__ import annotations

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "../features")
sys.path.insert(0, "../ingestion")
sys.path.insert(0, "../ml")
sys.path.insert(0, "../graph")
sys.path.insert(0, "../scoring")

DATA_RAW = "../data/raw"
DATA_PROC = "../data/processed"
ML_ARTIFACTS = "../ml/artifacts"


def timed(label: str, fn, results: dict):
    t0 = time.time()
    out = fn()
    results[label] = round(time.time() - t0, 3)
    return out


def run_benchmark() -> dict:
    results: dict = {}

    from loaders import load_raw_dataset, validate_and_normalize
    df = timed("ingestion_load", lambda: load_raw_dataset(os.path.join(DATA_RAW, "bitcoin_dataset.csv")), results)
    clean_df, report = timed("ingestion_validate", lambda: validate_and_normalize(df), results)
    clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], utc=True)

    from entity_features import build_features
    feats = timed("feature_engineering_full", lambda: build_features(clean_df), results)

    direct_corr = pd.read_csv(os.path.join(DATA_PROC, "correlations_direct.csv"))
    fallback_corr = pd.read_csv(os.path.join(DATA_PROC, "correlations_fallback.csv"))
    from build import build_graph, wallet_projection
    G = timed("graph_build", lambda: build_graph(clean_df, direct_corr, fallback_corr), results)
    H = timed("wallet_projection", lambda: wallet_projection(G), results)

    from anomaly import CalibratedAnomalyDetector
    detector = CalibratedAnomalyDetector.load(
        os.path.join(ML_ARTIFACTS, "isolation_forest.joblib"), os.path.join(ML_ARTIFACTS, "score_reference.npy"))
    preprocessor = joblib.load(os.path.join(ML_ARTIFACTS, "preprocessor.joblib"))
    with open(os.path.join(ML_ARTIFACTS, "feature_columns.json")) as f:
        cols = json.load(f)

    X_full = timed("preprocessing_transform_full", lambda: preprocessor.transform(feats[cols["log_columns"] + cols["other_columns"]]), results)
    scores = timed("anomaly_scoring_full_population", lambda: detector.score(X_full), results)

    from clustering import fit_behavioral_clustering
    cluster_result = timed("behavioral_clustering_full", lambda: fit_behavioral_clustering(X_full, min_samples=10), results)

    from risk import compute_flagged_neighbor_fraction
    anomaly_by_entity = dict(zip(feats["entity_id"], scores["calibrated_percentile"]))
    timed("flagged_neighbor_computation", lambda: compute_flagged_neighbor_fraction(feats["entity_id"].tolist(), anomaly_by_entity, H), results)

    results["_meta"] = {
        "input_rows": len(clean_df), "n_entities": len(feats), "n_graph_nodes": G.number_of_nodes(),
        "n_graph_edges": G.number_of_edges(), "n_wallet_projection_edges": H.number_of_edges(),
    }
    results["total_pipeline_seconds_excluding_betweenness"] = round(
        sum(v for k, v in results.items() if k != "_meta" and isinstance(v, (int, float))), 2
    )
    return results

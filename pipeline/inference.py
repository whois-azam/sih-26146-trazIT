"""
pipeline/inference.py

Rewritten architecture: NOTHING loads the raw 100k-row training dataset or
any precomputed dashboard artifact derived from it. Only the small, fast
TRAINED MODEL files load (weights + calibration reference + config) --
these are the model, not "training data" in the sense of dashboard
content, and loading them is instant (<1s: a few KB-to-MB of joblib/json/npy).

Everything else -- graph, behavioral clusters, graph communities, feature
baselines for evidence -- is computed FRESH and ONLY from whatever the user
uploads, each time they upload. This means:

  - anomaly_component: scored via the frozen IsolationForest + frozen
    calibration reference (trained once on the 100k dataset back in
    Phase 5) -- this is the actual "train once, predict many times"
    behavior. Fully valid regardless of upload size.
  - graph_centrality_component: computed on the uploaded-only mini-graph,
    percentile-ranked WITHIN that same uploaded population (no external
    reference population used -- fully self-contained).
  - behavioral clustering: a FRESH DBSCAN/HDBSCAN fit on just the uploaded
    entities' features. Weaker statistically on a small sample than the
    original 20,000-entity fit, but self-contained and honest about it.
  - graph communities: FRESH Louvain community detection on the
    uploaded-only mini-graph.
  - evidence baselines ("X standard deviations from the population..."):
    computed from the UPLOADED entities' own distribution, not the
    original training population. The evidence text says so explicitly.
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
for _rel in ("ingestion", "features", "ml", "graph", "scoring", "explainability", "innovations"):
    sys.path.insert(0, os.path.join(_HERE, "..", _rel))

from loaders import load_raw_dataset, validate_and_normalize
from entity_features import build_features, ALL_FEATURE_COLUMNS
from anomaly import CalibratedAnomalyDetector
from build import build_graph, wallet_projection
from risk import percentile_rank_against_reference, assign_risk_level
from evidence import compute_feature_baselines, generate_explanation
from clustering import fit_behavioral_clustering
from algorithms import detect_communities
from drift_check import check_drift

ML_ARTIFACTS = os.path.join(_HERE, "..", "ml", "artifacts")
DATA_PROC = os.path.join(_HERE, "..", "data", "processed")  # ONLY used for risk_evaluation.json (small config: weights/thresholds)


class InferenceContext:
    """Loads ONLY the small trained-model artifacts. No dataset, no
    precomputed dashboard content. This is what's safe (and fast, <1s) to
    load at app boot."""

    def __init__(self):
        self.detector = CalibratedAnomalyDetector.load(
            os.path.join(ML_ARTIFACTS, "isolation_forest.joblib"),
            os.path.join(ML_ARTIFACTS, "score_reference.npy"),
        )
        self.preprocessor = joblib.load(os.path.join(ML_ARTIFACTS, "preprocessor.joblib"))
        with open(os.path.join(ML_ARTIFACTS, "feature_columns.json")) as f:
            cols = json.load(f)
        self.log_columns = cols["log_columns"]
        self.other_columns = cols["other_columns"]
        self.all_cols = self.log_columns + self.other_columns

        # weights + thresholds are trained-model CONFIGURATION (a handful of
        # scalars), not dataset content -- fine to load at boot.
        with open(os.path.join(DATA_PROC, "risk_evaluation.json")) as f:
            risk_eval = json.load(f)
        self.weights = risk_eval["logistic_regression_weights_used"]
        self.thresholds = risk_eval["thresholds"]


def run_inference(uploaded_path: str, ctx: InferenceContext) -> dict:
    """Full self-contained pipeline for one uploaded file: ingest ->
    validate -> features -> score (frozen model) -> mini-graph -> fresh
    clustering -> fresh communities -> risk -> ready for explanations.
    Nothing here touches the original training dataset."""
    raw_df = load_raw_dataset(uploaded_path)
    clean_df, validation_report = validate_and_normalize(raw_df)

    if len(clean_df) == 0:
        return {
            "success": False,
            "error": "No valid rows after validation.",
            "validation_report": validation_report.summary(),
        }

    feats = build_features(clean_df)

    X = ctx.preprocessor.transform(feats[ctx.all_cols])
    anomaly_result = ctx.detector.score(X)
    anomaly_percentile = anomaly_result["calibrated_percentile"]
    anomaly_by_entity = dict(zip(feats["entity_id"], anomaly_percentile))

    # drift check: does this upload resemble what the model was trained on?
    # Uses only the small frozen reference array already in memory -- not
    # the raw training dataset.
    drift_result = None
    if len(anomaly_result["raw_score"]) >= 5:
        drift_result = check_drift(ctx.detector.reference_scores_, anomaly_result["raw_score"])

    # mini-graph from JUST the uploaded rows
    empty_corr = pd.DataFrame(columns=["wallet", "ip", "role", "txid", "timestamp", "correlation_type", "confidence"])
    G = build_graph(clean_df, empty_corr, None)
    H = wallet_projection(G)

    # centrality: self-referential percentile within THIS upload's own
    # population -- no external reference population needed.
    import networkx as nx
    degree_centrality = nx.degree_centrality(H) if H.number_of_nodes() > 1 else {}
    centrality_values = np.array([degree_centrality.get(e, 0.0) for e in feats["entity_id"]])
    centrality_reference = centrality_values if len(centrality_values) else np.array([0.0])
    centrality_pct = percentile_rank_against_reference(centrality_values, centrality_reference)

    flagged_neighbor_fraction = {}
    for eid in feats["entity_id"]:
        neighbors = list(H.neighbors(eid)) if eid in H else []
        if not neighbors:
            flagged_neighbor_fraction[eid] = 0.0
            continue
        flagged = sum(1 for n in neighbors if anomaly_by_entity.get(n, 0.0) >= 0.90)
        flagged_neighbor_fraction[eid] = flagged / len(neighbors)
    neighbor_frac = np.array([flagged_neighbor_fraction.get(e, 0.0) for e in feats["entity_id"]])

    # FRESH behavioral clustering on just the uploaded entities. Small-
    # sample caveat surfaced in the returned dict, not hidden.
    cluster_labels: dict = {}
    n_clusters_found = 0
    clustering_algorithm = None
    if len(feats) >= 10:
        min_samples = max(3, min(10, len(feats) // 5))
        cluster_result = fit_behavioral_clustering(X, min_samples=min_samples)
        labels = cluster_result["labels"]
        cluster_labels = dict(zip(feats["entity_id"], labels))
        n_clusters_found = cluster_result["evaluation"]["n_clusters"]
        clustering_algorithm = cluster_result["algorithm"]
    else:
        cluster_labels = {e: -1 for e in feats["entity_id"]}
    cluster_outlier = np.array([1.0 if cluster_labels.get(e, -1) == -1 else 0.0 for e in feats["entity_id"]])

    # FRESH graph community detection on the uploaded-only mini-graph.
    communities: dict = {}
    n_communities_found = 0
    if H.number_of_nodes() >= 3 and H.number_of_edges() >= 1:
        try:
            comm_result = detect_communities(H)
            communities = comm_result["wallet_to_community"]
            n_communities_found = comm_result["n_communities"]
        except Exception:
            communities = {}

    risk = (
        ctx.weights["anomaly"] * anomaly_percentile
        + ctx.weights["graph_centrality"] * centrality_pct
        + ctx.weights["cluster_outlier"] * cluster_outlier
        + ctx.weights["flagged_neighbors"] * neighbor_frac
    )
    risk_score = np.clip(risk, 0, 1) * 100

    risk_df = pd.DataFrame({
        "entity_id": feats["entity_id"],
        "anomaly_component": anomaly_percentile,
        "graph_centrality_component": centrality_pct,
        "cluster_outlier_component": cluster_outlier,
        "flagged_neighbor_component": neighbor_frac,
        "risk_score": risk_score,
    })
    risk_df["risk_level"] = risk_df["risk_score"].apply(lambda s: assign_risk_level(s, ctx.thresholds))
    risk_df["behavioral_cluster"] = risk_df["entity_id"].map(cluster_labels)
    risk_df["graph_community"] = risk_df["entity_id"].map(communities)

    # Country: an entity (wallet) can appear across multiple transactions,
    # each with its own resolved geo_country (different src_ip per tx).
    # Track the full set for filtering AND a single "primary" (most
    # frequent) value for display, rather than picking just one arbitrarily.
    import json as _json
    txid_to_country = dict(zip(clean_df["txid"], clean_df["geo_country"]))
    countries_by_entity = {}
    for _, frow in feats.iterrows():
        txids = _json.loads(frow["transaction_txids"])
        countries = [txid_to_country.get(t) for t in txids if txid_to_country.get(t)]
        countries_by_entity[frow["entity_id"]] = countries

    def _primary_country(countries):
        if not countries:
            return "UNKNOWN"
        return pd.Series(countries).mode().iloc[0]

    risk_df["countries"] = risk_df["entity_id"].map(lambda e: sorted(set(countries_by_entity.get(e, []))) or ["UNKNOWN"])
    risk_df["primary_country"] = risk_df["entity_id"].map(lambda e: _primary_country(countries_by_entity.get(e, [])))

    # baselines computed from THIS upload's own population -- evidence text
    # says "vs. this file's population", not the original training set.
    baselines = compute_feature_baselines(feats, ALL_FEATURE_COLUMNS) if len(feats) >= 5 else {}

    return {
        "success": True,
        "validation_report": validation_report.summary(),
        "n_rows_uploaded": len(raw_df),
        "n_rows_valid": len(clean_df),
        "n_entities": len(feats),
        "risk_df": risk_df,
        "feats": feats,
        "clean_df": clean_df,
        "G": G,
        "H": H,
        "cluster_labels": cluster_labels,
        "n_clusters_found": n_clusters_found,
        "clustering_algorithm": clustering_algorithm,
        "communities": communities,
        "n_communities_found": n_communities_found,
        "baselines": baselines,
        "baselines_source": "this upload's own population (not the original training set)",
        "drift_result": drift_result,
    }


def get_upload_entity_evidence(entity_id: str, result: dict) -> dict:
    risk_row = result["risk_df"][result["risk_df"]["entity_id"] == entity_id]
    if risk_row.empty:
        return {"entity_id": entity_id, "risk_score": 0.0, "risk_level": "LOW", "n_evidence_items": 0, "evidence": []}
    risk_row = risk_row.iloc[0]

    H = result["H"]
    n_neighbors = len(list(H.neighbors(entity_id))) if entity_id in H else 0
    n_flagged = round(risk_row["flagged_neighbor_component"] * n_neighbors)

    cluster = result["cluster_labels"].get(entity_id, -1)
    cluster_size = list(result["cluster_labels"].values()).count(cluster) if cluster != -1 else None

    if not result["baselines"]:
        return {
            "entity_id": entity_id, "risk_score": float(risk_row["risk_score"]), "risk_level": risk_row["risk_level"],
            "n_evidence_items": 0, "evidence": [],
            "note": "Upload too small for meaningful feature-deviation baselines (need 5+ entities).",
        }

    feats_indexed = result["feats"].set_index("entity_id")
    exp = generate_explanation(
        entity_id=entity_id, risk_score=risk_row["risk_score"], risk_level=risk_row["risk_level"],
        entity_features_row=feats_indexed.loc[entity_id], baselines=result["baselines"],
        anomaly_percentile=risk_row["anomaly_component"], centrality_percentile=risk_row["graph_centrality_component"],
        n_flagged_neighbors=n_flagged, n_total_neighbors=n_neighbors,
        is_cluster_outlier=(cluster == -1), cluster_size=cluster_size,
    )
    return exp.to_dict()

"""
dashboard/data_loader.py

Pure functions (no Streamlit dependency) so the dashboard's data logic can
be tested directly before trusting it inside an interactive app. app.py
imports these and only adds caching/rendering on top.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, "../features")
sys.path.insert(0, "../ml")
sys.path.insert(0, "../graph")
sys.path.insert(0, "../explainability")
sys.path.insert(0, "../scoring")

from entity_features import build_features, ALL_FEATURE_COLUMNS
from build import build_graph, wallet_projection
from evidence import compute_feature_baselines, generate_explanation

DATA_RAW = "../data/raw"
DATA_PROC = "../data/processed"


def load_everything() -> dict:
    """Single entrypoint loading every artifact the dashboard needs."""
    risk_df = pd.read_csv(os.path.join(DATA_PROC, "risk_scores.csv"))

    tx_df = pd.read_csv(os.path.join(DATA_RAW, "bitcoin_dataset.csv"))
    tx_df["timestamp"] = pd.to_datetime(tx_df["timestamp"], utc=True)

    feats = build_features(tx_df)
    with open(os.path.join(DATA_RAW, "ground_truth.json")) as f:
        ground_truth = json.load(f)
    # DEV-ONLY convenience field for this synthetic dataset -- clearly
    # labeled in the UI, never a real system's source of truth.
    feats["_dev_known_scenario"] = feats["entity_id"].map(
        lambda a: ground_truth.get(a, {}).get("scenario_type"))

    direct_corr = pd.read_csv(os.path.join(DATA_PROC, "correlations_direct.csv"))
    fallback_corr = pd.read_csv(os.path.join(DATA_PROC, "correlations_fallback.csv"))
    G = build_graph(tx_df, direct_corr, fallback_corr)
    H = wallet_projection(G)

    clusters_df = pd.read_csv(os.path.join(DATA_PROC, "behavioral_clusters.csv"))
    cluster_labels = dict(zip(clusters_df["entity_id"], clusters_df["behavioral_cluster"]))
    cluster_sizes = clusters_df["behavioral_cluster"].value_counts().to_dict()

    with open(os.path.join(DATA_PROC, "graph_centrality.json")) as f:
        centrality = json.load(f)
    degree_centrality = centrality["degree_centrality"]

    with open(os.path.join(DATA_PROC, "graph_communities.json")) as f:
        communities = json.load(f)

    risk_df["is_risky_flagged"] = risk_df["entity_id"].isin(
        [a for a, v in ground_truth.items() if v.get("is_risky")])

    # reproduce the SAME train split used in Phase 5/6 (same seed) so
    # baselines are calibrated on the same reference population.
    is_risky_arr = risk_df.set_index("entity_id")["is_risky_flagged"].reindex(feats["entity_id"]).values
    train_df, _ = train_test_split(feats, test_size=0.30, stratify=is_risky_arr, random_state=42)
    baselines = compute_feature_baselines(train_df, ALL_FEATURE_COLUMNS)

    return {
        "risk_df": risk_df, "feats": feats, "tx_df": tx_df, "G": G, "H": H,
        "cluster_labels": cluster_labels, "cluster_sizes": cluster_sizes,
        "degree_centrality": degree_centrality, "communities": communities,
        "baselines": baselines,
    }


def get_entity_transactions(entity_id: str, feats: pd.DataFrame, tx_df: pd.DataFrame) -> pd.DataFrame:
    """Fast lookup via the precomputed transaction_txids column (Phase 4),
    NOT a row-wise JSON-parse scan of the full 100k-row table per click."""
    row = feats.loc[feats["entity_id"] == entity_id]
    if row.empty:
        return tx_df.iloc[0:0]
    txids = set(json.loads(row.iloc[0]["transaction_txids"]))
    return tx_df[tx_df["txid"].isin(txids)].sort_values("timestamp")


def get_entity_evidence(entity_id: str, ctx: dict) -> dict:
    risk_row = ctx["risk_df"].loc[ctx["risk_df"]["entity_id"] == entity_id]
    if risk_row.empty:
        return {"entity_id": entity_id, "risk_score": 0.0, "risk_level": "LOW", "n_evidence_items": 0, "evidence": []}
    risk_row = risk_row.iloc[0]

    H = ctx["H"]
    n_neighbors = len(list(H.neighbors(entity_id))) if entity_id in H else 0
    n_flagged = round(risk_row["flagged_neighbor_component"] * n_neighbors)

    cl = ctx["cluster_labels"].get(entity_id, -1)
    feats_indexed = ctx["feats"].set_index("entity_id")
    if entity_id not in feats_indexed.index:
        return {"entity_id": entity_id, "risk_score": float(risk_row["risk_score"]), "risk_level": risk_row["risk_level"], "n_evidence_items": 0, "evidence": []}

    exp = generate_explanation(
        entity_id=entity_id, risk_score=risk_row["risk_score"], risk_level=risk_row["risk_level"],
        entity_features_row=feats_indexed.loc[entity_id], baselines=ctx["baselines"],
        anomaly_percentile=risk_row["anomaly_component"], centrality_percentile=risk_row["graph_centrality_component"],
        n_flagged_neighbors=n_flagged, n_total_neighbors=n_neighbors,
        is_cluster_outlier=(cl == -1), cluster_size=ctx["cluster_sizes"].get(cl) if cl != -1 else None,
    )
    return exp.to_dict()


def get_neighborhood_subgraph(entity_id: str, H, ctx: dict, max_neighbors: int = 20) -> tuple[list, list]:
    """Returns (nodes, edges) for the entity + its direct graph neighbors,
    each node annotated with its risk_score/risk_level for coloring."""
    if entity_id not in H:
        return [{"id": entity_id, "risk_score": 0.0, "risk_level": "LOW", "is_center": True}], []

    risk_lookup = ctx["risk_df"].set_index("entity_id")[["risk_score", "risk_level"]].to_dict("index")
    neighbors = list(H.neighbors(entity_id))[:max_neighbors]

    nodes = [{"id": entity_id, **risk_lookup.get(entity_id, {"risk_score": 0.0, "risk_level": "LOW"}), "is_center": True}]
    for n in neighbors:
        nodes.append({"id": n, **risk_lookup.get(n, {"risk_score": 0.0, "risk_level": "LOW"}), "is_center": False})

    edges = []
    for n in neighbors:
        weight = H[entity_id][n].get("weight", 1)
        edges.append({"source": entity_id, "target": n, "weight": weight})

    return nodes, edges


def compute_cluster_community_divergence(ctx: dict) -> pd.DataFrame:
    """Innovation #1 (audit Section 13): entities that are graph-connected
    (same community) but behaviorally different (different cluster), or
    behaviorally similar (same cluster) but graph-disconnected (different
    community) -- flagged as investigatively interesting divergence."""
    risk_df = ctx["risk_df"]
    rows = []
    for eid in risk_df["entity_id"]:
        cluster = ctx["cluster_labels"].get(eid, -1)
        community = ctx["communities"].get(eid)
        rows.append({"entity_id": eid, "behavioral_cluster": cluster, "graph_community": community})
    df = pd.DataFrame(rows).merge(risk_df[["entity_id", "risk_score", "risk_level"]], on="entity_id")
    return df

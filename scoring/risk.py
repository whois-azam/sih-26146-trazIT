"""
scoring/risk.py

Phase 6 -- ONE risk formula (audit Section 11), reconciling what the
prototype had as two divergent, disagreeing implementations.

    risk = 100 * clip(
        0.45 * anomaly_score_calibrated          # Stage 2 (Phase 5), frozen-reference
      + 0.20 * graph_centrality_percentile        # Stage 3b (Phase 3), real graph object
      + 0.15 * behavioral_cluster_outlier_flag    # Stage 3a (Phase 5), noise = elevated
      + 0.20 * relationship_to_flagged_entities   # graph-propagated: how many of THIS
                                                    # entity's direct graph neighbors are
                                                    # themselves already flagged
    , 0, 1)

Every component is stored alongside the final score -- "why is this a 78?"
has a mechanical answer (Section 11), not just a number.

Thresholds (CRITICAL/HIGH/MEDIUM/LOW) are derived from the TRAIN population's
score distribution, not hand-picked round numbers -- consistent with the
frozen-reference philosophy used everywhere else in this pipeline.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "../features")
sys.path.insert(0, "../ml")
sys.path.insert(0, "../graph")

WEIGHTS = {
    "anomaly": 0.45,
    "graph_centrality": 0.20,
    "cluster_outlier": 0.15,
    "flagged_neighbors": 0.20,
}

NEIGHBOR_FLAG_THRESHOLD = 0.90  # an entity counts as "flagged" for neighbor-propagation
                                  # purposes if its own anomaly percentile >= this


def percentile_rank_against_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Same frozen-reference philosophy as ml/anomaly.py: percentile of each
    value against a FIXED sorted reference array, not against whatever's in
    the current batch."""
    sorted_ref = np.sort(reference)
    return np.searchsorted(sorted_ref, values, side="right") / len(sorted_ref)


def compute_flagged_neighbor_fraction(
    wallet_ids: list[str],
    anomaly_percentile_by_wallet: dict[str, float],
    H,  # networkx wallet-projection graph
    threshold: float = NEIGHBOR_FLAG_THRESHOLD,
) -> dict[str, float]:
    """For each wallet, the fraction of its DIRECT graph neighbors (transfer
    edges) that are themselves anomaly-flagged. A wallet with no neighbors
    gets 0.0, not NaN -- absence of graph context isn't evidence of risk."""
    result = {}
    for w in wallet_ids:
        if w not in H:
            result[w] = 0.0
            continue
        neighbors = list(H.neighbors(w))
        if not neighbors:
            result[w] = 0.0
            continue
        flagged = sum(1 for n in neighbors if anomaly_percentile_by_wallet.get(n, 0.0) >= threshold)
        result[w] = flagged / len(neighbors)
    return result


def compute_risk_scores(
    entity_ids: list[str],
    anomaly_percentile: np.ndarray,
    degree_centrality: dict[str, float],
    cluster_labels: dict[str, int],
    flagged_neighbor_fraction: dict[str, float],
    centrality_reference: np.ndarray,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    weights = weights or WEIGHTS
    centrality_values = np.array([degree_centrality.get(e, 0.0) for e in entity_ids])
    centrality_pct = percentile_rank_against_reference(centrality_values, centrality_reference)

    cluster_outlier = np.array([1.0 if cluster_labels.get(e, -1) == -1 else 0.0 for e in entity_ids])
    neighbor_frac = np.array([flagged_neighbor_fraction.get(e, 0.0) for e in entity_ids])

    risk = (
        weights["anomaly"] * anomaly_percentile
        + weights["graph_centrality"] * centrality_pct
        + weights["cluster_outlier"] * cluster_outlier
        + weights["flagged_neighbors"] * neighbor_frac
    )
    risk = np.clip(risk, 0, 1) * 100

    return pd.DataFrame({
        "entity_id": entity_ids,
        "anomaly_component": anomaly_percentile,
        "graph_centrality_component": centrality_pct,
        "cluster_outlier_component": cluster_outlier,
        "flagged_neighbor_component": neighbor_frac,
        "risk_score": risk,
    })


def derive_thresholds(train_risk_scores: np.ndarray) -> dict[str, float]:
    """CRITICAL/HIGH/MEDIUM cut points from the TRAIN population's own
    score distribution -- not hand-picked round numbers (Section 11)."""
    return {
        "critical": float(np.percentile(train_risk_scores, 99)),
        "high": float(np.percentile(train_risk_scores, 95)),
        "medium": float(np.percentile(train_risk_scores, 85)),
    }


def assign_risk_level(score: float, thresholds: dict[str, float]) -> str:
    if score >= thresholds["critical"]:
        return "CRITICAL"
    if score >= thresholds["high"]:
        return "HIGH"
    if score >= thresholds["medium"]:
        return "MEDIUM"
    return "LOW"


def recombine_risk_score(components_df: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    """Recombine already-computed components with a different weight set,
    without recomputing anomaly/graph/cluster scores from scratch."""
    risk = (
        weights["anomaly"] * components_df["anomaly_component"]
        + weights["graph_centrality"] * components_df["graph_centrality_component"]
        + weights["cluster_outlier"] * components_df["cluster_outlier_component"]
        + weights["flagged_neighbors"] * components_df["flagged_neighbor_component"]
    )
    return np.clip(risk, 0, 1).values * 100


def calibrate_weights(
    component_df: pd.DataFrame, is_risky: np.ndarray, floor: float = 0.05
) -> dict[str, float]:
    """DEPRECATED approach, kept for comparison only: naive marginal
    correlation of each component with ground truth, normalized to sum to
    1. Measured on real data to perform WORSE than even the untested
    default weights (test PR-AUC 0.51 vs 0.64 vs 0.79 for anomaly alone) --
    because it ignores redundancy between components and treats a coarse
    binary flag (cluster_outlier) as equally rankable as a continuous
    score, which collapses fine-grained ordering within the flagged group.
    Use calibrate_weights_logistic() instead."""
    components = ["anomaly", "graph_centrality", "cluster_outlier", "flagged_neighbors"]
    col_map = {
        "anomaly": "anomaly_component", "graph_centrality": "graph_centrality_component",
        "cluster_outlier": "cluster_outlier_component", "flagged_neighbors": "flagged_neighbor_component",
    }
    raw_corr = {}
    for c in components:
        corr = np.corrcoef(component_df[col_map[c]], is_risky)[0, 1]
        raw_corr[c] = max(abs(corr), floor) if not np.isnan(corr) else floor

    total = sum(raw_corr.values())
    return {c: round(v / total, 4) for c, v in raw_corr.items()}


def calibrate_weights_logistic(
    component_df: pd.DataFrame, is_risky: np.ndarray
) -> tuple[dict[str, float], dict]:
    """Calibrate weights via logistic regression on the four components,
    fit on TRAIN only. Unlike naive marginal correlation, this jointly
    optimizes the combination and correctly discounts redundant components
    instead of double-counting shared signal. Still fully interpretable:
    coefficients ARE the weights, no hidden nonlinearity.
    class_weight='balanced' compensates for risky entities being ~2% of
    the population."""
    from sklearn.linear_model import LogisticRegression

    components = ["anomaly", "graph_centrality", "cluster_outlier", "flagged_neighbors"]
    col_map = {
        "anomaly": "anomaly_component", "graph_centrality": "graph_centrality_component",
        "cluster_outlier": "cluster_outlier_component", "flagged_neighbors": "flagged_neighbor_component",
    }
    X = component_df[[col_map[c] for c in components]].values
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(X, is_risky)

    coefs = clf.coef_[0]
    # a negative coefficient would mean "higher component value -> lower
    # risk", which doesn't fit the additive risk-score framing -- clip at
    # 0 and surface the raw coefficient rather than silently flipping sign.
    clipped = np.clip(coefs, 0, None)
    total = clipped.sum() or 1.0
    weights = {c: round(float(w / total), 4) for c, w in zip(components, clipped)}
    info = {
        "raw_coefficients": {c: round(float(w), 4) for c, w in zip(components, coefs)},
        "intercept": round(float(clf.intercept_[0]), 4),
    }
    return weights, info

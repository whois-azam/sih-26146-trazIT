"""
ml/clustering.py

Phase 5, Stage 3a -- BEHAVIORAL clustering ("who acts alike"), distinct from
Phase 3's graph community detection ("who's actually connected"). Per the
audit (Section 6), these are kept as two separate, separately-reported
signals -- never merged into one cluster ID.

Tries DBSCAN first (audit: "keep the algorithm choice provisionally but
validate quality at scale; swap to HDBSCAN if it keeps collapsing" -- this
is exactly what the prototype never actually did).
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def suggest_dbscan_eps(X: np.ndarray, k: int = 10) -> float:
    """k-distance heuristic for a reasonable DBSCAN eps: the 'elbow' of the
    sorted k-th nearest neighbor distance curve. Avoids hand-picking eps
    blindly."""
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nn.kneighbors(X)
    kth_distances = np.sort(distances[:, -1])
    # simple elbow heuristic: 90th percentile of the k-distance curve
    return float(np.percentile(kth_distances, 90))


def fit_dbscan(X: np.ndarray, eps: float | None = None, min_samples: int = 10) -> dict:
    if eps is None:
        eps = suggest_dbscan_eps(X, k=min_samples)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return {"labels": labels, "eps_used": eps, "min_samples": min_samples, "algorithm": "dbscan"}


def fit_hdbscan(X: np.ndarray, min_cluster_size: int = 15) -> dict:
    import hdbscan
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(X)
    return {"labels": labels, "min_cluster_size": min_cluster_size, "algorithm": "hdbscan"}


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> dict:
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_fraction = float((labels == -1).mean())

    silhouette = None
    non_noise_mask = labels != -1
    if n_clusters >= 2 and non_noise_mask.sum() > n_clusters:
        try:
            silhouette = float(silhouette_score(X[non_noise_mask], labels[non_noise_mask]))
        except ValueError:
            silhouette = None

    return {
        "n_clusters": n_clusters,
        "noise_fraction": round(noise_fraction, 4),
        "silhouette_score": round(silhouette, 4) if silhouette is not None else None,
    }


def fit_hdbscan_adaptive(X: np.ndarray, n_samples: int, primary_min_cluster_size: int = 10) -> dict:
    """Try the standard min_cluster_size FIRST -- this is what already
    works well at large scale (verified: 20,000 entities -> 5 clusters,
    ~0% noise, using min_cluster_size=10). Only search smaller absolute
    values if that specific result collapses (0-1 clusters or >=90% noise),
    which is what actually happens on small uploads (measured: 300 entities
    with min_cluster_size=10 -> 0 clusters, 100% noise).

    This is deliberately narrow instead of a general search-and-rank
    strategy: ranking candidates by noise fraction, dominant-cluster
    fraction, AND silhouette score were all tried and each was fooled at
    one scale or the other -- a large, internally-cohesive dominant
    cluster can score well on any of those metrics while still being the
    same degenerate "everything in one blob" collapse. Never touching the
    already-good large-scale path sidesteps that problem entirely instead
    of trying to out-clever it with a better metric."""
    primary = fit_hdbscan(X, min_cluster_size=primary_min_cluster_size)
    primary_eval = evaluate_clustering(X, primary["labels"])
    if primary_eval["n_clusters"] >= 2 and primary_eval["noise_fraction"] < 0.9:
        return {**primary, "min_cluster_size_used": primary_min_cluster_size, "candidates_tried": [primary_min_cluster_size]}

    fallback_candidates = [8, 6, 5, 4, 3, 2]
    results = []
    for mcs in fallback_candidates:
        r = fit_hdbscan(X, min_cluster_size=mcs)
        e = evaluate_clustering(X, r["labels"])
        results.append((mcs, r, e))

    viable = [(mcs, r, e) for mcs, r, e in results if e["n_clusters"] >= 2 and e["noise_fraction"] < 0.5]
    if viable:
        # among genuinely viable small-sample fits, prefer more clusters
        # (finer real structure) over fewer, breaking ties by lower noise
        mcs, r, e = max(viable, key=lambda t: (t[2]["n_clusters"], -t[2]["noise_fraction"]))
    else:
        mcs, r, e = min(results, key=lambda t: t[2]["noise_fraction"])

    return {**r, "min_cluster_size_used": mcs, "candidates_tried": [primary_min_cluster_size] + fallback_candidates}


def fit_behavioral_clustering(X: np.ndarray, min_samples: int = 10) -> dict:
    """DBSCAN first; automatically falls back to HDBSCAN if DBSCAN collapses.
    'Collapsed' includes not just >90% noise or <=1 cluster, but also a
    single cluster absorbing >70% of all non-noise points -- a DBSCAN run
    that reports 3 clusters can still be functionally degenerate if one of
    them contains 99% of everything (measured directly: this exact failure
    happened on the real 100k dataset and the narrower check missed it)."""
    dbscan_result = fit_dbscan(X, min_samples=min_samples)
    dbscan_eval = evaluate_clustering(X, dbscan_result["labels"])

    labels = dbscan_result["labels"]
    non_noise = labels[labels != -1]
    dominant_fraction = 0.0
    if len(non_noise) > 0:
        _, counts = np.unique(non_noise, return_counts=True)
        dominant_fraction = float(counts.max() / counts.sum())
    dbscan_eval["dominant_cluster_fraction"] = round(dominant_fraction, 4)

    collapsed = (
        dbscan_eval["noise_fraction"] > 0.9
        or dbscan_eval["n_clusters"] <= 1
        or dominant_fraction > 0.7
    )
    if not collapsed:
        return {**dbscan_result, "evaluation": dbscan_eval, "collapsed": False}

    hdbscan_result = fit_hdbscan_adaptive(X, n_samples=len(X))
    hdbscan_eval = evaluate_clustering(X, hdbscan_result["labels"])
    return {
        **hdbscan_result, "evaluation": hdbscan_eval, "collapsed": True,
        "dbscan_attempt_evaluation": dbscan_eval,
        "fallback_reason": (
            f"DBSCAN noise_fraction={dbscan_eval['noise_fraction']}, "
            f"n_clusters={dbscan_eval['n_clusters']}, "
            f"dominant_cluster_fraction={dominant_fraction} (>0.7 threshold)"
        ),
    }


def evaluate_cluster_purity_against_scenarios(labels: np.ndarray, scenario_ids: list) -> dict:
    """Do entities from the same synthetic scenario type land in the same
    behavioral cluster more often than chance? (audit Section 16)"""
    import pandas as pd
    df = pd.DataFrame({"cluster": labels, "scenario": scenario_ids})
    df = df[(df["cluster"] != -1) & (df["scenario"].notna())]
    if df.empty:
        return {"purity": None, "note": "no non-noise risky entities to evaluate"}

    majority_matches = 0
    for cluster_id, grp in df.groupby("cluster"):
        majority_scenario = grp["scenario"].mode().iloc[0]
        majority_matches += (grp["scenario"] == majority_scenario).sum()

    purity = majority_matches / len(df)
    return {"purity": round(float(purity), 4), "n_evaluated": int(len(df))}

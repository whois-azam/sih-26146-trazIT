"""
graph/algorithms.py

Phase 3 — the analytical algorithms the prototype never had (audit Section
8/18, finding #6). These run on the wallet-projection graph
(graph.build.wallet_projection) and produce per-wallet analytical facts that
feed Stage 3b of the ML pipeline (Section 6) and the risk score (Section 11)
directly -- not just visualization decoration.
"""

from __future__ import annotations

import time

import networkx as nx


def compute_centrality(H: nx.Graph, betweenness_sample_k: int = 500, seed: int = 42) -> dict:
    """Degree + approximate betweenness centrality on the wallet projection.
    Betweenness is APPROXIMATED via sampling (k nodes) rather than exact --
    exact betweenness is O(V*E), which at ~20k wallet nodes is too slow to
    run routinely (see benchmark below). This is a deliberate, documented
    trade-off, not an oversight."""
    degree_c = nx.degree_centrality(H)

    k = min(betweenness_sample_k, H.number_of_nodes())
    betweenness_c = nx.betweenness_centrality(H, k=k, seed=seed, weight=None)

    return {
        "degree_centrality": degree_c,
        "betweenness_centrality_approx": betweenness_c,
        "betweenness_sample_k": k,
    }


def detect_communities(H: nx.Graph, seed: int = 42) -> dict:
    """Louvain community detection on the wallet projection. Returns a
    wallet -> community_id mapping plus the modularity score (a sanity
    metric, not proof of real-world validity -- see audit Section 16)."""
    communities = nx.community.louvain_communities(H, weight="weight", seed=seed)
    modularity = nx.community.modularity(H, communities, weight="weight")

    wallet_to_community = {}
    for cid, members in enumerate(communities):
        for w in members:
            wallet_to_community[w] = cid

    community_sizes = {cid: len(members) for cid, members in enumerate(communities)}

    return {
        "wallet_to_community": wallet_to_community,
        "n_communities": len(communities),
        "modularity": modularity,
        "community_sizes": community_sizes,
    }


def benchmark(H: nx.Graph) -> dict:
    """Timing harness -- Phase 3 validation criterion from the roadmap:
    'benchmarked at 100k scale' before assuming performance is fine."""
    results = {"n_nodes": H.number_of_nodes(), "n_edges": H.number_of_edges()}

    t0 = time.time()
    degree_c = nx.degree_centrality(H)
    results["degree_centrality_seconds"] = round(time.time() - t0, 3)

    t0 = time.time()
    k = min(500, H.number_of_nodes())
    nx.betweenness_centrality(H, k=k, seed=42)
    results["betweenness_approx_k500_seconds"] = round(time.time() - t0, 3)

    t0 = time.time()
    communities = nx.community.louvain_communities(H, weight="weight", seed=42)
    results["louvain_seconds"] = round(time.time() - t0, 3)
    results["n_communities_found"] = len(communities)

    return results

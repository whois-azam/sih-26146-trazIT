"""
graph/run_phase3.py

Ties build.py + algorithms.py together, benchmarks at 100k scale (roadmap
Phase 3 validation criterion), and checks community detection against the
generator's own ring ground truth (audit Section 16 -- aware this measures
fit to the synthetic generator's definition, not real-world validity).
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd

from build import build_graph, wallet_projection, graph_summary
from algorithms import compute_centrality, detect_communities, benchmark


def evaluate_communities_against_rings(wallet_to_community: dict, ground_truth: dict) -> dict:
    """For each synthetic ring with >=2 wallets, did all its wallets land in
    the SAME detected community? This is the direct analogue of Section 16's
    'cluster purity against scenario_id ground truth' check."""
    rings: dict = {}
    for addr, info in ground_truth.items():
        rid = info.get("ring_id")
        if rid:
            rings.setdefault(rid, []).append(addr)

    multi_wallet_rings = {rid: members for rid, members in rings.items() if len(members) >= 2}
    recovered = 0
    for rid, members in multi_wallet_rings.items():
        comms = {wallet_to_community.get(m) for m in members}
        comms.discard(None)
        if len(comms) == 1:
            recovered += 1

    return {
        "multi_wallet_rings_total": len(multi_wallet_rings),
        "rings_fully_recovered_in_one_community": recovered,
        "ring_recovery_rate": round(recovered / len(multi_wallet_rings), 4) if multi_wallet_rings else None,
    }


def main():
    raw_dir = "../data/raw"
    proc_dir = "../data/processed"

    print("Loading Phase 1 + Phase 2 outputs...")
    df = pd.read_csv(os.path.join(raw_dir, "bitcoin_dataset.csv"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    direct_corr = pd.read_csv(os.path.join(proc_dir, "correlations_direct.csv"))
    fallback_corr = pd.read_csv(os.path.join(proc_dir, "correlations_fallback.csv"))
    with open(os.path.join(raw_dir, "ground_truth.json")) as f:
        ground_truth = json.load(f)

    print("Building graph (bulk construction)...")
    t0 = time.time()
    G = build_graph(df, direct_corr, fallback_corr)
    build_seconds = round(time.time() - t0, 3)

    summary = graph_summary(G)
    print(f"Graph built in {build_seconds}s")
    print(json.dumps(summary, indent=2))

    print("\nProjecting wallet-only graph for algorithms...")
    H = wallet_projection(G)
    print(f"Wallet projection: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

    print("\nBenchmarking algorithms at 100k-transaction scale...")
    bench = benchmark(H)
    print(json.dumps(bench, indent=2))

    print("\nComputing centrality...")
    centrality = compute_centrality(H)

    print("Detecting communities (Louvain)...")
    comm_result = detect_communities(H)
    print(f"Communities found: {comm_result['n_communities']}, modularity: {comm_result['modularity']:.4f}")

    print("\nEvaluating community detection against synthetic ring ground truth...")
    ring_eval = evaluate_communities_against_rings(comm_result["wallet_to_community"], ground_truth)
    print(json.dumps(ring_eval, indent=2))

    os.makedirs(proc_dir, exist_ok=True)
    with open(os.path.join(proc_dir, "graph_summary.json"), "w") as f:
        json.dump({
            "graph_summary": summary,
            "build_seconds": build_seconds,
            "wallet_projection_nodes": H.number_of_nodes(),
            "wallet_projection_edges": H.number_of_edges(),
            "benchmark": bench,
            "n_communities": comm_result["n_communities"],
            "modularity": comm_result["modularity"],
            "ring_recovery_eval": ring_eval,
        }, f, indent=2)

    with open(os.path.join(proc_dir, "graph_centrality.json"), "w") as f:
        json.dump({
            "degree_centrality": centrality["degree_centrality"],
            "betweenness_centrality_approx": centrality["betweenness_centrality_approx"],
        }, f)

    with open(os.path.join(proc_dir, "graph_communities.json"), "w") as f:
        json.dump(comm_result["wallet_to_community"], f)

    print(f"\nWritten: {proc_dir}/graph_summary.json, graph_centrality.json, graph_communities.json")


if __name__ == "__main__":
    main()

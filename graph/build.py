"""
graph/build.py

Phase 3 — the ONE graph builder (audit_report.md Section 8/19). Consumes
Phase 2's outputs directly rather than re-deriving correlation logic, so
there's a single source of truth for wallet<->ip confidence.

Node types: wallet, transaction, ip, asn, geo_country
Edge types:
  input              wallet -> transaction   (amount)
  output             transaction -> wallet   (amount)
  network_observation  wallet <-> ip         (role, confidence, correlation_type)
  resolves_to_asn    ip -> asn
  resolves_to_geo    ip -> geo_country
  transfer (derived) wallet -> wallet        (txid, amount, timestamp) -- direct
                                                value-flow projection, used by
                                                graph/algorithms.py for
                                                centrality & community detection

Built with add_nodes_from / add_edges_from (bulk), NOT per-row add_node/
add_edge calls -- the prototype's #1 flagged bottleneck (audit Section 15).

Ground truth (is_risky) is deliberately NEVER attached to graph nodes here.
The graph is an analytical input to Phase 5 (ML); leaking labels into it
would be leakage into every downstream feature that reads the graph.
"""

from __future__ import annotations

import json
from typing import Tuple

import networkx as nx
import pandas as pd


def build_graph(
    df: pd.DataFrame,
    direct_corr: pd.DataFrame,
    fallback_corr: pd.DataFrame | None = None,
) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    # ---- nodes (bulk) ----------------------------------------------------
    wallets = set()
    for row in df.itertuples(index=False):
        wallets.update(json.loads(row.input_addresses))
        wallets.update(json.loads(row.output_addresses))
    tx_nodes = [(txid, {"node_type": "transaction"}) for txid in df["txid"].unique()]
    wallet_nodes = [(w, {"node_type": "wallet"}) for w in wallets]
    ip_nodes = [(ip, {"node_type": "ip"}) for ip in set(df["src_ip"]) | set(df["dst_ip"])]
    asn_nodes = [(f"asn:{a}", {"node_type": "asn"}) for a in df["asn"].unique()]
    geo_nodes = [(f"geo:{g}", {"node_type": "geo_country"}) for g in df["geo_country"].unique()]

    G.add_nodes_from(tx_nodes)
    G.add_nodes_from(wallet_nodes)
    G.add_nodes_from(ip_nodes)
    G.add_nodes_from(asn_nodes)
    G.add_nodes_from(geo_nodes)

    # ---- input / output edges (bulk) --------------------------------------
    input_edges, output_edges, transfer_edges = [], [], []
    ip_geo_asn_edges = set()

    for row in df.itertuples(index=False):
        ins = json.loads(row.input_addresses)
        outs = json.loads(row.output_addresses)
        in_amts = json.loads(row.input_amounts)
        out_amts = json.loads(row.output_amounts)

        for w, amt in zip(ins, in_amts):
            input_edges.append((w, row.txid, {"amount": amt, "timestamp": row.timestamp}))
        for w, amt in zip(outs, out_amts):
            output_edges.append((row.txid, w, {"amount": amt, "timestamp": row.timestamp}))
            for in_w, in_amt in zip(ins, in_amts):
                transfer_edges.append((in_w, w, {
                    "txid": row.txid, "amount": amt, "timestamp": row.timestamp,
                }))

        ip_geo_asn_edges.add((row.src_ip, f"geo:{row.geo_country}"))
        ip_geo_asn_edges.add((row.src_ip, f"asn:{row.asn}"))

    G.add_edges_from([(u, v, "input", d) for u, v, d in input_edges])
    G.add_edges_from([(u, v, "output", d) for u, v, d in output_edges])
    G.add_edges_from([(u, v, "transfer", d) for u, v, d in transfer_edges])
    G.add_edges_from([(ip, dst, "resolves_to", {}) for ip, dst in ip_geo_asn_edges])

    # ---- network_observation edges, straight from Phase 2 (no re-derivation) ----
    net_obs_edges = []
    for row in direct_corr.itertuples(index=False):
        net_obs_edges.append((row.wallet, row.ip, {
            "role": row.role, "txid": row.txid, "timestamp": row.timestamp,
            "confidence": row.confidence, "correlation_type": row.correlation_type,
        }))
    if fallback_corr is not None and len(fallback_corr):
        for row in fallback_corr.itertuples(index=False):
            net_obs_edges.append((row.wallet, row.ip, {
                "role": "unknown", "txid": row.txid, "timestamp": row.timestamp,
                "confidence": row.confidence, "correlation_type": row.correlation_type,
            }))
    G.add_edges_from([(u, v, "network_observation", d) for u, v, d in net_obs_edges])

    return G


def wallet_projection(G: nx.MultiDiGraph) -> nx.Graph:
    """Collapse the multigraph down to a simple undirected wallet-only graph
    using the 'transfer' edges, weighted by number of transfers between each
    pair. This is what centrality/community-detection algorithms run on --
    Louvain and betweenness expect a simple weighted graph, not a
    heterogeneous multigraph with wallets, transactions, IPs all mixed in."""
    H = nx.Graph()
    wallet_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "wallet"]
    H.add_nodes_from(wallet_nodes)

    for u, v, k, data in G.edges(keys=True, data=True):
        if k != "transfer":
            continue
        if H.has_edge(u, v):
            H[u][v]["weight"] += 1
            H[u][v]["total_amount"] += data.get("amount", 0.0)
        else:
            H.add_edge(u, v, weight=1, total_amount=data.get("amount", 0.0))

    return H


def graph_summary(G: nx.MultiDiGraph) -> dict:
    node_counts: dict = {}
    for _, d in G.nodes(data=True):
        node_counts[d.get("node_type", "unknown")] = node_counts.get(d.get("node_type", "unknown"), 0) + 1
    edge_counts: dict = {}
    for _, _, k in G.edges(keys=True):
        edge_counts[k] = edge_counts.get(k, 0) + 1
    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "node_counts_by_type": node_counts,
        "edge_counts_by_type": edge_counts,
    }

"""
innovations/divergence.py

Innovation #1 (audit Section 13) -- behavioral-cluster-vs-graph-community
divergence, done BIDIRECTIONALLY (the dashboard's Phase 7 view only showed
one direction; this is the complete version):

  Type A: graph-connected but behaviorally different -- entities in the
          SAME graph community that land in DIFFERENT behavioral clusters
          (or are behavioral outliers). Possible mule moving funds for a
          different kind of operator than its neighbors.

  Type B: behaviorally similar but graph-disconnected -- entities in the
          SAME behavioral cluster that land in DIFFERENT graph communities.
          Possible parallel, non-communicating actors running the same
          playbook independently.

Cheap: no new ML, just a comparison of two already-computed label sets.
"""

from __future__ import annotations

import pandas as pd


def compute_divergence(
    cluster_labels: dict[str, int],
    communities: dict[str, int],
    risk_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for eid in risk_df["entity_id"]:
        rows.append({
            "entity_id": eid,
            "behavioral_cluster": cluster_labels.get(eid, -1),
            "graph_community": communities.get(eid),
        })
    df = pd.DataFrame(rows).merge(risk_df[["entity_id", "risk_score", "risk_level"]], on="entity_id")
    return df


def type_a_graph_connected_behaviorally_different(div_df: pd.DataFrame, min_community_size: int = 3) -> pd.DataFrame:
    """For each graph community with >= min_community_size members, flag
    members whose behavioral cluster differs from the community's own
    majority behavioral cluster (or is an outlier/noise)."""
    results = []
    for comm_id, grp in div_df.groupby("graph_community"):
        if len(grp) < min_community_size or comm_id is None:
            continue
        non_outlier = grp[grp["behavioral_cluster"] != -1]
        if non_outlier.empty:
            continue
        majority_cluster = non_outlier["behavioral_cluster"].mode()
        if majority_cluster.empty:
            continue
        majority_cluster = majority_cluster.iloc[0]
        diverging = grp[
            (grp["behavioral_cluster"] != majority_cluster)
        ].copy()
        if not diverging.empty:
            diverging["community_majority_cluster"] = majority_cluster
            diverging["community_size"] = len(grp)
            diverging["divergence_type"] = "graph_connected_behaviorally_different"
            results.append(diverging)
    return pd.concat(results, ignore_index=True) if results else div_df.iloc[0:0].copy()


def type_b_behaviorally_similar_graph_disconnected(
    div_df: pd.DataFrame, min_cluster_size: int = 3, max_cluster_size: int = 1000,
) -> pd.DataFrame:
    """For each SMALL, specific behavioral cluster (excluding noise) with
    min_cluster_size <= size <= max_cluster_size, flag members whose graph
    community differs from the cluster's own majority community.

    max_cluster_size matters: a dominant "general population" cluster
    (measured on real data: one cluster held 96% of all 20,000 entities)
    will trivially span nearly every graph community just by size alone --
    that's population structure, not investigatively meaningful divergence.
    Only clusters small enough to represent a specific, niche behavioral
    signature make this check meaningful."""
    results = []
    for cluster_id, grp in div_df.groupby("behavioral_cluster"):
        if cluster_id == -1 or len(grp) < min_cluster_size or len(grp) > max_cluster_size:
            continue
        non_null = grp[grp["graph_community"].notna()]
        if non_null.empty:
            continue
        majority_comm = non_null["graph_community"].mode()
        if majority_comm.empty:
            continue
        majority_comm = majority_comm.iloc[0]
        diverging = grp[grp["graph_community"] != majority_comm].copy()
        if not diverging.empty:
            diverging["cluster_majority_community"] = majority_comm
            diverging["cluster_size"] = len(grp)
            diverging["divergence_type"] = "behaviorally_similar_graph_disconnected"
            results.append(diverging)
    return pd.concat(results, ignore_index=True) if results else div_df.iloc[0:0].copy()


def compute_full_divergence_report(
    cluster_labels: dict[str, int], communities: dict[str, int], risk_df: pd.DataFrame,
) -> dict:
    div_df = compute_divergence(cluster_labels, communities, risk_df)
    type_a = type_a_graph_connected_behaviorally_different(div_df)
    type_b = type_b_behaviorally_similar_graph_disconnected(div_df)

    return {
        "type_a_count": len(type_a),
        "type_a_high_risk_count": int((type_a["risk_level"].isin(["CRITICAL", "HIGH"])).sum()) if len(type_a) else 0,
        "type_b_count": len(type_b),
        "type_b_high_risk_count": int((type_b["risk_level"].isin(["CRITICAL", "HIGH"])).sum()) if len(type_b) else 0,
        "type_a_entities": type_a,
        "type_b_entities": type_b,
    }

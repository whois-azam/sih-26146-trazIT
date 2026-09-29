"""
innovations/alert_chains.py

Innovation #3 (audit Section 13) -- one CHAIN-level alert per graph
community, in addition to individual entity alerts. Real investigations are
usually about rings, not single wallets -- aggregating already-computed
per-entity scores over each community is cheap and surfaces "this is a
ring of N entities" as a single alert instead of N disconnected ones.
"""

from __future__ import annotations

import pandas as pd


def build_chain_alerts(
    communities: dict[str, int],
    risk_df: pd.DataFrame,
    min_community_size: int = 2,
    min_members_flagged: int = 2,
) -> pd.DataFrame:
    """A community becomes a chain alert if it has at least
    min_members_flagged entities at HIGH or CRITICAL risk level."""
    comm_df = pd.DataFrame(list(communities.items()), columns=["entity_id", "graph_community"])
    merged = comm_df.merge(risk_df[["entity_id", "risk_score", "risk_level"]], on="entity_id", how="left")

    rows = []
    for comm_id, grp in merged.groupby("graph_community"):
        if len(grp) < min_community_size:
            continue
        n_flagged = int(grp["risk_level"].isin(["CRITICAL", "HIGH"]).sum())
        if n_flagged < min_members_flagged:
            continue
        rows.append({
            "graph_community": comm_id,
            "community_size": len(grp),
            "n_flagged_members": n_flagged,
            "n_critical": int((grp["risk_level"] == "CRITICAL").sum()),
            "n_high": int((grp["risk_level"] == "HIGH").sum()),
            "max_member_risk_score": round(float(grp["risk_score"].max()), 2),
            "mean_member_risk_score": round(float(grp["risk_score"].mean()), 2),
            "chain_alert_score": round(float(grp["risk_score"].max()) * (n_flagged / len(grp)), 2),
            "member_entity_ids": grp.sort_values("risk_score", ascending=False)["entity_id"].tolist(),
        })

    df = pd.DataFrame(rows)
    return df.sort_values("chain_alert_score", ascending=False) if len(df) else df

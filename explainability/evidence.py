"""
explainability/evidence.py

Phase 6 -- ONE explanation generator (audit Section 10), reconciling the
prototype's two divergent versions. Four evidence categories, kept
separately labeled so an investigator knows what KIND of signal they're
looking at:

  1. feature-level  -- deviating behavioral features vs. population baseline,
                        DEDUPLICATED by semantic group so correlated
                        features (e.g. fan_out and unique_output_addresses,
                        which move together by construction) don't count as
                        two independent reasons for the same underlying fact.
  2. model-output    -- the calibrated anomaly percentile, stated as what it
                        is ("more unusual than X% of the population"), never
                        as a probability of guilt.
  3. graph           -- centrality percentile + flagged-neighbor count, from
                        the real graph object, kept separate from...
  4. behavioral       -- ...cluster membership (Stage 3a), because "acts like"
                        and "connected to" are different facts and must not
                        be merged into one unlabeled sentence (Section 6/10).

HARD RULE: never pad to a target count. If only two signals clear
threshold, show two, with a plain note that only two were found.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FEATURE_GROUPS: dict[str, list[str]] = {
    "volume": [
        "transaction_count", "total_input_amount", "total_output_amount",
        "average_transaction_amount", "median_transaction_amount", "transaction_amount_std",
        "transaction_frequency", "amount_sum_last_1h", "amount_sum_last_24h", "amount_sum_last_7d",
        "tx_count_last_1h", "tx_count_last_24h", "tx_count_last_7d",
    ],
    "fan_in_out": [
        "fan_in", "fan_out", "unique_counterparties", "unique_input_addresses",
        "unique_output_addresses", "graph_degree", "graph_in_degree", "graph_out_degree",
        "incoming_outgoing_ratio", "incoming_tx_count", "outgoing_tx_count",
    ],
    "timing": [
        "average_time_between_transactions", "transaction_burstiness",
        "transaction_time_span_hours", "days_since_last_activity", "entity_age_days",
        "recent_vs_baseline_ratio_1h", "recent_vs_baseline_ratio_24h", "recent_vs_baseline_ratio_7d",
    ],
    "geography_network": [
        "unique_ips", "unique_countries", "unique_asns",
        "src_ip_association_count", "dst_ip_association_count",
    ],
    "fee_behavior": ["fee_mean", "fee_median", "fee_std"],
}

HUMAN_LABELS = {
    "transaction_count": "total transaction count",
    "total_input_amount": "total value sent",
    "total_output_amount": "total value received",
    "average_transaction_amount": "average transaction size",
    "transaction_frequency": "transactions per hour",
    "fan_out": "number of distinct recipients",
    "fan_in": "number of distinct senders",
    "unique_counterparties": "distinct counterparties",
    "incoming_outgoing_ratio": "send-vs-receive imbalance",
    "average_time_between_transactions": "typical time between transactions",
    "transaction_burstiness": "burstiness of activity timing",
    "unique_ips": "distinct IPs observed",
    "unique_countries": "distinct countries observed",
    "fee_mean": "average fee paid",
    "recent_vs_baseline_ratio_24h": "last-24h activity vs. own historical baseline",
    "recent_vs_baseline_ratio_7d": "last-7d activity vs. own historical baseline",
    "tx_count_last_24h": "transactions in the last 24h",
    "amount_sum_last_7d": "value moved in the last 7 days",
}


@dataclass
class EvidenceItem:
    category: str      # "feature", "model_output", "graph", "behavioral"
    statement: str
    magnitude: float = 0.0  # for sorting within a category, not shown to user


@dataclass
class EntityExplanation:
    entity_id: str
    risk_score: float
    risk_level: str
    items: list[EvidenceItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "risk_score": round(self.risk_score, 2),
            "risk_level": self.risk_level,
            "n_evidence_items": len(self.items),
            "evidence": [{"category": i.category, "statement": i.statement} for i in self.items],
        }


def compute_feature_baselines(train_features: pd.DataFrame, all_columns: list[str]) -> dict:
    """Mean/std per feature, computed on the TRAIN population ONLY -- frozen
    reference, same philosophy as the anomaly detector and risk thresholds."""
    baselines = {}
    for col in all_columns:
        vals = train_features[col].astype(float)
        baselines[col] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0)) or 1e-6}
    return baselines


def top_deviating_feature_per_group(
    entity_row: pd.Series, baselines: dict, z_threshold: float = 1.5
) -> list[EvidenceItem]:
    items = []
    for group_name, columns in FEATURE_GROUPS.items():
        best_col, best_z = None, 0.0
        for col in columns:
            if col not in baselines or col not in entity_row:
                continue
            b = baselines[col]
            z = (float(entity_row[col]) - b["mean"]) / b["std"]
            if abs(z) > abs(best_z):
                best_col, best_z = col, z
        if best_col is not None and abs(best_z) >= z_threshold:
            label = HUMAN_LABELS.get(best_col, best_col.replace("_", " "))
            direction = "higher" if best_z > 0 else "lower"
            b = baselines[best_col]
            observed = float(entity_row[best_col])
            items.append(EvidenceItem(
                category="feature",
                statement=(
                    f"{label} is unusually {direction} compared to similar wallets "
                    f"(about {abs(best_z):.1f}x the typical variation seen across the population; "
                    f"observed {observed:.4g} vs population average {b['mean']:.4g})"
                ),
                magnitude=abs(best_z),
            ))
    return items


def model_output_evidence(anomaly_percentile: float) -> EvidenceItem:
    return EvidenceItem(
        category="model_output",
        statement=(
            f"Overall activity pattern stands out as unusual compared to the wider population "
            f"(ranked more anomalous than {anomaly_percentile*100:.1f}% of entities by the detection model — "
            f"this flags it for investigation, not as proof of wrongdoing)"
        ),
        magnitude=anomaly_percentile,
    )


def graph_evidence(centrality_percentile: float, n_flagged_neighbors: int, n_total_neighbors: int) -> list[EvidenceItem]:
    items = []
    if centrality_percentile >= 0.90:
        items.append(EvidenceItem(
            category="graph",
            statement=(
                f"Occupies an unusually central position in the transaction network "
                f"(more connected than {centrality_percentile*100:.1f}% of entities by graph degree centrality)"
            ),
            magnitude=centrality_percentile,
        ))
    if n_flagged_neighbors > 0:
        items.append(EvidenceItem(
            category="graph",
            statement=(
                f"Has direct transaction links to counterparties that are themselves flagged as anomalous "
                f"({n_flagged_neighbors} of {n_total_neighbors} direct counterparties)"
            ),
            magnitude=n_flagged_neighbors / max(n_total_neighbors, 1),
        ))
    return items


def behavioral_evidence(is_cluster_outlier: bool, cluster_size: int | None) -> list[EvidenceItem]:
    items = []
    if is_cluster_outlier:
        items.append(EvidenceItem(
            category="behavioral",
            statement=(
                "Activity pattern does not fit any recognized group of similar wallets "
                "(flagged as a behavioral outlier — worth investigating, not proof of wrongdoing)"
            ),
            magnitude=1.0,
        ))
    elif cluster_size is not None and cluster_size < 50:
        items.append(EvidenceItem(
            category="behavioral",
            statement=(
                f"Activity looks similar to a small group of {cluster_size} wallets "
                f"with matching behavioral patterns"
            ),
            magnitude=1.0 / max(cluster_size, 1),
        ))
    return items


def generate_explanation(
    entity_id: str,
    risk_score: float,
    risk_level: str,
    entity_features_row: pd.Series,
    baselines: dict,
    anomaly_percentile: float,
    centrality_percentile: float,
    n_flagged_neighbors: int,
    n_total_neighbors: int,
    is_cluster_outlier: bool,
    cluster_size: int | None,
) -> EntityExplanation:
    items: list[EvidenceItem] = []
    items.extend(top_deviating_feature_per_group(entity_features_row, baselines))
    items.append(model_output_evidence(anomaly_percentile))
    items.extend(graph_evidence(centrality_percentile, n_flagged_neighbors, n_total_neighbors))
    items.extend(behavioral_evidence(is_cluster_outlier, cluster_size))

    items.sort(key=lambda i: i.magnitude, reverse=True)
    return EntityExplanation(entity_id=entity_id, risk_score=risk_score, risk_level=risk_level, items=items)

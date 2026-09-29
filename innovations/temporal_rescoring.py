"""
innovations/temporal_rescoring.py

Innovation #2 (audit Section 13) -- flag entities whose RECENT behavior
deviates sharply from their OWN historical baseline, not just from the
population. Phase 4 already computes the underlying features
(recent_vs_baseline_ratio_1h/24h/7d); this surfaces them as a dedicated,
investigator-facing "behavior change" alert category instead of leaving
them buried as one feature among 44.

Cheap: reuses existing Phase 4 features, no new model.
"""

from __future__ import annotations

import pandas as pd


def flag_behavior_change(
    feats: pd.DataFrame,
    windows: list[str] = ("24h", "7d"),
    ratio_threshold: float = 3.0,
    min_baseline_tx: int = 3,
) -> pd.DataFrame:
    """An entity is flagged if its recent activity rate is >= ratio_threshold
    times its own long-run baseline rate, for at least one window -- e.g. a
    wallet quiet for weeks that suddenly bursts. min_baseline_tx guards
    against flagging brand-new entities where the "baseline" itself is
    built from too few observations to be a meaningful rate."""
    results = []
    for _, row in feats.iterrows():
        if row["transaction_count"] < min_baseline_tx:
            continue
        triggered_windows = []
        max_ratio = 0.0
        for w in windows:
            col = f"recent_vs_baseline_ratio_{w}"
            if col not in row:
                continue
            ratio = row[col]
            if ratio >= ratio_threshold:
                triggered_windows.append({"window": w, "ratio": round(float(ratio), 2)})
                max_ratio = max(max_ratio, ratio)
        if triggered_windows:
            results.append({
                "entity_id": row["entity_id"],
                "max_ratio": round(float(max_ratio), 2),
                "triggered_windows": triggered_windows,
                "days_since_last_activity": round(float(row.get("days_since_last_activity", 0)), 2),
                "transaction_count": int(row["transaction_count"]),
            })

    df = pd.DataFrame(results)
    return df.sort_values("max_ratio", ascending=False) if len(df) else df

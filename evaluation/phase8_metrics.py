"""
evaluation/metrics.py

Phase 8 -- consolidates every metric produced across Phases 1-7 into one
place, PLUS a check none of the earlier phases did: automated verification
that explanation statements are factually correct, not just plausible-
sounding (audit Section 16: "spot-check a sample of alerts and manually
verify the stated evidence actually matches the underlying data" -- made
automatic and re-runnable instead of a one-time manual check).
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "../features")
sys.path.insert(0, "../ml")
sys.path.insert(0, "../graph")
sys.path.insert(0, "../explainability")
sys.path.insert(0, "../scoring")
sys.path.insert(0, "../dashboard")

DATA_RAW = "../data/raw"
DATA_PROC = "../data/processed"


def load_json(name: str) -> dict:
    path = os.path.join(DATA_PROC, name)
    if not os.path.exists(path):
        return {"_missing": True, "_path": path}
    with open(path) as f:
        return json.load(f)


def consolidate_all_metrics() -> dict:
    gen_meta_path = os.path.join(DATA_RAW, "generation_metadata.json")
    with open(gen_meta_path) as f:
        phase1 = json.load(f)

    return {
        "phase1_generation": phase1,
        "phase2_correlation": load_json("correlation_stats.json"),
        "phase3_graph": load_json("graph_summary.json"),
        "phase5_ml": load_json("ml_evaluation.json"),
        "phase6_risk_scoring": load_json("risk_evaluation.json"),
    }


def verify_evidence_accuracy(n_samples: int = 25, seed: int = 42) -> dict:
    """Pick a random sample of entities, generate their evidence fresh, and
    INDEPENDENTLY recompute the underlying numbers the evidence text claims
    -- checking the arithmetic is actually right, not just that it reads
    plausibly. This is what makes an explanation trustworthy rather than a
    confident-sounding guess."""
    from data_loader import load_everything, get_entity_evidence

    ctx = load_everything()
    risk_df = ctx["risk_df"]
    rng = np.random.RandomState(seed)
    sample_ids = rng.choice(risk_df["entity_id"].values, size=min(n_samples, len(risk_df)), replace=False)

    feats_indexed = ctx["feats"].set_index("entity_id")
    results = []

    for eid in sample_ids:
        evidence = get_entity_evidence(eid, ctx)
        row = risk_df[risk_df["entity_id"] == eid].iloc[0]

        checks = []
        for item in evidence["evidence"]:
            stmt = item["statement"]

            if item["category"] == "model_output":
                m = re.search(r"more unusual than ([\d.]+)% of the population", stmt)
                if m:
                    claimed = float(m.group(1))
                    actual = round(row["anomaly_component"] * 100, 1)
                    checks.append({"claim": "model_output_percentile", "claimed": claimed, "actual": actual, "match": abs(claimed - actual) < 0.15})

            elif item["category"] == "graph" and "more central than" in stmt:
                m = re.search(r"more central than ([\d.]+)% of entities", stmt)
                if m:
                    claimed = float(m.group(1))
                    actual = round(row["graph_centrality_component"] * 100, 1)
                    checks.append({"claim": "centrality_percentile", "claimed": claimed, "actual": actual, "match": abs(claimed - actual) < 0.15})

            elif item["category"] == "graph" and "counterparties that are themselves flagged" in stmt:
                m = re.search(r"connected to (\d+) of its (\d+) counterparties", stmt)
                if m:
                    claimed_flagged, claimed_total = int(m.group(1)), int(m.group(2))
                    actual_total = len(list(ctx["H"].neighbors(eid))) if eid in ctx["H"] else 0
                    checks.append({"claim": "neighbor_count", "claimed": claimed_total, "actual": actual_total, "match": claimed_total == actual_total})

            elif item["category"] == "feature":
                m = re.match(r"(.+) is ([\d.]+)x standard deviations (higher|lower)", stmt)
                if m:
                    label_text, claimed_z, direction = m.group(1), float(m.group(2)), m.group(3)
                    # recompute independently from the same baselines used to generate it
                    from evidence import FEATURE_GROUPS, HUMAN_LABELS
                    reverse_label = {v: k for k, v in HUMAN_LABELS.items()}
                    col = reverse_label.get(label_text)
                    if col and col in ctx["baselines"]:
                        b = ctx["baselines"][col]
                        actual_val = float(feats_indexed.loc[eid][col])
                        actual_z = abs((actual_val - b["mean"]) / b["std"])
                        checks.append({"claim": f"feature_z_{col}", "claimed": claimed_z, "actual": round(actual_z, 2), "match": abs(claimed_z - actual_z) < 0.05})

        results.append({
            "entity_id": eid, "risk_score": float(row["risk_score"]),
            "n_evidence_items": len(evidence["evidence"]), "n_checks_performed": len(checks),
            "n_checks_passed": sum(1 for c in checks if c["match"]),
            "checks": checks,
        })

    total_checks = sum(r["n_checks_performed"] for r in results)
    total_passed = sum(r["n_checks_passed"] for r in results)

    return {
        "n_entities_sampled": len(results),
        "total_verifiable_claims_checked": total_checks,
        "total_claims_passed": total_passed,
        "accuracy_rate": round(total_passed / total_checks, 4) if total_checks else None,
        "per_entity_detail": results,
    }

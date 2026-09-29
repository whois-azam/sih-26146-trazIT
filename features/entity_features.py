"""Entity-level feature engineering for synthetic Bitcoin transaction data."""
from __future__ import annotations

import ast
import json
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

MAIN_COLUMNS = [
    "timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "txid",
    "input_addresses", "output_addresses", "input_amounts", "output_amounts",
    "fee", "script_type", "geo_country", "asn"
]

SCRIPT_TYPES = ["P2WPKH", "P2WSH", "P2TR", "P2PKH"]


def _parse_array(value: Any) -> list:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")
    return parsed


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _ensure_transaction_schema(df: pd.DataFrame) -> None:
    missing = [c for c in MAIN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required transaction columns: {missing}")


def _canonical_ts(df: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce", format="mixed")
    if ts.isna().any():
        raise ValueError("One or more timestamps could not be parsed")
    return ts


def extract_entity_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction/network observations into one numerical row per entity.

    Entity IDs are wallet addresses unless an explicit entity_id column is supplied.
    The function never uses ground-truth labels.
    """
    df = transactions.copy()
    _ensure_transaction_schema(df)
    df["_timestamp"] = _canonical_ts(df)

    tx_records: list[dict] = []
    entity_events: dict[str, list[tuple[pd.Timestamp, str, float, float, str, str, str, str, str]]] = defaultdict(list)
    edges: dict[str, set[str]] = defaultdict(set)
    in_neighbors: dict[str, set[str]] = defaultdict(set)
    out_neighbors: dict[str, set[str]] = defaultdict(set)
    entity_txids: dict[str, list[str]] = defaultdict(list)
    entity_countries: dict[str, set[str]] = defaultdict(set)
    entity_asns: dict[str, set[str]] = defaultdict(set)
    entity_ips: dict[str, set[str]] = defaultdict(set)
    entity_scripts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    entity_in_amounts: dict[str, list[float]] = defaultdict(list)
    entity_out_amounts: dict[str, list[float]] = defaultdict(list)
    entity_fees: dict[str, list[float]] = defaultdict(list)
    entity_in_count: dict[str, int] = defaultdict(int)
    entity_out_count: dict[str, int] = defaultdict(int)

    for row, ts in zip(df.itertuples(index=False), df["_timestamp"].tolist()):
        inputs = [str(x) for x in _parse_array(row.input_addresses)]
        outputs = [str(x) for x in _parse_array(row.output_addresses)]
        in_amounts = [_safe_float(x) for x in _parse_array(row.input_amounts)]
        out_amounts = [_safe_float(x) for x in _parse_array(row.output_amounts)]
        if not inputs or not outputs or len(inputs) != len(in_amounts) or len(outputs) != len(out_amounts):
            raise ValueError(f"Malformed address/amount arrays for txid={row.txid}")
        if any(x <= 0 or not np.isfinite(x) for x in in_amounts + out_amounts):
            raise ValueError(f"Invalid transaction amount for txid={row.txid}")
        fee = _safe_float(row.fee)
        script = str(row.script_type)
        country = str(row.geo_country)
        asn = str(row.asn)
        src_ip = str(row.src_ip)
        dst_ip = str(row.dst_ip)
        txid = str(row.txid)

        total_in = float(sum(in_amounts))
        total_out = float(sum(out_amounts))
        tx_amount = total_out
        tx_records.append({"txid": txid, "timestamp": ts})

        for src in inputs:
            entity_in_count[src] += 1
            entity_in_amounts[src].append(total_in / len(inputs))
            entity_fees[src].append(fee)
            entity_events[src].append((ts, "in", total_in / len(inputs), fee, src_ip, country, asn, script, txid))
            entity_txids[src].append(txid)
            entity_countries[src].add(country)
            entity_asns[src].add(asn)
            entity_ips[src].update([src_ip, dst_ip])
            entity_scripts[src][script] += 1
        for dst in outputs:
            entity_out_count[dst] += 1
            entity_out_amounts[dst].append(tx_amount / len(outputs))
            entity_fees[dst].append(fee)
            entity_events[dst].append((ts, "out", tx_amount / len(outputs), fee, src_ip, country, asn, script, txid))
            entity_txids[dst].append(txid)
            entity_countries[dst].add(country)
            entity_asns[dst].add(asn)
            entity_ips[dst].update([src_ip, dst_ip])
            entity_scripts[dst][script] += 1
        for src in inputs:
            for dst in outputs:
                if src == dst:
                    continue
                edges[src].add(dst)
                out_neighbors[src].add(dst)
                in_neighbors[dst].add(src)

    entities = sorted(set(entity_events) | set(edges) | set(in_neighbors) | set(out_neighbors))
    if not entities:
        raise ValueError("No wallet/entity references found in transactions")

    rows: list[dict] = []
    for entity in entities:
        events = sorted(entity_events[entity], key=lambda x: x[0])
        timestamps = [e[0] for e in events]
        in_amounts = np.array(entity_in_amounts[entity], dtype=float)
        out_amounts = np.array(entity_out_amounts[entity], dtype=float)
        all_amounts = np.concatenate([in_amounts, out_amounts]) if len(in_amounts) or len(out_amounts) else np.array([], dtype=float)
        fees = np.array(entity_fees[entity], dtype=float)
        intervals = np.diff(np.array([t.value / 1e9 for t in timestamps], dtype=float)) if len(timestamps) > 1 else np.array([], dtype=float)
        mean_interval = float(np.mean(intervals)) if len(intervals) else 0.0
        interval_cv = float(np.std(intervals) / mean_interval) if len(intervals) and mean_interval > 0 else 0.0
        observation_hours = max((timestamps[-1] - timestamps[0]).total_seconds() / 3600.0, 1.0) if timestamps else 1.0
        frequency_per_hour = len(timestamps) / observation_hours
        burst_ratio = float(np.mean(intervals <= 60)) if len(intervals) else 0.0
        tx_count = len(set(entity_txids[entity]))
        in_count = entity_in_count[entity]
        out_count = entity_out_count[entity]
        unique_cps = len(out_neighbors[entity] | in_neighbors[entity])
        rows.append({
            "entity_id": entity,
            "transaction_count": tx_count,
            "total_input_amount": float(in_amounts.sum()) if len(in_amounts) else 0.0,
            "total_output_amount": float(out_amounts.sum()) if len(out_amounts) else 0.0,
            "average_transaction_amount": float(all_amounts.mean()) if len(all_amounts) else 0.0,
            "median_transaction_amount": float(np.median(all_amounts)) if len(all_amounts) else 0.0,
            "transaction_amount_std": float(all_amounts.std(ddof=0)) if len(all_amounts) else 0.0,
            "transaction_frequency": frequency_per_hour,
            "unique_counterparties": unique_cps,
            "unique_input_addresses": len(in_neighbors[entity]),
            "unique_output_addresses": len(out_neighbors[entity]),
            "unique_ips": len(entity_ips[entity]),
            "unique_countries": len(entity_countries[entity]),
            "unique_asns": len(entity_asns[entity]),
            "fan_in": len(in_neighbors[entity]),
            "fan_out": len(out_neighbors[entity]),
            "graph_degree": unique_cps,
            "graph_in_degree": len(in_neighbors[entity]),
            "graph_out_degree": len(out_neighbors[entity]),
            "average_time_between_transactions": mean_interval,
            "transaction_burstiness": interval_cv + burst_ratio,
            "fee_mean": float(fees.mean()) if len(fees) else 0.0,
            "fee_median": float(np.median(fees)) if len(fees) else 0.0,
            "fee_std": float(fees.std(ddof=0)) if len(fees) else 0.0,
            "incoming_outgoing_ratio": float(in_count / max(out_count, 1)),
            "incoming_tx_count": in_count,
            "outgoing_tx_count": out_count,
            "src_ip_association_count": len({e[4] for e in events}),
            "dst_ip_association_count": len(entity_ips[entity]),
            "script_type_p2wpkh_share": entity_scripts[entity].get("P2WPKH", 0) / max(len(events), 1),
            "script_type_p2wsh_share": entity_scripts[entity].get("P2WSH", 0) / max(len(events), 1),
            "script_type_p2tr_share": entity_scripts[entity].get("P2TR", 0) / max(len(events), 1),
            "script_type_p2pkh_share": entity_scripts[entity].get("P2PKH", 0) / max(len(events), 1),
            "transaction_time_span_hours": observation_hours,
            "transaction_txid_count": tx_count,
            "transaction_txids": json.dumps(sorted(set(entity_txids[entity])), separators=(",", ":")),
        })

    features = pd.DataFrame(rows)
    numerical = [c for c in features.columns if c not in {"entity_id", "transaction_txids"}]
    features[numerical] = features[numerical].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features


FEATURE_COLUMNS = [
    "transaction_count", "total_input_amount", "total_output_amount", "average_transaction_amount",
    "median_transaction_amount", "transaction_amount_std", "transaction_frequency", "unique_counterparties",
    "unique_input_addresses", "unique_output_addresses", "unique_ips", "unique_countries", "unique_asns",
    "fan_in", "fan_out", "graph_degree", "graph_in_degree", "graph_out_degree",
    "average_time_between_transactions", "transaction_burstiness", "fee_mean", "fee_median", "fee_std",
    "incoming_outgoing_ratio", "incoming_tx_count", "outgoing_tx_count", "src_ip_association_count",
    "dst_ip_association_count", "script_type_p2wpkh_share", "script_type_p2wsh_share", "script_type_p2tr_share",
    "script_type_p2pkh_share", "transaction_time_span_hours"
]

LOG_FEATURE_COLUMNS = [
    "transaction_count", "total_input_amount", "total_output_amount", "average_transaction_amount",
    "median_transaction_amount", "transaction_amount_std", "transaction_frequency",
    "unique_counterparties", "unique_ips", "fan_in", "fan_out", "graph_degree",
    "graph_in_degree", "graph_out_degree", "average_time_between_transactions",
    "fee_mean", "fee_median", "fee_std", "incoming_tx_count", "outgoing_tx_count",
    "transaction_time_span_hours"
]


# ---------------------------------------------------------------------------
# EXTENSION (Phase 4 / audit_report.md Section 6, Stage 1) -- added on top of
# the original extractor above, which is left untouched as the single source
# of truth (audit Section 17: "KEEP -- extend, don't fork").
#
# All-time aggregates (above) can't tell you an entity just CHANGED behavior.
# These add recency-weighted, windowed features so a wallet that's been
# quiet for weeks and suddenly bursts stands out against its OWN history,
# not just against the population -- this is Innovation #2 from Section 13.
# Still entirely label-blind: built only from timestamps already in the data.
# ---------------------------------------------------------------------------

DEFAULT_WINDOWS = [("1h", 3600), ("24h", 86400), ("7d", 604800)]


def extract_temporal_window_features(
    transactions: pd.DataFrame,
    reference_time: pd.Timestamp | None = None,
    windows: list[tuple[str, int]] | None = None,
) -> pd.DataFrame:
    """Per-entity recency/windowed activity features, computed independently
    of extract_entity_features() (separate pass, same input) so the
    label-blind, verified original function is never modified."""
    df = transactions.copy()
    _ensure_transaction_schema(df)
    df["_timestamp"] = _canonical_ts(df)
    windows = windows or DEFAULT_WINDOWS
    reference_time = reference_time or df["_timestamp"].max()

    entity_timestamps: dict[str, list[pd.Timestamp]] = defaultdict(list)
    entity_amounts: dict[str, list[tuple[pd.Timestamp, float]]] = defaultdict(list)

    for row, ts in zip(df.itertuples(index=False), df["_timestamp"].tolist()):
        inputs = [str(x) for x in _parse_array(row.input_addresses)]
        outputs = [str(x) for x in _parse_array(row.output_addresses)]
        in_amounts = [_safe_float(x) for x in _parse_array(row.input_amounts)]
        out_amounts = [_safe_float(x) for x in _parse_array(row.output_amounts)]

        for w, amt in zip(inputs, in_amounts):
            entity_timestamps[w].append(ts)
            entity_amounts[w].append((ts, amt))
        for w, amt in zip(outputs, out_amounts):
            entity_timestamps[w].append(ts)
            entity_amounts[w].append((ts, amt))

    rows: list[dict] = []
    for entity, ts_list in entity_timestamps.items():
        ts_arr = pd.Series(sorted(ts_list))
        amt_pairs = entity_amounts[entity]

        first_seen = ts_arr.iloc[0]
        last_seen = ts_arr.iloc[-1]
        observation_hours = max((last_seen - first_seen).total_seconds() / 3600.0, 1.0)
        baseline_rate_per_hour = len(ts_arr) / observation_hours

        rec = {
            "entity_id": entity,
            "days_since_last_activity": round((reference_time - last_seen).total_seconds() / 86400.0, 4),
            "entity_age_days": round((last_seen - first_seen).total_seconds() / 86400.0, 4),
        }

        for label, seconds in windows:
            window_start = reference_time - pd.Timedelta(seconds=seconds)
            count_in_window = int(((ts_arr > window_start) & (ts_arr <= reference_time)).sum())
            amount_in_window = float(sum(a for t, a in amt_pairs if window_start < t <= reference_time))

            expected_count = baseline_rate_per_hour * (seconds / 3600.0)
            # ratio of observed recent activity to what the entity's OWN
            # long-run baseline rate would predict -- >1 means recently
            # busier than usual, not just busier than the population.
            recent_vs_baseline_ratio = count_in_window / expected_count if expected_count > 0 else 0.0

            rec[f"tx_count_last_{label}"] = count_in_window
            rec[f"amount_sum_last_{label}"] = amount_in_window
            rec[f"recent_vs_baseline_ratio_{label}"] = round(recent_vs_baseline_ratio, 4)

        rows.append(rec)

    out = pd.DataFrame(rows)
    numerical = [c for c in out.columns if c != "entity_id"]
    out[numerical] = out[numerical].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


TEMPORAL_FEATURE_COLUMNS = ["days_since_last_activity", "entity_age_days"] + [
    f"{prefix}_{label}"
    for label, _ in DEFAULT_WINDOWS
    for prefix in ("tx_count_last", "amount_sum_last", "recent_vs_baseline_ratio")
]


def build_features(
    transactions: pd.DataFrame,
    reference_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Single entrypoint Phase 5 (ML) should call: all-time behavioral
    features (original extractor) + temporal-window features, merged on
    entity_id. This is the one function downstream stages import."""
    behavioral = extract_entity_features(transactions)
    temporal = extract_temporal_window_features(transactions, reference_time=reference_time)
    merged = behavioral.merge(temporal, on="entity_id", how="left")
    return merged


ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + TEMPORAL_FEATURE_COLUMNS

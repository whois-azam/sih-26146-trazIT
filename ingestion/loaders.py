"""
ingestion/loaders.py

Phase 2 — unified multi-format ingestion.

Fixes the prototype's divergence (forensic report: two separate ingestion
paths, only one of which supported JSON/XML). This is the ONE loader used
everywhere downstream.

Supports: .csv, .json, .jsonl, .xml
Performs: schema aliasing (tolerates reasonable header variants), type
coercion, and structural validation -- but does NOT silently invent data.
Bad rows are reported, not dropped silently.
"""

from __future__ import annotations

import ipaddress
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

CANONICAL_COLUMNS: List[str] = [
    "timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "txid",
    "input_addresses", "output_addresses", "input_amounts", "output_amounts",
    "fee", "script_type", "geo_country", "asn",
]

# Tolerated header aliases -> canonical column name.
SCHEMA_ALIASES: Dict[str, str] = {
    "time": "timestamp", "ts": "timestamp", "datetime": "timestamp",
    "source_ip": "src_ip", "sourceip": "src_ip", "ip_src": "src_ip",
    "dest_ip": "dst_ip", "destination_ip": "dst_ip", "ip_dst": "dst_ip",
    "sport": "src_port", "source_port": "src_port",
    "dport": "dst_port", "dest_port": "dst_port", "destination_port": "dst_port",
    "tx_id": "txid", "transaction_id": "txid",
    "inputs": "input_addresses", "input_address": "input_addresses",
    "outputs": "output_addresses", "output_address": "output_addresses",
    "input_amount": "input_amounts", "output_amount": "output_amounts",
    "fees": "fee", "tx_fee": "fee",
    "script": "script_type", "scripttype": "script_type",
    "country": "geo_country", "geo": "geo_country",
    "asn_number": "asn", "as_number": "asn",
}

JSON_ARRAY_COLUMNS = ["input_addresses", "output_addresses", "input_amounts", "output_amounts"]


@dataclass
class ValidationReport:
    total_rows: int = 0
    valid_rows: int = 0
    dropped_rows: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Total rows read: {self.total_rows}",
            f"Valid rows: {self.valid_rows}",
            f"Dropped rows: {self.dropped_rows}",
        ]
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            lines.extend(f"  - {w}" for w in self.warnings[:10])
            if len(self.warnings) > 10:
                lines.append(f"  ... and {len(self.warnings) - 10} more")
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            lines.extend(f"  - {e}" for e in self.errors[:10])
        return "\n".join(lines)


def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {c: SCHEMA_ALIASES[c.lower()] for c in df.columns if c.lower() in SCHEMA_ALIASES}
    df = df.rename(columns=rename_map)
    df = df.rename(columns={c: c.lower() for c in df.columns if c.lower() in CANONICAL_COLUMNS})
    return df


def _ensure_json_array_str(value) -> str:
    """Normalize a JSON-array field to a canonical JSON string, tolerating
    values that already arrived as Python lists (e.g. from a JSON source)."""
    if isinstance(value, str):
        # already a JSON string -- validate it parses
        json.loads(value)
        return value
    if isinstance(value, list):
        return json.dumps(value, separators=(",", ":"))
    raise ValueError(f"Cannot coerce value to JSON array: {value!r}")


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_json(path: Path) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "records" in data:
        data = data["records"]
    if not isinstance(data, list):
        raise ValueError("JSON input must be a list of records or {'records': [...]}")
    return pd.DataFrame(data)


def _load_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def _load_xml(path: Path) -> pd.DataFrame:
    tree = ET.parse(path)
    root = tree.getroot()
    records = []
    # Accept <records><record>...</record></records> or a flat list of
    # same-tag children at the root.
    row_elements = root.findall(".//record") or list(root)
    for row_el in row_elements:
        rec = {}
        for child in row_el:
            tag = child.tag
            if list(child):  # nested elements -> treat as a JSON array of children's text
                rec[tag] = [c.text for c in child]
            else:
                rec[tag] = child.text
        records.append(rec)
    return pd.DataFrame(records)


LOADERS = {
    ".csv": _load_csv,
    ".json": _load_json,
    ".jsonl": _load_jsonl,
    ".xml": _load_xml,
}


def load_raw_dataset(path: str) -> pd.DataFrame:
    """Load a raw transaction/network dataset from CSV, JSON, JSONL, or XML,
    applying schema aliasing so reasonable header variants are tolerated."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file extension '{ext}'. Supported: {list(LOADERS)}")
    df = LOADERS[ext](p)
    df = _apply_aliases(df)
    return df


def validate_and_normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate structure/types and return a cleaned DataFrame plus a report.
    Rows that fail hard structural checks are dropped and logged, not silently
    coerced into something that looks valid."""
    report = ValidationReport(total_rows=len(df))

    missing_cols = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing_cols:
        report.errors.append(f"Missing required columns: {missing_cols}")
        report.dropped_rows = len(df)
        return df.iloc[0:0], report

    df = df[CANONICAL_COLUMNS].copy()

    valid_mask = pd.Series(True, index=df.index)

    # timestamp
    parsed_ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    bad_ts = parsed_ts.isna()
    if bad_ts.any():
        report.warnings.append(f"{bad_ts.sum()} rows had unparseable timestamps")
        valid_mask &= ~bad_ts
    df["timestamp"] = parsed_ts

    # IPs
    for col in ("src_ip", "dst_ip"):
        def _valid_ip(v):
            try:
                ipaddress.ip_address(str(v))
                return True
            except ValueError:
                return False
        ip_ok = df[col].apply(_valid_ip)
        if (~ip_ok).any():
            report.warnings.append(f"{(~ip_ok).sum()} rows had invalid {col}")
            valid_mask &= ip_ok

    # ports
    for col in ("src_port", "dst_port"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad_port = df[col].isna() | (df[col] < 0) | (df[col] > 65535)
        if bad_port.any():
            report.warnings.append(f"{bad_port.sum()} rows had invalid {col}")
            valid_mask &= ~bad_port

    # txid
    txid_ok = df["txid"].astype(str).str.fullmatch(r"[0-9a-fA-F]{64}")
    if (~txid_ok.fillna(False)).any():
        report.warnings.append(f"{(~txid_ok.fillna(False)).sum()} rows had malformed txid")
        valid_mask &= txid_ok.fillna(False)

    # JSON array fields + amount consistency
    def _row_json_ok(row) -> bool:
        try:
            for col in JSON_ARRAY_COLUMNS:
                row[col] = _ensure_json_array_str(row[col])
            ins = json.loads(row["input_addresses"])
            outs = json.loads(row["output_addresses"])
            in_amt = json.loads(row["input_amounts"])
            out_amt = json.loads(row["output_amounts"])
            if len(ins) != len(in_amt) or len(outs) != len(out_amt):
                return False
            if not ins or not outs:
                return False
            input_total = sum(float(x) for x in in_amt)
            output_total = sum(float(x) for x in out_amt)
            fee = float(row["fee"])
            if input_total + 1e-8 < output_total:
                return False
            if abs((input_total - output_total) - fee) > 1e-5:
                return False
            return True
        except Exception:
            return False

    json_ok = df.apply(_row_json_ok, axis=1)
    if (~json_ok).any():
        report.warnings.append(f"{(~json_ok).sum()} rows failed amount/JSON consistency checks")
        valid_mask &= json_ok

    duplicated_txid = df["txid"].duplicated(keep="first")
    if duplicated_txid.any():
        report.warnings.append(f"{duplicated_txid.sum()} duplicate txid rows dropped")
        valid_mask &= ~duplicated_txid

    clean_df = df[valid_mask].reset_index(drop=True)
    report.dropped_rows = len(df) - len(clean_df)
    report.valid_rows = len(clean_df)

    return clean_df, report

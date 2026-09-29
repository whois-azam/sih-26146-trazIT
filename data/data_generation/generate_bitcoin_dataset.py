#!/usr/bin/env python3
"""
generate_bitcoin_dataset.py

Parametric synthetic Bitcoin transaction/network metadata generator.
Phase 1 of the real-build architecture (see audit_report.md, Section 7 & 19).

Design goals (fixing the issues found in the 100-row hand-authored fixture):

  1. GENERATED, not hand-authored: any number of records can be produced by
     changing --rows and --anomaly-rate, no per-record code changes needed.
  2. NO trivial single-feature separability: amounts, fees, and IP-reuse for
     risky entities are drawn from the SAME distributions as normal entities,
     just shifted in probability, not moved to disjoint ranges.
  3. NORMAL entities get overlapping tails: some ordinary wallets legitimately
     fan out (merchant payouts), run high frequency (exchanges), or reuse one
     IP (mobile wallet users) -- the same behaviors risky wallets exhibit.
  4. Ground truth is PERSISTED to a separate file, never embedded in the
     transaction CSV.
  5. Anomaly rate is a fraction of the population (~1-3%), not a fixed count.
  6. Timestamps are interleaved across the whole window, not segregated into
     a dedicated "anomaly block".

Usage:
    python generate_bitcoin_dataset.py --rows 100000 --anomaly-rate 0.02 --seed 42

Outputs (into --outdir, default ./output):
    bitcoin_dataset.csv        transaction/network observations (no label columns)
    ground_truth.json          {wallet_address: {is_risky, scenario_type, ring_id}}
    generation_metadata.json   seed, params, counts, leakage sanity-check stats
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SAT_PER_BTC = 100_000_000

REQUIRED_COLUMNS: List[str] = [
    "timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "txid",
    "input_addresses", "output_addresses", "input_amounts", "output_amounts",
    "fee", "script_type", "geo_country", "asn",
]

SCRIPT_TYPES = ["P2WPKH", "P2TR", "P2PKH", "P2SH", "P2WSH"]
SCRIPT_WEIGHTS = [38, 24, 16, 14, 8]

BITCOIN_DST_PORTS = [8333] * 6 + [18333, 38333, 18444]

SCENARIO_TYPES = [
    "fan_out", "fan_in", "burst", "layered_chain",
    "ip_reuse", "amount_anomaly", "mixed",
]


# --------------------------------------------------------------------------
# Address / TXID / timestamp helpers
# --------------------------------------------------------------------------

def _digest_stream(label: str, length: int) -> bytes:
    out = b""
    counter = 0
    while len(out) < length:
        out += hashlib.sha256(f"{label}:{counter}".encode()).digest()
        counter += 1
    return out[:length]


def generate_wallet(label: str, rng: random.Random) -> str:
    style = "legacy" if rng.random() < 0.3 else "bc1q"
    if style == "legacy":
        chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        stream = _digest_stream("legacy:" + label, 33)
        return "1" + "".join(chars[b % len(chars)] for b in stream)
    chars = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
    stream = _digest_stream("bc1q:" + label, 38)
    return "bc1q" + "".join(chars[b % len(chars)] for b in stream)


def generate_txid(seed_material: str) -> str:
    return hashlib.sha256(seed_material.encode()).hexdigest()


def random_timestamp(rng: random.Random, start_day: int, span_days: int) -> pd.Timestamp:
    offset_seconds = rng.uniform(0, span_days * 86400)
    base = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=start_day)
    return base + pd.Timedelta(seconds=offset_seconds)


# --------------------------------------------------------------------------
# IP / GeoIP pools
# --------------------------------------------------------------------------

def build_ip_pool(n_public: int, n_private: int, rng: random.Random) -> Tuple[List[str], List[str], Dict[str, Tuple[str, int]]]:
    public_blocks = ["192.0.2.", "198.51.100.", "203.0.113."]
    private_blocks = ["10.12.", "172.16.", "192.168."]
    geo_by_block = {
        "192.0.2.": ("US", 64500), "198.51.100.": ("DE", 64512), "203.0.113.": ("SG", 64520),
    }

    public_ips, private_ips = [], []
    geoip: Dict[str, Tuple[str, int]] = {}

    for i in range(n_public):
        block = public_blocks[i % len(public_blocks)]
        octet3 = (i // len(public_blocks)) % 254 + 1
        ip = f"{block}{octet3}" if block.count(".") == 3 else f"{block}{(i % 253) + 1}"
        ip = f"{block}{rng.randint(1, 253)}"
        ip = f"{block}{(i % 253) + 1}"
        public_ips.append(ip)
        geoip[ip] = geo_by_block[block]

    for i in range(n_private):
        block = private_blocks[i % len(private_blocks)]
        sub = (i // 200) % 254 + 1
        host = (i % 200) + 1
        if block == "172.16.":
            ip = f"172.{16 + (i % 16)}.{sub}.{host}"
        else:
            ip = f"{block}{sub}.{host}"
        private_ips.append(ip)
        geoip[ip] = ("PRIVATE", 0)

    return public_ips, private_ips, geoip


def geo_for_ip(ip: str, geoip: Dict[str, Tuple[str, int]]) -> Tuple[str, object]:
    country, asn = geoip.get(ip, ("US", 64500))
    if asn == 0:
        return "PRIVATE", "PRIVATE"
    return country, asn


# --------------------------------------------------------------------------
# Amount splitting (exact integer arithmetic)
# --------------------------------------------------------------------------

def split_sats(total: int, n: int, rng: random.Random, weights: Optional[List[float]] = None) -> List[int]:
    if n == 0:
        return []
    if total <= 0:
        return [0] * n
    if weights is None:
        weights = [rng.uniform(0.4, 1.0) for _ in range(n)]
    weights = list(weights)[:n] + [1.0] * max(0, n - len(weights))
    sw = sum(weights) or 1.0
    quots, rems = [], []
    for w in weights:
        num = total * w
        q = int(num // sw)
        quots.append(q)
        rems.append(num - q * sw)
    remainder = total - sum(quots)
    order = sorted(range(n), key=lambda i: rems[i], reverse=True)
    i = 0
    while remainder > 0:
        quots[order[i % n]] += 1
        remainder -= 1
        i += 1
    return quots


def sat_to_btc(sat: int) -> float:
    return round(sat / SAT_PER_BTC, 8)


# --------------------------------------------------------------------------
# Wallet universe with behavior profiles (this is what creates overlap)
# --------------------------------------------------------------------------

@dataclass
class WalletProfile:
    address: str
    kind: str                    # retail, merchant, exchange, miner, risky
    scenario_type: Optional[str] = None
    ring_id: Optional[str] = None
    home_ip: Optional[str] = None          # if set, wallet leans toward reusing this IP
    fanout_bias: float = 0.0               # 0-1, probability of a wide-output tx
    burst_bias: float = 0.0                # 0-1, probability of tight-timing activity


def build_wallet_universe(n_wallets: int, anomaly_rate: float, rng: random.Random) -> Dict[str, WalletProfile]:
    n_risky = max(SCENARIO_TYPES.__len__(), int(n_wallets * anomaly_rate))
    kinds = (
        ["retail"] * int(n_wallets * 0.75)
        + ["merchant"] * int(n_wallets * 0.12)
        + ["exchange"] * int(n_wallets * 0.03)
        + ["miner"] * int(n_wallets * 0.02)
    )
    kinds += ["retail"] * (n_wallets - n_risky - len(kinds))  # pad
    kinds = kinds[: n_wallets - n_risky]
    kinds += ["risky"] * n_risky
    rng.shuffle(kinds)

    profiles: Dict[str, WalletProfile] = {}
    for i, kind in enumerate(kinds):
        addr = generate_wallet(f"wallet_{i:07d}", rng)
        p = WalletProfile(address=addr, kind=kind)

        # Overlapping-tail behavior: a *minority* of NORMAL wallets get the
        # same kind of trait a risky wallet would have, so no single trait
        # perfectly separates the classes.
        if kind == "merchant":
            p.fanout_bias = rng.uniform(0.15, 0.45)          # payouts to many payees
        elif kind == "exchange":
            p.fanout_bias = rng.uniform(0.25, 0.6)
            p.burst_bias = rng.uniform(0.1, 0.3)             # batch settlement runs
        elif kind == "retail" and rng.random() < 0.06:
            # a small slice of ordinary users legitimately look "bursty" or "fan-out-y"
            p.fanout_bias = rng.uniform(0.05, 0.2)
            p.burst_bias = rng.uniform(0.05, 0.15)
        if kind == "retail" and rng.random() < 0.08:
            p.home_ip = "PENDING"   # mobile-wallet-style IP stickiness, resolved later

        profiles[addr] = p

    return profiles


def assign_risky_scenarios(profiles: Dict[str, WalletProfile], rng: random.Random) -> List[dict]:
    """Group risky wallets into scenario instances (rings) with randomized intensity."""
    risky = [p for p in profiles.values() if p.kind == "risky"]
    rng.shuffle(risky)

    instances: List[dict] = []
    idx = 0
    ring_counter = 0
    while idx < len(risky):
        scenario = SCENARIO_TYPES[ring_counter % len(SCENARIO_TYPES)]
        # ring size varies by scenario: layered_chain/mixed need 2-3 wallets, others need 1
        ring_size = {"layered_chain": rng.randint(2, 3), "mixed": rng.randint(1, 2)}.get(scenario, 1)
        ring_wallets = risky[idx: idx + ring_size]
        if not ring_wallets:
            break
        idx += ring_size
        ring_id = f"ring_{ring_counter:04d}"
        for w in ring_wallets:
            w.scenario_type = scenario
            w.ring_id = ring_id
        # randomized intensity per instance -- avoids every instance of a
        # given scenario looking identical
        intensity = rng.uniform(0.5, 1.5)
        instances.append({
            "ring_id": ring_id,
            "scenario_type": scenario,
            "wallets": [w.address for w in ring_wallets],
            "intensity": intensity,
            "n_records": max(3, int(rng.uniform(4, 18) * intensity)),
        })
        ring_counter += 1

    return instances


# --------------------------------------------------------------------------
# Record builder (shared by normal + risky paths -> same amount/fee logic)
# --------------------------------------------------------------------------

def build_record(
    rng: random.Random,
    ts: pd.Timestamp,
    src_ip: str,
    dst_ip: str,
    input_addrs: List[str],
    output_addrs: List[str],
    geoip: Dict[str, Tuple[str, int]],
    amount_scale: float = 1.0,
    fee_bump: float = 1.0,
    out_weights: Optional[List[float]] = None,
) -> dict:
    # Shared lognormal amount distribution for ALL entities (normal + risky).
    # amount_scale shifts the risky "amount_anomaly" instances toward the
    # upper tail of this SAME distribution rather than a disjoint range.
    base_sats = int(np.random.lognormal(mean=16.5, sigma=1.1) * amount_scale)
    base_sats = max(50_000, min(base_sats, 2_000_000_000))

    n_in, n_out = len(input_addrs), len(output_addrs)
    input_sats = split_sats(base_sats, n_in, rng)
    input_total = sum(input_sats)

    min_out = 1000 * n_out
    if input_total <= min_out + 1000:
        input_total = min_out + 100_000
        input_sats = split_sats(input_total, n_in, rng)
        input_total = sum(input_sats)

    # Fee: shared lognormal distribution; fee_bump nudges probability, not a
    # hard rule -- most "amount_anomaly" records still get an ordinary fee.
    fee = int(np.random.lognormal(mean=10.5, sigma=0.9) * fee_bump)
    fee = max(150, min(fee, input_total - min_out))

    output_total = input_total - fee
    output_sats = split_sats(output_total, n_out, rng, out_weights)
    fee = input_total - sum(output_sats)

    src_port = rng.randint(49152, 65535)
    dst_port = rng.choice(BITCOIN_DST_PORTS)
    geo_country, asn = geo_for_ip(src_ip, geoip)
    script_type = rng.choices(SCRIPT_TYPES, weights=SCRIPT_WEIGHTS, k=1)[0]

    seed_material = "|".join([
        ts.isoformat(), src_ip, dst_ip, str(src_port), str(dst_port),
        json.dumps(input_addrs), json.dumps(output_addrs),
        json.dumps(input_sats), json.dumps(output_sats), str(fee), str(rng.random()),
    ])

    return {
        "timestamp": ts,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": src_port,
        "dst_port": dst_port,
        "txid": generate_txid(seed_material),
        "input_addresses": json.dumps(input_addrs, separators=(",", ":")),
        "output_addresses": json.dumps(output_addrs, separators=(",", ":")),
        "input_amounts": "[" + ",".join(f"{s/SAT_PER_BTC:.8f}" for s in input_sats) + "]",
        "output_amounts": "[" + ",".join(f"{s/SAT_PER_BTC:.8f}" for s in output_sats) + "]",
        "fee": sat_to_btc(fee),
        "script_type": script_type,
        "geo_country": geo_country,
        "asn": asn,
    }


# --------------------------------------------------------------------------
# Normal record generation
# --------------------------------------------------------------------------

def generate_normal_records(
    n_records: int,
    normal_wallets: List[WalletProfile],
    public_ips: List[str],
    private_ips: List[str],
    geoip: Dict[str, Tuple[str, int]],
    span_days: int,
    rng: random.Random,
) -> List[dict]:
    records: List[dict] = []
    pool_addrs = [w.address for w in normal_wallets]

    # resolve "PENDING" home IPs to a real sticky IP
    ip_choices = private_ips + public_ips
    for w in normal_wallets:
        if w.home_ip == "PENDING":
            w.home_ip = rng.choice(ip_choices)

    for _ in range(n_records):
        wallet = rng.choice(normal_wallets)
        src_ip = wallet.home_ip if (wallet.home_ip and rng.random() < 0.7) else rng.choice(ip_choices)
        dst_ip = rng.choice(public_ips)
        ts = random_timestamp(rng, 0, span_days)

        does_fanout = rng.random() < wallet.fanout_bias
        n_out = rng.randint(4, 9) if does_fanout else rng.choice([1, 1, 1, 2, 2, 3])
        outputs = rng.sample(pool_addrs, k=min(n_out, len(pool_addrs)))
        if wallet.address not in outputs:
            n_in = 1 if rng.random() < 0.75 else 2
            inputs = [wallet.address] + (rng.sample(pool_addrs, k=n_in - 1) if n_in > 1 else [])
        else:
            inputs = [wallet.address]

        rec = build_record(rng, ts, src_ip, dst_ip, inputs, outputs, geoip)
        records.append(rec)

    return records


# --------------------------------------------------------------------------
# Risky record generation, per scenario instance
# --------------------------------------------------------------------------

def generate_risky_records(
    instances: List[dict],
    normal_wallets: List[WalletProfile],
    public_ips: List[str],
    private_ips: List[str],
    geoip: Dict[str, Tuple[str, int]],
    span_days: int,
    rng: random.Random,
) -> List[dict]:
    records: List[dict] = []
    pool_addrs = [w.address for w in normal_wallets]
    all_ips = private_ips + public_ips

    for inst in instances:
        scenario = inst["scenario_type"]
        wallets = inst["wallets"]
        n = inst["n_records"]
        intensity = inst["intensity"]
        window_start = rng.uniform(0, max(1, span_days - 3))

        # sticky IP for this ring -- probabilistic reuse, not deterministic
        # 100% reuse (that alone was the trivial giveaway in the old script)
        sticky_ip = rng.choice(all_ips)
        reuse_prob = {"ip_reuse": 0.85, "mixed": 0.5}.get(scenario, 0.3)

        for k in range(n):
            primary = rng.choice(wallets)
            src_ip = sticky_ip if rng.random() < reuse_prob else rng.choice(all_ips)
            dst_ip = rng.choice(public_ips)

            if scenario == "burst":
                # tight timing window somewhere in the instance's span --
                # this IS the defining signal for this pattern by design.
                burst_day = window_start + rng.uniform(0, 1)
                ts = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=burst_day) \
                    + pd.Timedelta(minutes=rng.uniform(0, 90))
            else:
                # FIX: every other scenario used to also confine to the same
                # tight 1-3 day window, which made an entity's own observed
                # activity span artificially short -> transaction_frequency
                # (tx count / observation hours) became a near-perfect
                # single-feature giveaway (91x risky/normal ratio measured
                # in Phase 4 testing). Only "burst" should legitimately look
                # temporally concentrated; everything else spreads across
                # the full dataset span like normal wallets do.
                ts = random_timestamp(rng, 0, span_days)

            if scenario == "fan_out":
                n_out = rng.randint(5, 6 + int(4 * intensity))
                outputs = rng.sample(pool_addrs, k=min(n_out, len(pool_addrs)))
                inputs = [primary]
            elif scenario == "fan_in":
                n_in = rng.randint(3, 3 + int(3 * intensity))
                inputs = rng.sample(pool_addrs, k=min(n_in, len(pool_addrs)))
                outputs = [primary]
            elif scenario == "layered_chain":
                nxt = wallets[(wallets.index(primary) + 1) % len(wallets)] if len(wallets) > 1 else rng.choice(pool_addrs)
                inputs = [primary]
                outputs = [nxt]
            else:  # burst, ip_reuse, amount_anomaly, mixed
                # FIX: previously "primary" was ALWAYS the sender (input),
                # which made incoming_outgoing_ratio a near-giveaway too
                # (39x ratio measured). Mix sender/receiver role instead --
                # fan_out/fan_in/layered_chain keep their inherent direction
                # since that IS the definition of those patterns; these four
                # don't have an inherent direction, so they shouldn't fake one.
                if rng.random() < 0.5:
                    inputs = [primary]
                    n_out = rng.choice([1, 1, 2]) if scenario != "mixed" else rng.randint(1, 4)
                    outputs = rng.sample(pool_addrs, k=min(n_out, len(pool_addrs)))
                else:
                    outputs = [primary]
                    n_in = rng.choice([1, 1, 2]) if scenario != "mixed" else rng.randint(1, 4)
                    inputs = rng.sample(pool_addrs, k=min(n_in, len(pool_addrs)))

            # amount_anomaly nudges toward the upper tail of the SAME lognormal
            # distribution normal records use -- not a disjoint hardcoded range.
            amount_scale = rng.uniform(3.0, 8.0) if scenario in ("amount_anomaly", "mixed") and rng.random() < 0.6 else 1.0
            fee_bump = rng.uniform(1.5, 3.0) if scenario == "amount_anomaly" and rng.random() < 0.3 else 1.0

            rec = build_record(rng, ts, src_ip, dst_ip, inputs, outputs, geoip,
                                amount_scale=amount_scale, fee_bump=fee_bump)
            records.append(rec)

    return records


# --------------------------------------------------------------------------
# Validation + leakage sanity check
# --------------------------------------------------------------------------

def validate(df: pd.DataFrame) -> None:
    assert df["txid"].nunique() == len(df), "duplicate TXIDs"
    for col in REQUIRED_COLUMNS:
        assert df[col].notna().all(), f"missing values in {col}"
    for col in ("risk", "is_risky", "label", "fraud", "suspicious", "risk_score"):
        assert col not in df.columns, f"leaked label column present: {col}"
    for ip_col in ("src_ip", "dst_ip"):
        df[ip_col].apply(ipaddress.ip_address)


def generate_network_only_observations(
    all_records: List[dict],
    ground_truth: Dict[str, dict],
    public_ips: List[str],
    private_ips: List[str],
    geoip: Dict[str, Tuple[str, int]],
    rng: random.Random,
    fraction: float = 0.12,
    risky_weight: float = 2.5,
) -> Tuple[List[dict], List[dict]]:
    """
    PATCH: standalone network-layer observations -- IP + timestamp + port
    ONLY, no txid, no wallet field -- that must be probabilistically linked
    back to a transaction via the timestamp-window fallback correlation path
    (audit Section 9). Without this, direct txid-linked rows give 100%
    correlation confidence and the fallback path never gets real exercise.

    ~75% of observations are "seeded" near a real transaction's timing/IP
    (jittered, sometimes past the correlation window on purpose, so not
    every one is recoverable) and ~25% are pure background noise unrelated
    to any transaction -- exactly the ambiguity a real investigator faces.

    Risky-involving transactions are weighted 2.5x more likely to get a
    companion network observation, mirroring that suspicious actors are
    disproportionately who investigators are trying to correlate.

    Returns (network_observations, eval_truth). eval_truth records which
    txid (if any) each observation was actually seeded from -- used ONLY to
    evaluate the correlation engine afterward, never fed into it.
    """
    n_obs = int(len(all_records) * fraction)
    n_seeded = int(n_obs * 0.75)
    n_noise = n_obs - n_seeded

    weights = []
    for r in all_records:
        ins = json.loads(r["input_addresses"])
        outs = json.loads(r["output_addresses"])
        touches_risky = any(ground_truth.get(a, {}).get("is_risky") for a in set(ins) | set(outs))
        weights.append(risky_weight if touches_risky else 1.0)

    seeded_indices = rng.choices(range(len(all_records)), weights=weights, k=n_seeded)

    observations: List[dict] = []
    eval_truth: List[dict] = []
    counter = 0

    for i in seeded_indices:
        r = all_records[i]
        # jitter beyond the typical 120s correlation window ~30% of the time,
        # so not every seeded observation is actually recoverable -- realistic.
        jitter = rng.uniform(-240, 240)
        ts = r["timestamp"] + pd.Timedelta(seconds=jitter)
        ip = r["src_ip"] if rng.random() < 0.8 else r["dst_ip"]
        country, asn = geo_for_ip(ip, geoip)
        obs_id = f"netobs_{counter:07d}"
        observations.append({
            "obs_id": obs_id, "timestamp": ts, "ip": ip,
            "port": rng.choice(BITCOIN_DST_PORTS), "geo_country": country, "asn": asn,
        })
        eval_truth.append({"obs_id": obs_id, "seeded_from_txid": r["txid"], "is_noise": False})
        counter += 1

    all_ips = private_ips + public_ips
    min_ts = min(r["timestamp"] for r in all_records)
    max_ts = max(r["timestamp"] for r in all_records)
    span_seconds = (max_ts - min_ts).total_seconds()

    for _ in range(n_noise):
        ip = rng.choice(all_ips)
        ts = min_ts + pd.Timedelta(seconds=rng.uniform(0, span_seconds))
        country, asn = geo_for_ip(ip, geoip)
        obs_id = f"netobs_{counter:07d}"
        observations.append({
            "obs_id": obs_id, "timestamp": ts, "ip": ip,
            "port": rng.choice(BITCOIN_DST_PORTS), "geo_country": country, "asn": asn,
        })
        eval_truth.append({"obs_id": obs_id, "seeded_from_txid": None, "is_noise": True})
        counter += 1

    combined = list(zip(observations, eval_truth))
    rng.shuffle(combined)
    observations, eval_truth = [list(t) for t in zip(*combined)]
    return observations, eval_truth


def leakage_sanity_check(df: pd.DataFrame, ground_truth: dict) -> dict:
    """Quick check that no single field trivially separates risky vs normal."""
    risky_addrs = {a for a, v in ground_truth.items() if v["is_risky"]}

    def touches_risky(row) -> bool:
        ins = json.loads(row["input_addresses"])
        outs = json.loads(row["output_addresses"])
        return bool(risky_addrs.intersection(set(ins) | set(outs)))

    df = df.copy()
    df["_involves_risky"] = df.apply(touches_risky, axis=1)

    stats = {}
    for col in ("fee",):
        stats[f"{col}_mean_risky"] = float(df.loc[df["_involves_risky"], col].mean())
        stats[f"{col}_mean_normal"] = float(df.loc[~df["_involves_risky"], col].mean())

    def mean_amount(row):
        return float(np.mean(json.loads(row["input_amounts"])))

    amt = df.apply(mean_amount, axis=1)
    stats["amount_mean_risky"] = float(amt[df["_involves_risky"]].mean())
    stats["amount_mean_normal"] = float(amt[~df["_involves_risky"]].mean())
    stats["risky_row_fraction"] = float(df["_involves_risky"].mean())

    return stats


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--anomaly-rate", type=float, default=0.02, help="fraction of WALLETS that are risky")
    ap.add_argument("--n-wallets", type=int, default=20_000)
    ap.add_argument("--n-public-ips", type=int, default=3_500)
    ap.add_argument("--n-private-ips", type=int, default=800)
    ap.add_argument("--span-days", type=int, default=90)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="./output")
    ap.add_argument("--risky-record-fraction", type=float, default=0.15,
                     help="target fraction of ROWS involving risky entities")
    ap.add_argument("--network-obs-fraction", type=float, default=0.12,
                     help="fraction of txid-less network-only observations to generate, relative to row count")
    args = ap.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    print("Building wallet universe...")
    profiles = build_wallet_universe(args.n_wallets, args.anomaly_rate, rng)
    normal_wallets = [p for p in profiles.values() if p.kind != "risky"]

    print("Building IP pool...")
    public_ips, private_ips, geoip = build_ip_pool(args.n_public_ips, args.n_private_ips, rng)

    print("Assigning risky scenario instances...")
    instances = assign_risky_scenarios(profiles, rng)

    # scale instance record counts to roughly hit the target risky-row fraction
    target_risky_rows = int(args.rows * args.risky_record_fraction)
    current_total = sum(i["n_records"] for i in instances)
    if current_total > 0:
        scale = target_risky_rows / current_total
        for i in instances:
            i["n_records"] = max(3, int(i["n_records"] * scale))

    n_risky_records = sum(i["n_records"] for i in instances)
    n_normal_records = max(0, args.rows - n_risky_records)

    print(f"Generating {n_risky_records} risky-associated records across {len(instances)} scenario instances...")
    risky_records = generate_risky_records(instances, normal_wallets, public_ips, private_ips, geoip, args.span_days, rng)

    print(f"Generating {n_normal_records} normal records...")
    normal_records = generate_normal_records(n_normal_records, normal_wallets, public_ips, private_ips, geoip, args.span_days, rng)

    all_records = normal_records + risky_records
    all_records.sort(key=lambda r: r["timestamp"])
    # de-dup txid collisions (astronomically unlikely, but re-seed on collision to be safe)
    seen = set()
    for r in all_records:
        while r["txid"] in seen:
            r["txid"] = generate_txid(r["txid"] + str(rng.random()))
        seen.add(r["txid"])

    df = pd.DataFrame(all_records, columns=REQUIRED_COLUMNS)
    validate(df)

    ground_truth = {
        addr: {"is_risky": p.kind == "risky", "scenario_type": p.scenario_type, "ring_id": p.ring_id, "kind": p.kind}
        for addr, p in profiles.items()
    }

    print("Running leakage sanity check...")
    leak_stats = leakage_sanity_check(df, ground_truth)

    print("Generating txid-less network-only observations (fallback-correlation fixture)...")
    net_obs, net_eval_truth = generate_network_only_observations(
        all_records, ground_truth, public_ips, private_ips, geoip, rng,
        fraction=args.network_obs_fraction,
    )
    net_df = pd.DataFrame(net_obs).sort_values("timestamp").reset_index(drop=True)

    csv_path = os.path.join(args.outdir, "bitcoin_dataset.csv")
    gt_path = os.path.join(args.outdir, "ground_truth.json")
    meta_path = os.path.join(args.outdir, "generation_metadata.json")
    net_obs_path = os.path.join(args.outdir, "network_observations.csv")
    net_eval_path = os.path.join(args.outdir, "network_observation_eval_truth.json")

    df.to_csv(csv_path, index=False)
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f)
    net_df.to_csv(net_obs_path, index=False)
    with open(net_eval_path, "w") as f:
        json.dump(net_eval_truth, f)
    with open(meta_path, "w") as f:
        json.dump({
            "seed": args.seed, "rows": len(df), "n_wallets": args.n_wallets,
            "anomaly_rate": args.anomaly_rate, "n_risky_wallets": sum(1 for p in profiles.values() if p.kind == "risky"),
            "n_scenario_instances": len(instances), "span_days": args.span_days,
            "leakage_sanity_check": leak_stats,
            "network_only_observations": len(net_df),
            "network_only_seeded": sum(1 for t in net_eval_truth if not t["is_noise"]),
            "network_only_noise": sum(1 for t in net_eval_truth if t["is_noise"]),
        }, f, indent=2)

    print("========================================")
    print("SYNTHETIC BITCOIN DATASET (parametric)")
    print("========================================")
    print(f"Rows: {len(df)}")
    print(f"Unique TXIDs: {df['txid'].nunique()}")
    print(f"Wallets: {args.n_wallets}  (risky: {sum(1 for p in profiles.values() if p.kind=='risky')}, rate={args.anomaly_rate})")
    print(f"Scenario instances: {len(instances)}")
    print(f"Risky-involved rows: ~{leak_stats['risky_row_fraction']:.1%}")
    print()
    print("Leakage sanity check (risky vs normal -- should be close, NOT wildly separated):")
    for k, v in leak_stats.items():
        if k != "risky_row_fraction":
            print(f"  {k}: {v:.6f}")
    print()
    print(f"Network-only observations: {len(net_df)} (seeded: {sum(1 for t in net_eval_truth if not t['is_noise'])}, noise: {sum(1 for t in net_eval_truth if t['is_noise'])})")
    print()
    print(f"Output CSV: {csv_path}")
    print(f"Ground truth: {gt_path}")
    print(f"Metadata: {meta_path}")
    print(f"Network observations: {net_obs_path}")
    print(f"Network observation eval truth: {net_eval_path}")


if __name__ == "__main__":
    main()

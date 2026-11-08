# TrazIT  Offline Bitcoin Transaction Forensics

TrazIT is a fully offline system for analyzing Bitcoin transaction data. You upload a CSV, JSON, or XML file of transaction records, and the system runs them through a pipeline of anomaly detection, graph analysis, behavioral clustering, and GeoIP enrichment, then surfaces the results as a ranked, evidence-backed list of entities that warrant investigation.

Nothing calls home. No API keys. No cloud model. The entire pipeline — including a locally hosted LLM for evidence Q&A — runs on your machine.

---

## What it actually does

When you upload transaction data, TrazIT:

1. **Validates and ingests** — checks referential integrity, normalizes timestamps, handles multiple input formats with schema aliasing so column names don't need to match exactly.

2. **Correlates network and blockchain trails** — where a TXID link exists, the correlation is certain. Where it doesn't, the system uses a timestamp-window fallback and explicitly reports a confidence score (which degrades as ambiguity increases). A network observation is always reported as "this IP was observed adjacent to this wallet's activity, confidence X," never as an identity claim.

3. **Builds a transaction graph** — a NetworkX multigraph with five node types (wallet, transaction, IP, ASN, country) and five edge types. Graph construction for 100k transactions takes about 17 seconds.

4. **Extracts entity-level features** — 33+ features per wallet covering transaction volume, fan-in/fan-out, fee patterns, temporal burstiness, and network/geographic diversity.

5. **Scores anomalies** — IsolationForest trained on a 14,000-entity reference population, with frozen-reference calibration so the same entity gets the same score regardless of what else is in the upload batch. Scores are percentile-ranked against that fixed reference, not re-normalized per batch.

6. **Clusters entities two ways** — behavioral clustering (HDBSCAN on feature vectors, tells you who acts alike) and graph community detection (Louvain algorithm, tells you who transacts with whom). These are reported separately. An entity that's behaviorally similar to flagged wallets but graph-disconnected from them is a different investigative situation from one that's directly connected.

7. **Scores risk** — one formula, four weighted components:

   ```
   risk = 100 × clip(
     0.5786 × anomaly_score
   + 0.1973 × flagged_neighbor_score
   + 0.1259 × graph_centrality_percentile
   + 0.0982 × behavioral_cluster_outlier_flag
   , 0, 1)
   ```

   Weights are logistic regression coefficients calibrated against synthetic ground truth, not hand-picked. Thresholds (CRITICAL ≥ 78.6, HIGH ≥ 65.9, MEDIUM ≥ 57.9) come from the training population's score distribution.

8. **Generates evidence** — four categories: feature-level deviations, model output, graph facts, and temporal patterns. No padding to a fixed count. If two signals clear threshold, you see two.

9. **Checks for model drift** — a two-sample KS test compares the upload's score distribution against the training reference. If the distributions diverge meaningfully, the dashboard shows a warning rather than silently serving stale scores.

10. **Answers questions** — a local Qwen3-1.7B model (GGUF, runs via llama-cpp-python) handles Q&A about flagged entities. It only answers from the computed evidence for that entity. Any numeric claim in its response that doesn't trace back to the evidence is rejected before it reaches the user.

---

## Evaluation numbers

These measure fit to the synthetic dataset's own ground truth — see the note in `evaluation/phase8_consolidated_metrics.json` about circularity.

| Metric | Value |
|--------|-------|
| Anomaly detection PR-AUC (test) | 0.79 |
| Anomaly detection ROC-AUC (test) | 0.99 |
| Precision@50 alerts | 0.80 |
| Precision@100 alerts | 0.51 |
| Graph ring recovery rate | 83.3% (55 of 66 synthetic rings fully contained in one community) |
| Louvain communities found | 52 |
| Score stability | Identical across all batch sizes (fixed in prior prototype) |

---

## Project structure

```
btc_forensics/
├── ingestion/          Multi-format loader with schema aliasing and validation
├── correlation/        TXID-direct and timestamp-window fallback correlation
├── graph/              NetworkX graph builder + centrality + Louvain community detection
├── features/           33-feature entity extractor (label-blind)
├── ml/                 IsolationForest (anomaly) + HDBSCAN (behavioral clustering)
│   └── artifacts/      Trained model files (not committed, regenerate or download separately)
├── scoring/            Single risk formula with component breakdown
├── explainability/     Four-category evidence generator
├── innovations/        Drift check, temporal re-scoring, alert chains, divergence flagging
├── geoip/              Offline MaxMind GeoLite2 enrichment
│   └── data/           .mmdb files (not committed, see geoip/geoip_README.md)
├── llm/                Qwen3-1.7B GGUF query layer with grounding and numeric verification
│   └── models/         Model file (not committed, ~1.2 GB, see llm/llm_README.md)
├── pipeline/           Single inference entry point — upload → results
├── dashboard/          Streamlit app (three tabs: Overview, Entity Investigation, Clusters)
├── data/
│   ├── data_generation/ 100k-row synthetic dataset generator
│   ├── demo/           Small demo CSV for quick testing
│   └── processed/      risk_evaluation.json (risk weights/thresholds, committed)
├── evaluation/         Benchmark scripts and metrics
├── submit/             SIH submission dataset (trazit_sih_100.csv)
└── requirements.txt
```

---

## Setup

**Python 3.11+** required.

```bash
pip install -r requirements.txt
pip install maxminddb hdbscan llama-cpp-python
```

**GeoIP databases** — download GeoLite2-Country.mmdb and GeoLite2-ASN.mmdb from MaxMind (free account required) and place them in `geoip/data/`. The system works without them but won't resolve country/ASN.

**LLM model** — download `Qwen3-1.7B-Q4_K_M.gguf` (or any compatible GGUF) and place it in `llm/models/`. See `llm/llm_README.md`. The dashboard works without it, but the AI Q&A tab won't load.

**ML artifacts** — either run the full pipeline to generate them (`ml/run_phase5.py` then `scoring/run_phase6.py`), or obtain pre-trained artifacts separately.

**Run the dashboard:**

```bash
cd dashboard
streamlit run app.py
```

---

## Data format

Upload a CSV, JSON, or XML file with these fields (column names are flexible — the loader handles common aliases):

| Field | Required | Notes |
|-------|----------|-------|
| `txid` | Yes | Unique transaction ID |
| `sender` | Yes | Sending wallet address |
| `receiver` | Yes | Receiving wallet address |
| `amount` | Yes | In BTC |
| `fee` | Yes | In BTC |
| `timestamp` | Yes | Parseable datetime |
| `input_total` | Yes | Total input value |
| `output_total` | Yes | Total output value |
| `ip_address` | No | Source IP if available |
| `script_type` | No | e.g. P2PKH, P2SH |

A 121-row demo file is at `submit/trazit_sih_100.csv`.

---

## What this is not

TrazIT is not a blockchain node or indexer — it works on uploaded data files, not live chain data. Risk scores are prioritization signals for human investigators, not determinations of guilt. Every evidence statement traces to a specific computed value, and the system never presents a network observation as a confirmed identity link.

---

## Dependencies

`pandas`, `numpy`, `scikit-learn`, `networkx`, `plotly`, `streamlit`, `pyyaml`, `maxminddb`, `hdbscan`, `joblib`, `scipy`, `llama-cpp-python`

Built for Python 3.11. Tested onLinux (Ubuntu)

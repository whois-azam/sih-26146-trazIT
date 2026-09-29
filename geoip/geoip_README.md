# GeoIP Setup — `geoip/` directory

Offline GeoIP/ASN enrichment for **real** IP addresses. Not needed for the synthetic dataset (it already carries its own `geo_country`/`asn` fields from the generator) — this is for when the pipeline processes real network captures.

**You already have these two files downloaded** — this just tells you where they go.

## Directory layout

```
geoip/
├── data/
│   ├── GeoLite2-Country.mmdb   <- drop your existing file here
│   └── GeoLite2-ASN.mmdb        <- drop your existing file here
├── enrichment.py                 (matches the exact schema you already verified working)
└── README.md                     (this file)
```

Just copy both `.mmdb` files straight into `geoip/data/` — no renaming needed, `enrichment.py` looks for those exact filenames.

## Test it

```bash
pip install maxminddb --break-system-packages
cd geoip
python3 enrichment.py
```

Expected output (same test IPs, same format you already confirmed working):
```
10.8.174.72 -> COUNTRY=PRIVATE/RESERVED/LOCAL ASN=None ORG=None
127.0.0.1 -> COUNTRY=PRIVATE/RESERVED/LOCAL ASN=None ORG=None
203.0.113.8 -> COUNTRY=PRIVATE/RESERVED/LOCAL ASN=None ORG=None
8.8.8.8 -> COUNTRY=US ASN=15169 ORG=Google LLC
1.1.1.1 -> COUNTRY=None ASN=13335 ORG=Cloudflare, Inc.
```

(`1.1.1.1` showing `COUNTRY=None` is expected, not a bug — the Country database just doesn't have a record for that address; ASN still resolves fine.)


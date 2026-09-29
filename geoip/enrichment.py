"""
geoip/enrichment.py

Offline GeoIP/ASN enrichment for REAL IP addresses, matching the exact
database pair already verified working:
    geoip/data/GeoLite2-Country.mmdb
    geoip/data/GeoLite2-ASN.mmdb

Uses `maxminddb` directly (lighter than the `geoip2` wrapper, and matches
what was already proven working) rather than a City-database API.

Degrades gracefully when either file is absent: returns None fields rather
than raising.
"""

from __future__ import annotations

import ipaddress
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
COUNTRY_DB_PATH = os.path.join(DATA_DIR, "GeoLite2-Country.mmdb")
ASN_DB_PATH = os.path.join(DATA_DIR, "GeoLite2-ASN.mmdb")


class GeoIPEnricher:
    def __init__(self, country_db_path: str = COUNTRY_DB_PATH, asn_db_path: str = ASN_DB_PATH):
        import maxminddb

        self.country_db_path = country_db_path
        self.asn_db_path = asn_db_path
        self._country_db = maxminddb.open_database(country_db_path) if os.path.exists(country_db_path) else None
        self._asn_db = maxminddb.open_database(asn_db_path) if os.path.exists(asn_db_path) else None

    @property
    def available(self) -> bool:
        return self._country_db is not None or self._asn_db is not None

    def lookup(self, ip: str) -> dict:
        """Returns {ip, country, asn, asn_org}. PRIVATE/RESERVED/LOCAL for
        non-global addresses (10.x/127.x/192.168.x/documentation ranges
        like 203.0.113.x all report as private -- expected, not a bug)."""
        result = {"ip": ip, "country": None, "asn": None, "asn_org": None}

        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return result

        if not addr.is_global:
            result["country"] = "PRIVATE/RESERVED/LOCAL"
            return result

        if self._country_db is not None:
            rec = self._country_db.get(ip) or {}
            result["country"] = rec.get("country", {}).get("iso_code")

        if self._asn_db is not None:
            rec = self._asn_db.get(ip) or {}
            result["asn"] = rec.get("autonomous_system_number")
            result["asn_org"] = rec.get("autonomous_system_organization")

        return result

    def close(self):
        if self._country_db is not None:
            self._country_db.close()
        if self._asn_db is not None:
            self._asn_db.close()


def enrich_dataframe(df, ip_column: str = "src_ip", country_db_path: str = COUNTRY_DB_PATH, asn_db_path: str = ASN_DB_PATH):
    """Bulk-enrich a DataFrame's IP column."""
    enricher = GeoIPEnricher(country_db_path, asn_db_path)
    if not enricher.available:
        df[f"{ip_column}_country"] = None
        df[f"{ip_column}_asn"] = None
        df[f"{ip_column}_asn_org"] = None
        return df

    results = df[ip_column].apply(enricher.lookup)
    df[f"{ip_column}_country"] = results.apply(lambda r: r["country"])
    df[f"{ip_column}_asn"] = results.apply(lambda r: r["asn"])
    df[f"{ip_column}_asn_org"] = results.apply(lambda r: r["asn_org"])
    enricher.close()
    return df


if __name__ == "__main__":
    enricher = GeoIPEnricher()
    print(f"Country DB: {COUNTRY_DB_PATH} (available: {enricher._country_db is not None})")
    print(f"ASN DB:     {ASN_DB_PATH} (available: {enricher._asn_db is not None})")
    print()
    test_ips = ["10.8.174.72", "127.0.0.1", "203.0.113.8", "8.8.8.8", "1.1.1.1"]
    for ip in test_ips:
        r = enricher.lookup(ip)
        print(f"  {ip} -> COUNTRY={r['country']} ASN={r['asn']} ORG={r['asn_org']}")

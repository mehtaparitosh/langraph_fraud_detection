"""Synthetic payment-data generator.

Produces six reference/transaction CSVs under ``data/seed/`` that are committed
to the repo. ``seed.py`` later loads them into a local SQLite file.

Design goals:
  * **Deterministic** — a fixed RNG seed and a fixed reference "now" so the
    generated seeds are stable across machines and never age against the wall
    clock. (Velocity windows are evaluated relative to a transaction's own
    timestamp, not the real current time.)
  * **Explainable** — a handful of clearly labelled fraud patterns so the
    Phase 2 rule engine and ML model have real signal.
  * **Crafted rows** — four hand-built transactions (clean, blocklisted,
    velocity spike, high-value grey-zone) that every later phase and test
    references by the constants exported here.

Run:  ``uv run python -m fraudgraph.data.generate``
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

from fraudgraph import config

# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------
RNG_SEED = 42
# Fixed anchor so seed timestamps never age. Everything is generated relative
# to this instant; the four crafted demo transactions sit exactly here.
REFERENCE_NOW = datetime(2026, 9, 1, 12, 0, 0)

N_CARDS = 200
N_DEVICES = 160
N_MERCHANTS = 40
N_IPS = 120
N_NORMAL_TXNS = 2500

GEOS = [
    "IN-Mumbai", "IN-Delhi", "IN-Bangalore", "IN-Chennai", "IN-Kolkata",
    "IN-Hyderabad", "IN-Pune", "IN-Ahmedabad",
]
FOREIGN_GEOS = ["US-NewYork", "GB-London", "SG-Singapore", "AE-Dubai", "RU-Moscow"]

MERCHANT_CATEGORIES = {
    "grocery": "low", "utilities": "low", "restaurant": "low", "retail": "low",
    "travel": "medium", "electronics": "medium", "jewellery": "medium",
    "gambling": "high", "crypto": "high", "gift_cards": "high",
}

# --------------------------------------------------------------------------
# Crafted entities — stable IDs referenced by tests and demo inputs.
# --------------------------------------------------------------------------
# Clean, low-risk transaction.
CLEAN = {
    "txn_id": "TXN_CLEAN", "card_id": "CARD_CLEAN", "device_id": "DEV_CLEAN",
    "ip": "10.0.0.10", "merchant_id": "MERCH_GROCERY", "amount": 1250.0,
    "geo": "IN-Mumbai", "label": 0,
}
# Card present on the blocklist -> forced high / auto-decline.
BLOCKED = {
    "txn_id": "TXN_BLOCKED", "card_id": "CARD_BLOCKED", "device_id": "DEV_BLOCKED",
    "ip": "203.0.113.66", "merchant_id": "MERCH_CRYPTO", "amount": 9800.0,
    "geo": "RU-Moscow", "label": 1,
}
# Same card fired many times inside the velocity window. Deliberately kept
# clean on every OTHER axis (non-proxy device, medium merchant, domestic geo,
# clean IP) so velocity is the *sole* strong signal -> lands in the grey band.
VELOCITY = {
    "txn_id": "TXN_VELOCITY", "card_id": "CARD_VELOCITY", "device_id": "DEV_VELOCITY",
    "ip": "10.0.0.30", "merchant_id": "MERCH_ELECTRONICS", "amount": 4500.0,
    "geo": "IN-Delhi", "label": 1,
}
# Number of transactions in the velocity burst (well over VELOCITY_MAX_TXNS).
VELOCITY_BURST = 6
# High-value but otherwise clean -> grey-zone by amount, routes to investigate.
HIGH_VALUE = {
    "txn_id": "TXN_HIVAL", "card_id": "CARD_HIVAL", "device_id": "DEV_HIVAL",
    "ip": "10.0.0.20", "merchant_id": "MERCH_JEWELLERY", "amount": 250000.0,
    "geo": "IN-Bangalore", "label": 0,
}

# Blocklist seed values (entity_type, value, reason).
BLOCKLIST_SEED = [
    ("card", "CARD_BLOCKED", "confirmed_fraud_ring"),
    ("card", "CARD_BAD_2", "chargeback_abuse"),
    ("device", "DEV_BLOCKED", "known_fraud_device"),
    ("device", "DEV_BAD_2", "device_farm"),
    ("ip", "203.0.113.66", "tor_exit_node"),
    ("ip", "203.0.113.99", "botnet_c2"),
]


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _make_ip(rng: random.Random) -> str:
    return f"{rng.randint(1, 223)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"


def _build_reference(rng: random.Random):
    """Build cards, devices, ip_reputation, merchants reference rows."""
    cards = []
    for i in range(N_CARDS):
        cards.append({
            "card_id": f"CARD_{i:04d}",
            "bank": rng.choice(["HDFC", "ICICI", "SBI", "Axis", "Kotak"]),
            "country": "IN",
            "home_geo": rng.choice(GEOS),
            "holder_email": f"user{i:04d}@example.com",
            "status": "active",
        })

    devices = []
    for i in range(N_DEVICES):
        devices.append({
            "device_id": f"DEV_{i:04d}",
            "os": rng.choice(["Android", "iOS", "Windows", "macOS"]),
            "proxy_flag": 1 if rng.random() < 0.10 else 0,
            "first_seen": _iso(REFERENCE_NOW - timedelta(days=rng.randint(1, 720))),
        })

    ips = []
    for _ in range(N_IPS):
        ip = _make_ip(rng)
        risky = rng.random() < 0.15
        ips.append({
            "ip": ip,
            "risk_score": round(rng.uniform(0.6, 0.99), 2) if risky else round(rng.uniform(0.0, 0.3), 2),
            "is_proxy": 1 if risky and rng.random() < 0.6 else 0,
            "country": rng.choice(["IN", "US", "GB", "SG", "RU"]) if risky else "IN",
        })

    merchants = []
    cat_names = list(MERCHANT_CATEGORIES.keys())
    for i in range(N_MERCHANTS):
        cat = cat_names[i % len(cat_names)]
        risk = MERCHANT_CATEGORIES[cat]
        cb = {"low": rng.uniform(0.0, 0.01), "medium": rng.uniform(0.01, 0.04),
              "high": rng.uniform(0.05, 0.15)}[risk]
        merchants.append({
            "merchant_id": f"MERCH_{i:04d}",
            "name": f"{cat.title()} Store {i}",
            "category": cat,
            "risk_level": risk,
            "chargeback_rate": round(cb, 4),
        })

    return cards, devices, ips, merchants


def _append_crafted_reference(cards, devices, ips, merchants):
    """Add reference rows for the crafted entities so lookups resolve."""
    cards.extend([
        {"card_id": "CARD_CLEAN", "bank": "HDFC", "country": "IN",
         "home_geo": "IN-Mumbai", "holder_email": config.DEMO_USER_EMAIL or "demo@example.com",
         "status": "active"},
        {"card_id": "CARD_BLOCKED", "bank": "SBI", "country": "IN",
         "home_geo": "IN-Delhi", "holder_email": "blocked@example.com", "status": "active"},
        {"card_id": "CARD_VELOCITY", "bank": "ICICI", "country": "IN",
         "home_geo": "IN-Delhi", "holder_email": "velocity@example.com", "status": "active"},
        {"card_id": "CARD_HIVAL", "bank": "Axis", "country": "IN",
         "home_geo": "IN-Bangalore", "holder_email": config.DEMO_USER_EMAIL or "demo@example.com",
         "status": "active"},
    ])
    devices.extend([
        {"device_id": "DEV_CLEAN", "os": "iOS", "proxy_flag": 0,
         "first_seen": _iso(REFERENCE_NOW - timedelta(days=400))},
        {"device_id": "DEV_BLOCKED", "os": "Android", "proxy_flag": 1,
         "first_seen": _iso(REFERENCE_NOW - timedelta(days=3))},
        {"device_id": "DEV_VELOCITY", "os": "Android", "proxy_flag": 0,
         "first_seen": _iso(REFERENCE_NOW - timedelta(days=90))},
        {"device_id": "DEV_HIVAL", "os": "macOS", "proxy_flag": 0,
         "first_seen": _iso(REFERENCE_NOW - timedelta(days=500))},
    ])
    ips.extend([
        {"ip": "10.0.0.10", "risk_score": 0.02, "is_proxy": 0, "country": "IN"},
        {"ip": "10.0.0.20", "risk_score": 0.03, "is_proxy": 0, "country": "IN"},
        {"ip": "10.0.0.30", "risk_score": 0.04, "is_proxy": 0, "country": "IN"},
        {"ip": "203.0.113.66", "risk_score": 0.97, "is_proxy": 1, "country": "RU"},
        {"ip": "198.51.100.23", "risk_score": 0.71, "is_proxy": 1, "country": "IN"},
    ])
    merchants.extend([
        {"merchant_id": "MERCH_GROCERY", "name": "Daily Grocery", "category": "grocery",
         "risk_level": "low", "chargeback_rate": 0.004},
        {"merchant_id": "MERCH_CRYPTO", "name": "CoinBazaar", "category": "crypto",
         "risk_level": "high", "chargeback_rate": 0.11},
        {"merchant_id": "MERCH_GIFTCARDS", "name": "GiftCard Hub", "category": "gift_cards",
         "risk_level": "high", "chargeback_rate": 0.09},
        {"merchant_id": "MERCH_JEWELLERY", "name": "Golden Jewels", "category": "jewellery",
         "risk_level": "medium", "chargeback_rate": 0.02},
        {"merchant_id": "MERCH_ELECTRONICS", "name": "Gadget World", "category": "electronics",
         "risk_level": "medium", "chargeback_rate": 0.018},
    ])


def _generate_transactions(rng, cards, devices, merchants, ips):
    """Generate normal + fraud-pattern transactions, then the crafted rows."""
    card_ids = [c["card_id"] for c in cards if c["card_id"].startswith("CARD_") and c["card_id"][5:].isdigit()]
    device_ids = [d["device_id"] for d in devices if d["device_id"][4:].isdigit()]
    merchant_ids = [m["merchant_id"] for m in merchants if m["merchant_id"][6:].isdigit()]
    ip_pool = [i["ip"] for i in ips]
    home = {c["card_id"]: c["home_geo"] for c in cards}

    txns = []
    n = 0
    for _ in range(N_NORMAL_TXNS):
        card = rng.choice(card_ids)
        # ~4% of rows carry a deliberate fraud pattern -> label 1.
        is_fraud = rng.random() < 0.04
        ts = REFERENCE_NOW - timedelta(minutes=rng.randint(5, 30 * 24 * 60))
        if is_fraud:
            geo = rng.choice(FOREIGN_GEOS)
            amount = round(rng.uniform(20000, 120000), 2)
            device = rng.choice(device_ids)
            merchant = rng.choice([m["merchant_id"] for m in merchants
                                   if m["risk_level"] == "high"] or merchant_ids)
            ip = rng.choice(ip_pool)
            label = 1
        else:
            geo = home[card] if rng.random() < 0.9 else rng.choice(GEOS)
            amount = round(rng.lognormvariate(6.5, 0.9), 2)  # ~ hundreds to few-thousand
            device = rng.choice(device_ids)
            merchant = rng.choice(merchant_ids)
            ip = rng.choice(ip_pool)
            label = 0
        txns.append({
            "txn_id": f"TXN_{n:06d}", "card_id": card, "device_id": device,
            "ip": ip, "merchant_id": merchant, "amount": amount,
            "ts": _iso(ts), "geo": geo, "label": label,
        })
        n += 1

    # --- Crafted: clean ---
    txns.append({**{k: CLEAN[k] for k in
                    ("txn_id", "card_id", "device_id", "ip", "merchant_id", "amount", "geo", "label")},
                 "ts": _iso(REFERENCE_NOW)})

    # --- Crafted: blocklisted card ---
    txns.append({**{k: BLOCKED[k] for k in
                    ("txn_id", "card_id", "device_id", "ip", "merchant_id", "amount", "geo", "label")},
                 "ts": _iso(REFERENCE_NOW)})

    # --- Crafted: velocity burst (VELOCITY_BURST txns within the window) ---
    # Spread evenly across the configured window so the whole burst falls inside
    # it when evaluated as-of the final (incoming) transaction's timestamp.
    step = max(1, config.VELOCITY_WINDOW_MIN // (VELOCITY_BURST + 1))
    for k in range(VELOCITY_BURST - 1):
        ts = REFERENCE_NOW - timedelta(minutes=step * (VELOCITY_BURST - k))
        txns.append({
            "txn_id": f"TXN_VELOCITY_{k}", "card_id": VELOCITY["card_id"],
            "device_id": VELOCITY["device_id"], "ip": VELOCITY["ip"],
            "merchant_id": VELOCITY["merchant_id"],
            "amount": round(rng.uniform(2000, 6000), 2),
            "ts": _iso(ts), "geo": VELOCITY["geo"], "label": 1,
        })
    # The incoming transaction sits at the end of the burst.
    txns.append({**{k: VELOCITY[k] for k in
                    ("txn_id", "card_id", "device_id", "ip", "merchant_id", "amount", "geo", "label")},
                 "ts": _iso(REFERENCE_NOW)})

    # --- Crafted: high-value grey-zone ---
    txns.append({**{k: HIGH_VALUE[k] for k in
                    ("txn_id", "card_id", "device_id", "ip", "merchant_id", "amount", "geo", "label")},
                 "ts": _iso(REFERENCE_NOW)})

    return txns


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate(seed_dir: Path | None = None) -> dict[str, Path]:
    """Generate all seed CSVs. Returns a map of table name -> written path."""
    seed_dir = seed_dir or config.SEED_DIR
    rng = random.Random(RNG_SEED)

    cards, devices, ips, merchants = _build_reference(rng)
    _append_crafted_reference(cards, devices, ips, merchants)
    txns = _generate_transactions(rng, cards, devices, merchants, ips)

    blocklist = [
        {"entity_type": t, "value": v, "reason": r, "added_ts": _iso(REFERENCE_NOW - timedelta(days=10))}
        for (t, v, r) in BLOCKLIST_SEED
    ]

    outputs = {
        "transactions": (seed_dir / "transactions.csv", txns,
                         ["txn_id", "card_id", "device_id", "ip", "merchant_id", "amount", "ts", "geo", "label"]),
        "cards": (seed_dir / "cards.csv", cards,
                  ["card_id", "bank", "country", "home_geo", "holder_email", "status"]),
        "devices": (seed_dir / "devices.csv", devices,
                    ["device_id", "os", "proxy_flag", "first_seen"]),
        "ip_reputation": (seed_dir / "ip_reputation.csv", ips,
                          ["ip", "risk_score", "is_proxy", "country"]),
        "merchants": (seed_dir / "merchants.csv", merchants,
                      ["merchant_id", "name", "category", "risk_level", "chargeback_rate"]),
        "blocklist": (seed_dir / "blocklist.csv", blocklist,
                      ["entity_type", "value", "reason", "added_ts"]),
    }
    written = {}
    for name, (path, rows, fields) in outputs.items():
        _write_csv(path, rows, fields)
        written[name] = path
    return written


def main() -> None:
    written = generate()
    print("Generated seed CSVs:")
    for name, path in written.items():
        n = sum(1 for _ in path.open()) - 1
        print(f"  {name:14s} -> {path.relative_to(config.PROJECT_ROOT)}  ({n} rows)")


if __name__ == "__main__":
    main()

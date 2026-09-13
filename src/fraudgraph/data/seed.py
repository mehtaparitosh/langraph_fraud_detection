"""Create the SQLite schema and load it from the committed seed CSVs.

Reads ``data/seed/*.csv`` (generating them first if missing) and writes the
gitignored ``data/fraud_demo.db``. Idempotent: rebuilds tables each run.

Run:  ``uv run python -m fraudgraph.data.seed``
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from fraudgraph import config
from fraudgraph.data import generate

SCHEMA = """
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS cards;
DROP TABLE IF EXISTS devices;
DROP TABLE IF EXISTS ip_reputation;
DROP TABLE IF EXISTS merchants;
DROP TABLE IF EXISTS blocklist;

CREATE TABLE transactions (
    txn_id      TEXT PRIMARY KEY,
    card_id     TEXT NOT NULL,
    device_id   TEXT,
    ip          TEXT,
    merchant_id TEXT,
    amount      REAL NOT NULL,
    ts          TEXT NOT NULL,      -- ISO 'YYYY-MM-DD HH:MM:SS'
    geo         TEXT,
    label       INTEGER NOT NULL    -- 0 legit, 1 fraud
);

CREATE TABLE cards (
    card_id      TEXT PRIMARY KEY,
    bank         TEXT,
    country      TEXT,
    home_geo     TEXT,
    holder_email TEXT,
    status       TEXT
);

CREATE TABLE devices (
    device_id  TEXT PRIMARY KEY,
    os         TEXT,
    proxy_flag INTEGER NOT NULL DEFAULT 0,
    first_seen TEXT
);

CREATE TABLE ip_reputation (
    ip         TEXT PRIMARY KEY,
    risk_score REAL,
    is_proxy   INTEGER NOT NULL DEFAULT 0,
    country    TEXT
);

CREATE TABLE merchants (
    merchant_id     TEXT PRIMARY KEY,
    name            TEXT,
    category        TEXT,
    risk_level      TEXT,
    chargeback_rate REAL
);

CREATE TABLE blocklist (
    entity_type TEXT NOT NULL,      -- 'card' | 'device' | 'ip'
    value       TEXT NOT NULL,
    reason      TEXT,
    added_ts    TEXT,
    PRIMARY KEY (entity_type, value)
);

CREATE INDEX idx_txn_card_ts ON transactions (card_id, ts);
CREATE INDEX idx_txn_device  ON transactions (device_id);
CREATE INDEX idx_txn_merchant ON transactions (merchant_id);
CREATE INDEX idx_blocklist_value ON blocklist (value);
"""

# CSV file -> (table, columns) mapping.
_LOADS = {
    "transactions.csv": ("transactions",
                         ["txn_id", "card_id", "device_id", "ip", "merchant_id", "amount", "ts", "geo", "label"]),
    "cards.csv": ("cards", ["card_id", "bank", "country", "home_geo", "holder_email", "status"]),
    "devices.csv": ("devices", ["device_id", "os", "proxy_flag", "first_seen"]),
    "ip_reputation.csv": ("ip_reputation", ["ip", "risk_score", "is_proxy", "country"]),
    "merchants.csv": ("merchants", ["merchant_id", "name", "category", "risk_level", "chargeback_rate"]),
    "blocklist.csv": ("blocklist", ["entity_type", "value", "reason", "added_ts"]),
}


def _load_csv(conn: sqlite3.Connection, csv_path: Path, table: str, cols: list[str]) -> int:
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [tuple(r[c] for c in cols) for r in reader]
    placeholders = ", ".join("?" for _ in cols)
    conn.executemany(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})", rows
    )
    return len(rows)


def seed(db_path: Path | None = None, seed_dir: Path | None = None,
         regenerate: bool = False) -> Path:
    """Build the SQLite database from the seed CSVs.

    Generates the CSVs first if they are missing (or ``regenerate=True``).
    """
    db_path = db_path or config.DB_PATH
    seed_dir = seed_dir or config.SEED_DIR

    if regenerate or not (seed_dir / "transactions.csv").exists():
        generate.generate(seed_dir)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        counts = {}
        for fname, (table, cols) in _LOADS.items():
            counts[table] = _load_csv(conn, seed_dir / fname, table, cols)
        conn.commit()
    finally:
        conn.close()

    seed.last_counts = counts  # type: ignore[attr-defined]
    return db_path


def main() -> None:
    db = seed(regenerate=True)
    print(f"Seeded {db.relative_to(config.PROJECT_ROOT)}")
    for table, n in getattr(seed, "last_counts", {}).items():
        print(f"  {table:14s} {n} rows")


if __name__ == "__main__":
    main()

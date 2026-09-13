"""Read-only evidence lookups over the SQLite database.

Each function is pure (no writes, no global state beyond opening a connection)
and returns plain dict/int evidence the enrichment node can drop straight into
``FraudState.evidence``. Nothing here makes a decision — these only surface
facts.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from fraudgraph import config

_TS_FMT = "%Y-%m-%d %H:%M:%S"


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def gather_evidence(transaction: dict, db_path: Path | None = None) -> dict:
    """Run all four lookups for a transaction into one canonical evidence dict.

    This is the exact shape ``FraudState.evidence`` carries and that the scoring
    engine (Phase 2) and the enrich node (Phase 3) both consume::

        {
          "blocklist": {"hit": bool, "matches": [...]},
          "velocity":  int,
          "device":    {...device_reputation...},
          "merchant":  {...merchant_reputation...},
        }

    Velocity is anchored to the transaction's own ``ts`` so it means "recent
    activity as of this transaction", independent of the wall clock.
    """
    return {
        "blocklist": check_blocklist(
            transaction["card_id"], transaction.get("device_id"),
            transaction.get("ip"), db_path=db_path,
        ),
        "velocity": velocity(
            transaction["card_id"], as_of=transaction.get("ts"), db_path=db_path,
        ),
        "device": device_reputation(transaction.get("device_id"), db_path=db_path),
        "merchant": merchant_reputation(transaction.get("merchant_id"), db_path=db_path),
    }


def check_blocklist(card_id: str, device_id: str, ip: str,
                    db_path: Path | None = None) -> dict:
    """Is any of the card / device / IP on the blocklist?

    Returns ``{"hit": bool, "matches": [{type, value, reason}, ...]}``.
    """
    conn = _connect(db_path)
    try:
        pairs = [("card", card_id), ("device", device_id), ("ip", ip)]
        matches = []
        for entity_type, value in pairs:
            if value is None:
                continue
            row = conn.execute(
                "SELECT reason FROM blocklist WHERE entity_type = ? AND value = ?",
                (entity_type, value),
            ).fetchone()
            if row is not None:
                matches.append({"type": entity_type, "value": value, "reason": row["reason"]})
        return {"hit": bool(matches), "matches": matches}
    finally:
        conn.close()


def velocity(card_id: str, window_min: int | None = None,
             as_of: str | datetime | None = None,
             db_path: Path | None = None) -> int:
    """Count transactions on ``card_id`` within the trailing velocity window.

    Counts rows with ``ts`` in ``(as_of - window_min, as_of]``. ``as_of``
    defaults to now; callers evaluating a specific incoming transaction should
    pass that transaction's timestamp so the window is anchored to it rather
    than the wall clock.
    """
    window_min = window_min if window_min is not None else config.VELOCITY_WINDOW_MIN
    if as_of is None:
        as_of_dt = datetime.now()
    elif isinstance(as_of, datetime):
        as_of_dt = as_of
    else:
        as_of_dt = datetime.strptime(as_of, _TS_FMT)
    start = as_of_dt - timedelta(minutes=window_min)

    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM transactions "
            "WHERE card_id = ? AND ts > ? AND ts <= ?",
            (card_id, start.strftime(_TS_FMT), as_of_dt.strftime(_TS_FMT)),
        ).fetchone()
        return int(row["n"])
    finally:
        conn.close()


def latest_txn_ts(card_id: str, db_path: Path | None = None) -> str | None:
    """Timestamp of the most recent transaction on a card (or None)."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT MAX(ts) AS ts FROM transactions WHERE card_id = ?", (card_id,)
        ).fetchone()
        return row["ts"] if row and row["ts"] else None
    finally:
        conn.close()


def device_reputation(device_id: str, db_path: Path | None = None) -> dict:
    """Reputation for a device: proxy flag, distinct cards seen, txn count.

    Returns ``{"known": bool, "proxy_flag": int, "cards_seen": int,
    "txn_count": int, "first_seen": str | None}``. ``cards_seen`` (many cards
    on one device) is a classic device-farm signal.
    """
    conn = _connect(db_path)
    try:
        dev = conn.execute(
            "SELECT proxy_flag, first_seen FROM devices WHERE device_id = ?",
            (device_id,),
        ).fetchone()
        stats = conn.execute(
            "SELECT COUNT(*) AS txns, COUNT(DISTINCT card_id) AS cards "
            "FROM transactions WHERE device_id = ?",
            (device_id,),
        ).fetchone()
        return {
            "known": dev is not None,
            "proxy_flag": int(dev["proxy_flag"]) if dev else 0,
            "first_seen": dev["first_seen"] if dev else None,
            "cards_seen": int(stats["cards"]),
            "txn_count": int(stats["txns"]),
        }
    finally:
        conn.close()


def merchant_reputation(merchant_id: str, db_path: Path | None = None) -> dict:
    """Reputation for a merchant: category, risk level, chargeback rate.

    Returns ``{"known": bool, "category": str | None, "risk_level": str | None,
    "chargeback_rate": float | None, "txn_count": int}``.
    """
    conn = _connect(db_path)
    try:
        m = conn.execute(
            "SELECT category, risk_level, chargeback_rate FROM merchants WHERE merchant_id = ?",
            (merchant_id,),
        ).fetchone()
        stats = conn.execute(
            "SELECT COUNT(*) AS txns FROM transactions WHERE merchant_id = ?",
            (merchant_id,),
        ).fetchone()
        return {
            "known": m is not None,
            "category": m["category"] if m else None,
            "risk_level": m["risk_level"] if m else None,
            "chargeback_rate": float(m["chargeback_rate"]) if m and m["chargeback_rate"] is not None else None,
            "txn_count": int(stats["txns"]),
        }
    finally:
        conn.close()

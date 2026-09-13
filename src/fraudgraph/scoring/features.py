"""Turn a transaction + its evidence into a flat feature dict.

One builder feeds both consumers so training and inference never drift:
  * the deterministic rule engine (``rules.py``) reads the rich feature dict;
  * the ML model reads the ordered numeric vector via ``to_vector``.

Evidence is the canonical dict produced by ``db.gather_evidence`` (see there).
"""

from __future__ import annotations

import math

from fraudgraph import config

# Ordered feature vector consumed by the ML model. Order is the contract
# between train time and predict time — append only, never reorder.
FEATURE_NAMES = [
    "amount_log",
    "is_high_value",
    "blocklist_hit",
    "velocity_count",
    "velocity_over",
    "device_known",
    "device_proxy",
    "device_cards_seen",
    "merchant_high_risk",
    "merchant_medium_risk",
    "merchant_chargeback",
    "geo_foreign",
]


def _country(geo: str | None) -> str | None:
    """'IN-Mumbai' -> 'IN'."""
    if not geo:
        return None
    return geo.split("-", 1)[0]


def build_features(transaction: dict, evidence: dict) -> dict:
    """Build the full feature dict from a transaction and its evidence."""
    amount = float(transaction.get("amount", 0.0) or 0.0)

    bl = evidence.get("blocklist", {}) or {}
    vel = int(evidence.get("velocity", 0) or 0)
    dev = evidence.get("device", {}) or {}
    mer = evidence.get("merchant", {}) or {}

    risk_level = mer.get("risk_level")
    chargeback = mer.get("chargeback_rate")

    return {
        "amount": amount,
        "amount_log": math.log1p(max(amount, 0.0)),
        "is_high_value": int(amount >= config.HIGH_VALUE_THRESHOLD),
        "blocklist_hit": int(bool(bl.get("hit"))),
        "velocity_count": vel,
        "velocity_over": int(vel > config.VELOCITY_MAX_TXNS),
        "device_known": int(bool(dev.get("known"))),
        "device_proxy": int(dev.get("proxy_flag", 0) or 0),
        "device_cards_seen": int(dev.get("cards_seen", 0) or 0),
        "merchant_high_risk": int(risk_level == "high"),
        "merchant_medium_risk": int(risk_level == "medium"),
        "merchant_chargeback": float(chargeback) if chargeback is not None else 0.0,
        "geo_foreign": int(_country(transaction.get("geo")) not in (None, "IN")),
    }


def to_vector(features: dict) -> list[float]:
    """Ordered numeric vector for the ML model, per ``FEATURE_NAMES``."""
    return [float(features[name]) for name in FEATURE_NAMES]

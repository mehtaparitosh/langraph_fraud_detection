"""Deterministic, additive rule engine — the ONLY place a risk score is set.

Read this file top to bottom and you can account for every point on the 0-100
scale. That transparency is the whole point: the decision path is deterministic
and auditable, and the LLM (Phase 4) never touches it.

Each rule is a small function of the feature dict returning
``(points, reason_code, detail)`` when it fires, or ``None`` when it doesn't.
The score is the clamped sum of the points; the band comes from config cutoffs.

The optional ``ml_score`` (0-1, from the local model) is an *input* rule here —
it can nudge the score, but it never decides. Tests score with ``ml_score=None``
so the deterministic core is verified in isolation.
"""

from __future__ import annotations

from typing import Callable, Optional

from fraudgraph import config

# A rule takes (features, ml_score) and returns (points, code, detail) or None.
Rule = Callable[[dict, Optional[float]], Optional[tuple[float, str, str]]]

# Blocklist is a hard override: a confirmed-bad entity forces the top of the
# scale on its own, so nothing downstream can wash it out.
BLOCKLIST_POINTS = 100.0


def _blocklist(f: dict, ml: Optional[float]):
    if f["blocklist_hit"]:
        return (BLOCKLIST_POINTS, "BLOCKLIST_HIT", "card/device/ip on blocklist")
    return None


def _velocity(f: dict, ml: Optional[float]):
    if f["velocity_over"]:
        return (40.0, "VELOCITY_OVER_LIMIT",
                f"{f['velocity_count']} txns > limit {config.VELOCITY_MAX_TXNS} "
                f"in {config.VELOCITY_WINDOW_MIN}m")
    return None


def _device_proxy(f: dict, ml: Optional[float]):
    if f["device_proxy"]:
        return (20.0, "DEVICE_PROXY", "device flagged as proxy/anonymised")
    return None


def _device_many_cards(f: dict, ml: Optional[float]):
    if f["device_cards_seen"] >= 5:
        return (20.0, "DEVICE_MANY_CARDS",
                f"{f['device_cards_seen']} distinct cards seen on device")
    return None


def _merchant_high_risk(f: dict, ml: Optional[float]):
    if f["merchant_high_risk"]:
        return (25.0, "MERCHANT_HIGH_RISK", "high-risk merchant category")
    return None


def _merchant_chargebacks(f: dict, ml: Optional[float]):
    if f["merchant_chargeback"] > 0.05:
        return (10.0, "MERCHANT_ELEVATED_CHARGEBACKS",
                f"merchant chargeback rate {f['merchant_chargeback']:.2%}")
    return None


def _geo_foreign(f: dict, ml: Optional[float]):
    if f["geo_foreign"]:
        return (20.0, "GEO_FOREIGN", "transaction geo outside home country")
    return None


def _high_value(f: dict, ml: Optional[float]):
    if f["is_high_value"]:
        return (10.0, "HIGH_VALUE_AMOUNT",
                f"amount >= high-value threshold {config.HIGH_VALUE_THRESHOLD:g}")
    return None


def _ml_elevated(f: dict, ml: Optional[float]):
    # The model is an input, not the decider: a confident-high model score adds
    # a modest, clearly-labelled contribution and nothing more.
    if ml is not None and ml >= 0.80:
        return (15.0, "ML_ELEVATED_RISK", f"model score {ml:.2f} >= 0.80")
    return None


# Order is cosmetic (points sum regardless); it just controls reason ordering.
RULES: list[Rule] = [
    _blocklist,
    _velocity,
    _device_proxy,
    _device_many_cards,
    _merchant_high_risk,
    _merchant_chargebacks,
    _geo_foreign,
    _high_value,
    _ml_elevated,
]


def score_features(features: dict, ml_score: float | None = None) -> dict:
    """Score a feature dict. Returns rule_score, risk_band, reasons, breakdown."""
    breakdown = []
    total = 0.0
    for rule in RULES:
        result = rule(features, ml_score)
        if result is None:
            continue
        points, code, detail = result
        total += points
        breakdown.append({"code": code, "points": points, "detail": detail})

    rule_score = max(0.0, min(100.0, total))
    return {
        "rule_score": rule_score,
        "risk_band": config.band_for_score(rule_score),
        "reasons": [b["code"] for b in breakdown],
        "breakdown": breakdown,
    }

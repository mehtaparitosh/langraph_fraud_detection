"""Phase 3: the routing function is a pure function of the state."""

from __future__ import annotations

from fraudgraph import config
from fraudgraph.graph import router


def _state(*, band="low", amount=1000.0, blocklist_hit=False, velocity=1):
    return {
        "transaction": {"card_id": "C", "amount": amount},
        "evidence": {
            "blocklist": {"hit": blocklist_hit, "matches": []},
            "velocity": velocity,
        },
        "risk_band": band,
    }


def test_blocklist_hit_declines():
    assert router.route_after_score(_state(blocklist_hit=True)) == router.AUTO_DECLINE


def test_extreme_velocity_declines():
    s = _state(velocity=config.VELOCITY_EXTREME_TXNS)
    assert router.route_after_score(s) == router.AUTO_DECLINE


def test_low_and_below_threshold_approves():
    s = _state(band="low", amount=config.HIGH_VALUE_THRESHOLD - 1)
    assert router.route_after_score(s) == router.AUTO_APPROVE


def test_grey_investigates():
    assert router.route_after_score(_state(band="grey")) == router.INVESTIGATE


def test_high_band_without_blocklist_investigates():
    # A soft "high" score never auto-declines — a human decides.
    assert router.route_after_score(_state(band="high")) == router.INVESTIGATE


def test_low_band_but_high_value_investigates():
    s = _state(band="low", amount=config.HIGH_VALUE_THRESHOLD + 1)
    assert router.route_after_score(s) == router.INVESTIGATE

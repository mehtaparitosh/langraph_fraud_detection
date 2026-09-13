"""Phase 2: the four crafted transactions land in their expected bands, and
every point on the score is accounted for by a reason code.

Scored with ml_score=None so the deterministic core is verified in isolation
(independent of whatever the trained model happens to predict).
"""

from __future__ import annotations

import pytest

from fraudgraph import config
from fraudgraph.data import db, seed
from fraudgraph.data.generate import (
    BLOCKED,
    CLEAN,
    HIGH_VALUE,
    REFERENCE_NOW,
    VELOCITY,
)
from fraudgraph.scoring import features as feats
from fraudgraph.scoring import rules

REF_TS = REFERENCE_NOW.strftime("%Y-%m-%d %H:%M:%S")

_TXN_FIELDS = ("txn_id", "card_id", "device_id", "ip", "merchant_id", "amount", "geo")


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("fraudgraph_scoring")
    return seed.seed(db_path=d / "test.db", seed_dir=d / "seed", regenerate=True)


def _score(crafted: dict, db_path, ml_score=None) -> dict:
    txn = {k: crafted[k] for k in _TXN_FIELDS}
    txn["ts"] = REF_TS
    evidence = db.gather_evidence(txn, db_path=db_path)
    features = feats.build_features(txn, evidence)
    return rules.score_features(features, ml_score=ml_score)


# --------------------------------------------------------------------------
# Bands
# --------------------------------------------------------------------------
def test_clean_is_low(db_path):
    out = _score(CLEAN, db_path)
    assert out["risk_band"] == "low"
    assert out["rule_score"] == 0.0
    assert out["reasons"] == []


def test_velocity_is_grey(db_path):
    out = _score(VELOCITY, db_path)
    assert out["risk_band"] == "grey"
    assert "VELOCITY_OVER_LIMIT" in out["reasons"]
    assert "BLOCKLIST_HIT" not in out["reasons"]  # velocity != auto-decline


def test_blocked_is_high(db_path):
    out = _score(BLOCKED, db_path)
    assert out["risk_band"] == "high"
    assert out["rule_score"] == 100.0
    assert "BLOCKLIST_HIT" in out["reasons"]


def test_high_value_is_low_score_but_flagged(db_path):
    out = _score(HIGH_VALUE, db_path)
    # Clean on every risk axis -> low rule score...
    assert out["risk_band"] == "low"
    # ...but the amount is flagged, which is what routes it to investigate.
    assert "HIGH_VALUE_AMOUNT" in out["reasons"]


# --------------------------------------------------------------------------
# Every point is explained; band cutoffs behave
# --------------------------------------------------------------------------
def test_breakdown_sums_to_score(db_path):
    for crafted in (CLEAN, VELOCITY, BLOCKED, HIGH_VALUE):
        out = _score(crafted, db_path)
        summed = min(100.0, sum(b["points"] for b in out["breakdown"]))
        assert summed == out["rule_score"]


def test_band_cutoffs():
    assert config.band_for_score(config.LOW_MAX) == "low"
    assert config.band_for_score(config.LOW_MAX + 0.1) == "grey"
    assert config.band_for_score(config.GREY_MAX) == "grey"
    assert config.band_for_score(config.GREY_MAX + 0.1) == "high"


# --------------------------------------------------------------------------
# ML score is an input, not the decider
# --------------------------------------------------------------------------
def test_ml_score_adds_reason_but_never_decides(db_path):
    base = _score(CLEAN, db_path)
    boosted = _score(CLEAN, db_path, ml_score=0.95)
    assert "ML_ELEVATED_RISK" in boosted["reasons"]
    assert boosted["rule_score"] > base["rule_score"]
    # A confident model on a clean txn nudges the score but cannot force a band
    # jump to high on its own (15 points from a base of 0 stays out of high).
    assert boosted["risk_band"] in ("low", "grey")


def test_trained_model_predicts_probability(db_path):
    from fraudgraph.scoring import model
    from fraudgraph.scoring.train_model import train

    mp = db_path.parent / "model.pkl"
    train(db_path=db_path, model_path=mp)
    model.load_model.cache_clear()

    txn = {k: BLOCKED[k] for k in _TXN_FIELDS}
    txn["ts"] = REF_TS
    evidence = db.gather_evidence(txn, db_path=db_path)
    score = model.predict_score(txn, evidence, model_path=str(mp))
    assert 0.0 <= score <= 1.0

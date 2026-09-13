"""Phase 1 lookups exercised against the four crafted transactions.

The DB is built once into a temp file from the seed CSVs, so the test is
self-contained and never depends on a pre-existing data/fraud_demo.db.
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
    VELOCITY_BURST,
)

_TS_FMT = "%Y-%m-%d %H:%M:%S"
REF_TS = REFERENCE_NOW.strftime(_TS_FMT)


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("fraudgraph_db")
    path = seed.seed(db_path=d / "test.db", seed_dir=d / "seed", regenerate=True)
    return path


# --------------------------------------------------------------------------
# check_blocklist
# --------------------------------------------------------------------------
def test_clean_not_blocklisted(db_path):
    result = db.check_blocklist(CLEAN["card_id"], CLEAN["device_id"], CLEAN["ip"], db_path=db_path)
    assert result["hit"] is False
    assert result["matches"] == []


def test_blocked_card_hits_blocklist(db_path):
    result = db.check_blocklist(BLOCKED["card_id"], BLOCKED["device_id"], BLOCKED["ip"], db_path=db_path)
    assert result["hit"] is True
    hit_types = {m["type"] for m in result["matches"]}
    # The crafted blocked transaction trips card, device, AND ip blocklist entries.
    assert "card" in hit_types
    assert any(m["value"] == BLOCKED["card_id"] for m in result["matches"])


# --------------------------------------------------------------------------
# velocity
# --------------------------------------------------------------------------
def test_velocity_spike_over_limit(db_path):
    n = db.velocity(VELOCITY["card_id"], window_min=config.VELOCITY_WINDOW_MIN,
                    as_of=REF_TS, db_path=db_path)
    assert n == VELOCITY_BURST
    assert n > config.VELOCITY_MAX_TXNS


def test_clean_card_low_velocity(db_path):
    n = db.velocity(CLEAN["card_id"], window_min=config.VELOCITY_WINDOW_MIN,
                    as_of=REF_TS, db_path=db_path)
    assert n <= config.VELOCITY_MAX_TXNS


# --------------------------------------------------------------------------
# device_reputation
# --------------------------------------------------------------------------
def test_device_reputation_clean(db_path):
    rep = db.device_reputation(CLEAN["device_id"], db_path=db_path)
    assert rep["known"] is True
    assert rep["proxy_flag"] == 0


def test_device_reputation_velocity_device(db_path):
    rep = db.device_reputation(VELOCITY["device_id"], db_path=db_path)
    assert rep["known"] is True
    # Velocity row is deliberately clean on the device axis: velocity is the
    # only strong signal, so this case lands in grey (not high).
    assert rep["proxy_flag"] == 0
    # The whole burst ran through this one device.
    assert rep["txn_count"] >= VELOCITY_BURST


# --------------------------------------------------------------------------
# merchant_reputation
# --------------------------------------------------------------------------
def test_merchant_reputation_high_risk(db_path):
    rep = db.merchant_reputation(BLOCKED["merchant_id"], db_path=db_path)
    assert rep["known"] is True
    assert rep["risk_level"] == "high"
    assert rep["chargeback_rate"] > 0.05


def test_merchant_reputation_low_risk(db_path):
    rep = db.merchant_reputation(CLEAN["merchant_id"], db_path=db_path)
    assert rep["known"] is True
    assert rep["risk_level"] == "low"


# --------------------------------------------------------------------------
# high-value crafted row is genuinely grey-zone material
# --------------------------------------------------------------------------
def test_high_value_row_above_threshold_but_clean(db_path):
    # Amount trips the high-value branch...
    assert HIGH_VALUE["amount"] > config.HIGH_VALUE_THRESHOLD
    # ...yet nothing else is wrong: not blocklisted, clean device, low velocity.
    bl = db.check_blocklist(HIGH_VALUE["card_id"], HIGH_VALUE["device_id"], HIGH_VALUE["ip"], db_path=db_path)
    assert bl["hit"] is False
    assert db.device_reputation(HIGH_VALUE["device_id"], db_path=db_path)["proxy_flag"] == 0
    assert db.velocity(HIGH_VALUE["card_id"], as_of=REF_TS, db_path=db_path) <= config.VELOCITY_MAX_TXNS

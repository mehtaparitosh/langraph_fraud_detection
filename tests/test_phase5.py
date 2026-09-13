"""Phase 5: OTP store idempotency + the human-gate / step-up flow.

Email is mocked (no real send). The graph runs with a MemorySaver and stubbed
LLM nodes so the human-in-the-loop mechanics are tested in isolation.
"""

from __future__ import annotations

import uuid

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from fraudgraph import config
from fraudgraph.data import seed
from fraudgraph.data.generate import HIGH_VALUE, REFERENCE_NOW
from fraudgraph.scoring import model

_TS = REFERENCE_NOW.strftime("%Y-%m-%d %H:%M:%S")
_TXN_FIELDS = ("txn_id", "card_id", "device_id", "ip", "merchant_id", "amount", "geo")


# --------------------------------------------------------------------------
# OTP store — unit level
# --------------------------------------------------------------------------
def test_otp_send_is_idempotent_and_verifies(tmp_path, monkeypatch):
    from fraudgraph import notifications

    sends = []
    monkeypatch.setattr(notifications.messenger, "send_email",
                        lambda *a, **k: sends.append(k.get("to")))
    db = tmp_path / "otp.db"

    code1 = notifications.send_otp("thread-A", "u@example.com", db_path=db)
    assert code1 and len(code1) == config.OTP_LENGTH
    assert len(sends) == 1

    # Re-run (as on resume): no second email, same code stays active.
    code2 = notifications.send_otp("thread-A", "u@example.com", db_path=db)
    assert code2 is None
    assert len(sends) == 1
    assert notifications.get_otp("thread-A", db_path=db) == code1

    assert notifications.verify_otp("thread-A", code1, db_path=db) is True
    assert notifications.verify_otp("thread-A", "000000", db_path=db) is False


# --------------------------------------------------------------------------
# Full human-gate / step-up flow through the graph
# --------------------------------------------------------------------------
@pytest.fixture
def flow(tmp_path, monkeypatch):
    db_path = seed.seed(db_path=tmp_path / "test.db", seed_dir=tmp_path / "seed", regenerate=True)
    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "no_model.pkl")
    monkeypatch.setattr(config, "OTP_DB", tmp_path / "otp.db")
    monkeypatch.setattr(config, "DEMO_USER_EMAIL", "demo@example.com")
    model.load_model.cache_clear()

    # Stub the LLM nodes (offline); recommend step_up.
    from fraudgraph.graph import nodes
    monkeypatch.setattr(nodes, "investigate", lambda s: {
        "analyst_summary": "stub",
        "recommendation": {"recommend": "step_up", "rationale": "stub", "confidence": 0.5},
    })
    monkeypatch.setattr(nodes, "draft_sar", lambda s: {"sar_draft": "stub"})

    # Count OTP emails.
    from fraudgraph import notifications
    sends = []
    monkeypatch.setattr(notifications.messenger, "send_email",
                        lambda *a, **k: sends.append(k.get("to")))

    from fraudgraph.graph.build import build_graph
    graph = build_graph(checkpointer=MemorySaver())
    return graph, sends


def _cfg():
    tid = str(uuid.uuid4())
    return tid, {"configurable": {"thread_id": tid}}


def _txn():
    t = {k: HIGH_VALUE[k] for k in _TXN_FIELDS}
    t["ts"] = _TS
    return t


def test_consent_approve_finalizes(flow):
    graph, sends = flow
    _, cfg = _cfg()
    graph.invoke({"transaction": _txn()}, cfg)          # pause at human_gate
    graph.invoke(Command(resume="approve"), cfg)        # analyst approves
    vals = graph.get_state(cfg).values
    assert vals["decision"] == "approve"
    assert sends == []                                   # no OTP on a direct approve


def test_stepup_success_sends_one_otp_and_approves(flow):
    graph, sends = flow
    from fraudgraph import notifications
    tid, cfg = _cfg()

    graph.invoke({"transaction": _txn()}, cfg)           # pause at human_gate
    graph.invoke(Command(resume="step_up"), cfg)         # -> step_up sends OTP, pause
    assert len(sends) == 1                               # OTP emailed once

    code = notifications.get_otp(tid, db_path=config.OTP_DB)
    graph.invoke(Command(resume=code), cfg)              # correct code -> approve
    vals = graph.get_state(cfg).values

    assert vals["otp_ok"] is True
    assert vals["decision"] == "approve"
    assert len(sends) == 1                               # NOT resent on resume re-run


def test_stepup_wrong_code_declines_with_sar(flow):
    graph, sends = flow
    _, cfg = _cfg()
    graph.invoke({"transaction": _txn()}, cfg)
    graph.invoke(Command(resume="step_up"), cfg)
    graph.invoke(Command(resume="000000"), cfg)          # wrong OTP
    vals = graph.get_state(cfg).values
    assert vals["otp_ok"] is False
    assert vals["decision"] == "decline"
    assert vals.get("sar_draft") == "stub"               # SAR path taken


def test_consent_decline_writes_sar(flow):
    graph, sends = flow
    _, cfg = _cfg()
    graph.invoke({"transaction": _txn()}, cfg)
    graph.invoke(Command(resume="decline"), cfg)
    vals = graph.get_state(cfg).values
    assert vals["decision"] == "decline"
    assert vals.get("sar_draft") == "stub"


def test_consent_escalate_sends_email(flow):
    graph, sends = flow
    _, cfg = _cfg()
    graph.invoke({"transaction": _txn()}, cfg)
    graph.invoke(Command(resume="escalate"), cfg)
    vals = graph.get_state(cfg).values
    assert vals["decision"] == "escalate"
    assert vals.get("sar_draft") == "stub"       # SAR drafted on the escalate path
    assert len(sends) == 1                        # escalation email sent once

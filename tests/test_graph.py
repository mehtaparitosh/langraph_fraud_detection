"""Phase 3: the compiled graph runs end-to-end and each crafted transaction
takes a distinct, correct, fully deterministic path.

Self-contained: builds a temp SQLite DB and points config at it, and runs with
a synchronous MemorySaver so no event loop / AsyncSqliteSaver is needed. No
model is trained here (ml_score degrades to 0), so bands are pure-rule.
"""

from __future__ import annotations

import uuid

import pytest
from langgraph.checkpoint.memory import MemorySaver

from fraudgraph import config
from fraudgraph.data import seed
from fraudgraph.data.generate import BLOCKED, CLEAN, HIGH_VALUE, REFERENCE_NOW, VELOCITY
from fraudgraph.scoring import model

_TS = REFERENCE_NOW.strftime("%Y-%m-%d %H:%M:%S")
_TXN_FIELDS = ("txn_id", "card_id", "device_id", "ip", "merchant_id", "amount", "geo")


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    d = tmp_path_factory.mktemp("fraudgraph_graph")
    db_path = seed.seed(db_path=d / "test.db", seed_dir=d / "seed", regenerate=True)

    # Point the modules' config at the temp DB and a non-existent model
    # (predict_score -> 0.0), so the run is deterministic and dependency-free.
    orig_db, orig_model = config.DB_PATH, config.MODEL_PATH
    config.DB_PATH = db_path
    config.MODEL_PATH = d / "no_model.pkl"
    model.load_model.cache_clear()

    # Stub the LLM nodes so path/routing tests stay offline and fast. The real
    # LLM behaviour is exercised by the CLI; the guard is tested separately.
    from fraudgraph.graph import nodes
    orig_inv, orig_sar = nodes.investigate, nodes.draft_sar
    nodes.investigate = lambda s: {"analyst_summary": "stub",
                                   "recommendation": {"recommend": "step_up",
                                                      "rationale": "stub", "confidence": 0.5}}
    nodes.draft_sar = lambda s: {"sar_draft": "stub"}

    from fraudgraph.graph.build import build_graph
    compiled = build_graph(checkpointer=MemorySaver())
    yield compiled

    nodes.investigate, nodes.draft_sar = orig_inv, orig_sar
    config.DB_PATH, config.MODEL_PATH = orig_db, orig_model
    model.load_model.cache_clear()


def _run(graph, crafted: dict) -> dict:
    txn = {k: crafted[k] for k in _TXN_FIELDS}
    txn["ts"] = _TS
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    return graph.invoke({"transaction": txn}, cfg)


def test_clean_auto_approves(graph):
    out = _run(graph, CLEAN)
    assert out["decision"] == "approve"
    assert out["risk_band"] == "low"
    # auto-approve path never touches the LLM stubs.
    assert "analyst_summary" not in out
    assert "sar_draft" not in out


def test_blocklist_auto_declines(graph):
    out = _run(graph, BLOCKED)
    assert out["decision"] == "decline"
    assert out["risk_band"] == "high"
    assert "BLOCKLIST_HIT" in out["reasons"]
    # decline routes through draft_sar, not investigate.
    assert "sar_draft" in out
    assert "analyst_summary" not in out


def _run_until_pause(graph, crafted: dict):
    txn = {k: crafted[k] for k in _TXN_FIELDS}
    txn["ts"] = _TS
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    graph.invoke({"transaction": txn}, cfg)
    return graph.get_state(cfg)


def test_velocity_investigates_then_pauses_for_consent(graph):
    st = _run_until_pause(graph, VELOCITY)
    assert st.values["risk_band"] == "grey"
    assert "VELOCITY_OVER_LIMIT" in st.values["reasons"]
    # investigate ran, then the graph paused at the human gate — not decided.
    assert st.values.get("analyst_summary") == "stub"
    assert "decision" not in st.values
    assert st.next == ("human_gate",)


def test_high_value_investigates_by_amount(graph):
    st = _run_until_pause(graph, HIGH_VALUE)
    assert st.values["risk_band"] == "low"
    assert "HIGH_VALUE_AMOUNT" in st.values["reasons"]
    assert st.next == ("human_gate",)


def test_llm_path_writes_recommendation_not_decision(graph):
    st = _run_until_pause(graph, VELOCITY)
    assert st.values.get("recommendation", {}).get("recommend") == "step_up"
    # The node proposed; the graph paused without any decision being set.
    assert "decision" not in st.values

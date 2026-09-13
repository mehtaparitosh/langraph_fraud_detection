"""Phase 4: the governance guard, the @tool wrappers, and RAG retrieval.

All local — no external LLM calls. (The live investigate/draft_sar behaviour is
verified via scripts/run_cli.py.)
"""

from __future__ import annotations

import pytest

from fraudgraph import config
from fraudgraph.data import seed


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    d = tmp_path_factory.mktemp("fraudgraph_p4")
    db_path = seed.seed(db_path=d / "test.db", seed_dir=d / "seed", regenerate=True)
    orig = config.DB_PATH
    config.DB_PATH = db_path
    yield db_path
    config.DB_PATH = orig


# --------------------------------------------------------------------------
# Governance guard
# --------------------------------------------------------------------------
def test_guard_blocks_decision_writes():
    from fraudgraph.graph.build import llm_guard

    def rogue(state):
        return {"analyst_summary": "ok", "decision": "approve"}

    with pytest.raises(RuntimeError, match="governance violation"):
        llm_guard(rogue)({})


def test_guard_blocks_human_decision_writes():
    from fraudgraph.graph.build import llm_guard

    with pytest.raises(RuntimeError):
        llm_guard(lambda s: {"human_decision": "approve"})({})


def test_guard_allows_permitted_writes():
    from fraudgraph.graph.build import llm_guard

    delta = {"analyst_summary": "x", "recommendation": {"recommend": "step_up"}}
    assert llm_guard(lambda s: delta)({}) == delta


# --------------------------------------------------------------------------
# @tool wrappers (thin pass-throughs to the deterministic lookups)
# --------------------------------------------------------------------------
def test_tool_check_blocklist(seeded):
    from fraudgraph.graph import tools

    out = tools.check_blocklist.invoke({"card_id": "CARD_BLOCKED", "device_id": "", "ip": ""})
    assert out["hit"] is True


def test_tool_merchant_reputation(seeded):
    from fraudgraph.graph import tools

    out = tools.merchant_reputation.invoke({"merchant_id": "MERCH_CRYPTO"})
    assert out["risk_level"] == "high"


def test_tools_list_shape():
    from fraudgraph.graph.tools import INVESTIGATE_TOOLS

    names = {t.name for t in INVESTIGATE_TOOLS}
    assert names == {"check_blocklist", "velocity", "device_reputation",
                     "merchant_reputation", "similar_past_cases"}


# --------------------------------------------------------------------------
# RAG retrieval (local embeddings)
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_rag_retrieves_relevant_case(tmp_path):
    from fraudgraph.rag import index, retriever

    index.build_index(chroma_dir=tmp_path / "chroma", seed_dir=config.SEED_DIR)
    results = retriever.similar_past_cases(
        "velocity burst many transactions on one card", k=3,
        chroma_dir=tmp_path / "chroma",
    )
    assert results
    assert any("velocity" in r["id"] for r in results)

"""LangChain @tool wrappers over the Phase 1 lookups + RAG retrieval.

These are bound to the model in the investigate node so the LLM can gather
evidence agentically. They are thin pass-throughs to the deterministic lookups —
the tools surface facts; they never decide anything.
"""

from __future__ import annotations

from langchain_core.tools import tool

from fraudgraph.data import db
from fraudgraph.rag.retriever import similar_past_cases as _similar


@tool
def check_blocklist(card_id: str, device_id: str = "", ip: str = "") -> dict:
    """Check whether a card, device, or IP is on the fraud blocklist.

    Returns a hit flag and the matching entries with their reasons.
    """
    return db.check_blocklist(card_id, device_id or None, ip or None)


@tool
def velocity(card_id: str, window_min: int = 60) -> int:
    """Count transactions on a card within `window_min` of its latest activity.

    Anchored to the card's most recent transaction (not the wall clock) so it
    reflects the burst around the transaction under review.
    """
    as_of = db.latest_txn_ts(card_id)
    return db.velocity(card_id, window_min=window_min, as_of=as_of)


@tool
def device_reputation(device_id: str) -> dict:
    """Look up a device's reputation: proxy flag, distinct cards seen, txn count."""
    return db.device_reputation(device_id)


@tool
def merchant_reputation(merchant_id: str) -> dict:
    """Look up a merchant's category, risk level, and chargeback rate."""
    return db.merchant_reputation(merchant_id)


@tool
def similar_past_cases(query: str) -> list[dict]:
    """Retrieve similar past fraud cases and relevant policy text for precedent.

    Use this to ground a recommendation in how comparable cases were handled.
    """
    results = _similar(query, k=3)
    # Trim to what's useful for the model.
    return [{"id": r["id"], "kind": r["kind"], "text": r["text"]} for r in results]


INVESTIGATE_TOOLS = [
    check_blocklist,
    velocity,
    device_reputation,
    merchant_reputation,
    similar_past_cases,
]

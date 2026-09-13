"""The frozen state contract. Everything in the graph depends on this shape.

Governance invariant (enforced by a guard in ``build.py`` from Phase 4 on):
``decision`` is only ever written by the deterministic decision nodes
(``auto_approve`` / ``auto_decline``), the human gate, or the step-up result —
**never** by an LLM node. LLM nodes may only write ``analyst_summary`` and
``sar_draft`` (and propose a recommendation), never ``decision``.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class FraudState(TypedDict, total=False):
    transaction: dict          # amount, card_id, device_id, ip, merchant_id, ts, geo, user_email
    evidence: dict             # filled by enrich: {blocklist, velocity, device, merchant}
    rule_score: float          # 0-100, set by the deterministic engine ONLY
    ml_score: float            # 0-1 from the local model (an input to scoring)
    risk_band: str             # "low" | "grey" | "high"
    decision: str              # approve | decline | hold | step_up | escalate (engine/human ONLY)
    reasons: list[str]         # machine reason codes
    analyst_summary: str       # LLM narration (never a decision) — Phase 4
    recommendation: dict       # LLM's proposed action {recommend, rationale, confidence} — NOT a decision
    sar_draft: str             # LLM case note grounded in evidence — Phase 4
    otp_ok: bool               # result of step-up — Phase 5
    human_decision: str        # set after interrupt() — Phase 5
    messages: Annotated[list, add_messages]


# The set of state keys an LLM node is permitted to write. Used by the guard.
LLM_WRITABLE_KEYS = {"analyst_summary", "recommendation", "sar_draft", "messages"}

# Keys an LLM node must NEVER write — the core governance invariant.
LLM_FORBIDDEN_KEYS = {"decision", "human_decision"}

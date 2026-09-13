"""Deterministic routing after the score node.

Pure function of the state — no side effects, no I/O — so it is trivial to
unit-test. This is the branch the frontend renders as ``route{...}``.

Routing policy (governed, conservative):
  * blocklist hit                 -> auto_decline   (confirmed-bad, certain)
  * extreme velocity              -> auto_decline   (deterministic enough)
  * low band AND below threshold  -> auto_approve
  * everything else               -> investigate    (grey, high, or high-value
                                                      by amount -> needs a human)

Note only *certain* signals auto-decline. A merely "high" rule score without a
blocklist hit still goes to investigate → the human gate decides. The engine
never quietly declines on a soft signal.
"""

from __future__ import annotations

from fraudgraph import config
from fraudgraph.graph.state import FraudState

AUTO_APPROVE = "auto_approve"
AUTO_DECLINE = "auto_decline"
INVESTIGATE = "investigate"


def route_after_score(state: FraudState) -> str:
    """Return the name of the next node."""
    txn = state["transaction"]
    evidence = state.get("evidence", {}) or {}
    amount = float(txn.get("amount", 0.0) or 0.0)

    blocklist = evidence.get("blocklist", {}) or {}
    velocity = int(evidence.get("velocity", 0) or 0)

    if blocklist.get("hit"):
        return AUTO_DECLINE
    if velocity >= config.VELOCITY_EXTREME_TXNS:
        return AUTO_DECLINE
    if state.get("risk_band") == "low" and amount < config.HIGH_VALUE_THRESHOLD:
        return AUTO_APPROVE
    return INVESTIGATE


def route_after_human(state: FraudState) -> str:
    """Route on the analyst's choice from the human gate."""
    choice = (state.get("human_decision") or "").lower()
    if choice == "approve":
        return "finalize"
    if choice == "step_up":
        return "step_up"
    # decline or escalate -> write a SAR narrative first.
    return "draft_sar"


def route_after_step_up(state: FraudState) -> str:
    """Passed OTP -> finalize; failed -> SAR then finalize."""
    return "finalize" if state.get("otp_ok") else "draft_sar"

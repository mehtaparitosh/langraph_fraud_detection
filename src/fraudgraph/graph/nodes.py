"""Graph node functions.

Phase 3 implements the deterministic spine end to end:
    ingest -> enrich -> score -> (auto_approve | auto_decline | investigate) -> finalize

``investigate``, ``draft_sar``, ``human_gate`` and ``step_up`` are stubs here;
Phases 4-5 fill them in. The stubs deliberately do NOT write ``decision`` — that
invariant already holds so the Phase 4 guard has nothing to catch.

Each node takes the state and returns only the keys it changes (LangGraph merges
the delta), which is also what the CLI/WebSocket streams as ``state_delta``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from fraudgraph import config
from fraudgraph.data import db
from fraudgraph.scoring import features as feats
from fraudgraph.scoring import model, rules
from fraudgraph.graph.state import FraudState

# Cap the agentic tool loop so a misbehaving model can't spin forever.
MAX_TOOL_ITERS = 5


def _emit(message: str) -> None:
    """Emit a fine-grained progress step for the UI (custom stream channel).

    A no-op when the graph isn't being streamed with stream_mode="custom"
    (e.g. under graph.invoke in tests), so it never affects behaviour.
    """
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
        if writer:
            writer({"step": message})
    except Exception:
        pass

_TS_FMT = "%Y-%m-%d %H:%M:%S"
_REQUIRED = ("card_id", "amount")


# --------------------------------------------------------------------------
# Deterministic spine
# --------------------------------------------------------------------------
def ingest(state: FraudState) -> dict:
    """Normalise + validate the incoming transaction."""
    txn = dict(state["transaction"])
    missing = [k for k in _REQUIRED if not txn.get(k)]
    if missing:
        raise ValueError(f"transaction missing required field(s): {missing}")
    txn["amount"] = float(txn["amount"])
    txn.setdefault("ts", datetime.now().strftime(_TS_FMT))
    for k in ("device_id", "ip", "merchant_id", "geo", "user_email"):
        txn.setdefault(k, None)
    return {"transaction": txn}


def enrich(state: FraudState) -> dict:
    """Run all local lookups into the canonical evidence dict (no scoring)."""
    txn = state["transaction"]
    _emit("checking card / device / IP against blocklist")
    blocklist = db.check_blocklist(txn["card_id"], txn.get("device_id"), txn.get("ip"))
    _emit(f"counting recent transactions (velocity, {config.VELOCITY_WINDOW_MIN}m window)")
    velocity = db.velocity(txn["card_id"], as_of=txn.get("ts"))
    _emit("looking up device reputation")
    device = db.device_reputation(txn.get("device_id"))
    _emit("looking up merchant reputation")
    merchant = db.merchant_reputation(txn.get("merchant_id"))
    return {"evidence": {"blocklist": blocklist, "velocity": velocity,
                         "device": device, "merchant": merchant}}


def score(state: FraudState) -> dict:
    """Deterministic scoring. Sets score/ml_score/band/reasons — NOT decision."""
    txn, evidence = state["transaction"], state["evidence"]
    _emit("building feature vector from evidence")
    features = feats.build_features(txn, evidence)
    _emit("scoring with the local ML model")
    ml_score = model.predict_score(txn, evidence)
    _emit("applying the deterministic rule engine")
    result = rules.score_features(features, ml_score=ml_score)
    return {
        "rule_score": result["rule_score"],
        "ml_score": ml_score,
        "risk_band": result["risk_band"],
        "reasons": result["reasons"],
    }


def auto_approve(state: FraudState) -> dict:
    """Deterministic terminal decision for clear, low-risk cases."""
    return {"decision": "approve"}


def auto_decline(state: FraudState) -> dict:
    """Deterministic terminal decision for confirmed-bad cases."""
    return {"decision": "decline"}


def _notify_escalation(state: FraudState) -> None:
    """Email the team that a case was escalated. Best-effort — never crashes."""
    from fraudgraph import notifications

    txn = state.get("transaction", {})
    to = txn.get("user_email") or config.DEMO_USER_EMAIL or config.EMAIL_ADDRESS
    try:
        notifications.send_escalation(
            transaction=txn,
            risk_band=state.get("risk_band"),
            rule_score=state.get("rule_score"),
            reasons=state.get("reasons"),
            analyst_summary=state.get("analyst_summary") or "",
            sar_draft=state.get("sar_draft") or "",
            to=to,
        )
    except Exception as exc:  # email is best-effort; don't fail the decision
        print(f"[finalize] escalation email failed: {exc}")


def finalize(state: FraudState) -> dict:
    """Persist + notify. Escalated cases email the team; pending settle to 'hold'."""
    decision = state.get("decision") or "hold"
    if decision == "escalate":
        _emit("sending escalation email")
        _notify_escalation(state)
    return {"decision": decision}


# --------------------------------------------------------------------------
# LLM nodes (Phase 4). These propose and explain — they NEVER set `decision`.
# The guard in build.py enforces that at runtime.
# --------------------------------------------------------------------------
def _run_tool_loop(model_with_tools, messages, tool_map):
    """Let the model call tools until it stops, returning the final messages."""
    from langchain_core.messages import ToolMessage

    llm_cfg = {"run_name": "investigate · reason + tool calls", "tags": ["fraudgraph", "investigate"]}
    for _ in range(MAX_TOOL_ITERS):
        ai = model_with_tools.invoke(messages, config=llm_cfg)
        messages.append(ai)
        if not getattr(ai, "tool_calls", None):
            break
        for call in ai.tool_calls:
            args = ", ".join(f"{k}={v}" for k, v in (call.get("args") or {}).items())
            _emit(f"tool call: {call['name']}({args})")
            fn = tool_map.get(call["name"])
            try:
                result = fn.invoke(call["args"]) if fn else f"unknown tool {call['name']}"
            except Exception as exc:  # surface tool errors to the model, don't crash
                result = f"tool error: {exc}"
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return messages


def investigate(state: FraudState) -> dict:
    """LLM gathers evidence via tools + RAG, then proposes a recommendation.

    Writes ``analyst_summary`` (narration) and ``recommendation`` (a proposal) —
    never ``decision``.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import BaseModel, Field

    from fraudgraph.graph.tools import INVESTIGATE_TOOLS
    from fraudgraph.llm import prompts
    from fraudgraph.llm.factory import get_model

    class Recommendation(BaseModel):
        recommend: Literal["approve", "step_up", "decline", "escalate"] = Field(
            description="Proposed action for the human analyst — NOT a decision."
        )
        rationale: str = Field(description="One or two sentences grounded in the evidence.")
        confidence: float = Field(description="0-1 confidence in the recommendation.")

    context = prompts.format_case(
        state["transaction"], state["evidence"],
        rule_score=state.get("rule_score"), risk_band=state.get("risk_band"),
        reasons=state.get("reasons"),
    )

    model = get_model()
    tool_map = {t.name: t for t in INVESTIGATE_TOOLS}
    messages = [
        SystemMessage(content=prompts.INVESTIGATE_SYSTEM),
        HumanMessage(content=f"Investigate this flagged transaction:\n\n{context}"),
    ]
    _emit("querying the LLM with evidence-gathering tools")
    messages = _run_tool_loop(model.bind_tools(INVESTIGATE_TOOLS), messages, tool_map)
    _emit("writing the analyst summary")
    analyst_summary = messages[-1].content if isinstance(messages[-1].content, str) \
        else str(messages[-1].content)

    # Second, structured pass: extract the recommendation from the investigation.
    _emit("extracting a structured recommendation")
    rec_model = get_model().with_structured_output(Recommendation)
    rec = rec_model.invoke(
        [
            SystemMessage(content=prompts.RECOMMENDATION_SYSTEM),
            HumanMessage(content=f"Investigation notes:\n{analyst_summary}\n\nCase:\n{context}"),
        ],
        config={"run_name": "investigate · structured recommendation",
                "tags": ["fraudgraph", "investigate", "recommendation"]},
    )

    return {
        "analyst_summary": analyst_summary,
        "recommendation": rec.model_dump() if hasattr(rec, "model_dump") else dict(rec),
    }


def draft_sar(state: FraudState) -> dict:
    """LLM writes a SAR narrative grounded strictly in the evidence."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from fraudgraph.llm import prompts
    from fraudgraph.llm.factory import get_model

    _emit("grounding the SAR narrative in the recorded evidence")
    context = prompts.format_case(
        state["transaction"], state["evidence"],
        rule_score=state.get("rule_score"), risk_band=state.get("risk_band"),
        reasons=state.get("reasons"), decision=state.get("decision"),
    )
    resp = get_model().invoke(
        [
            SystemMessage(content=prompts.SAR_SYSTEM),
            HumanMessage(content=f"Write the SAR narrative from these facts only:\n\n{context}"),
        ],
        config={"run_name": "draft_sar · SAR narrative", "tags": ["fraudgraph", "draft_sar"]},
    )
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    return {"sar_draft": content}


# --------------------------------------------------------------------------
# Human-in-the-loop + step-up (Phase 5). These are the deterministic/human
# decision points, so they ARE allowed to write `decision`.
# --------------------------------------------------------------------------
# Terminal analyst choices that settle the decision immediately.
_TERMINAL_CHOICES = {"approve", "decline", "escalate"}


def human_gate(state: FraudState) -> dict:
    """Pause for analyst consent, then record the human's choice.

    Presents the case + the LLM's recommendation and interrupts. The resume
    value is the analyst's choice: approve | decline | step_up | escalate.
    """
    from langgraph.types import interrupt

    payload = {
        "kind": "consent",
        "transaction": state["transaction"],
        "evidence": state.get("evidence"),
        "rule_score": state.get("rule_score"),
        "risk_band": state.get("risk_band"),
        "reasons": state.get("reasons"),
        "recommendation": state.get("recommendation"),
        "analyst_summary": state.get("analyst_summary"),
    }
    choice = str(interrupt(payload)).strip().lower()

    delta = {"human_decision": choice}
    if choice in _TERMINAL_CHOICES:
        delta["decision"] = choice
    return delta


def step_up(state: FraudState, config) -> dict:
    """Send an OTP (once), pause for the code, verify, and set the outcome.

    The OTP send is guarded by the notifications store keyed on thread_id, so it
    is NOT re-sent when this node re-runs on resume. Note: interrupt() is not
    wrapped in try/except — that would swallow the resume mechanism.

    ``config`` is injected by LangGraph (it matches the parameter name), so it
    must stay named ``config``; the config *module* is aliased to ``cfg`` here.
    """
    from langgraph.types import interrupt

    from fraudgraph import config as cfg
    from fraudgraph import notifications

    thread_id = config["configurable"]["thread_id"]
    email = state["transaction"].get("user_email") or cfg.DEMO_USER_EMAIL or cfg.EMAIL_ADDRESS

    # Idempotent side effect: only actually emails on the first pass.
    notifications.send_otp(thread_id, email)

    entered = interrupt({"kind": "otp", "email": email})
    ok = notifications.verify_otp(thread_id, str(entered))

    # step-up result sets the decision: passed -> approve, failed -> decline.
    return {"otp_ok": ok, "decision": "approve" if ok else "decline"}

"""Assemble and compile the graph.

``build_graph(checkpointer)`` wires the StateGraph and returns the compiled
graph. ``open_graph()`` is an async context manager that owns an
``AsyncSqliteSaver`` for durable, resumable runs (used by the CLI and, later,
the web app). Tests can compile with a ``MemorySaver`` (or none) and run
synchronously.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import wraps
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from fraudgraph import config
from fraudgraph.graph import nodes, router
from fraudgraph.graph.state import LLM_FORBIDDEN_KEYS, FraudState


def llm_guard(fn):
    """Wrap an LLM node so it can never write a decision key.

    Enforces the system-wide invariant: the LLM proposes and explains, but the
    decision is only ever set by the deterministic engine or a human.
    """
    @wraps(fn)
    def wrapper(state):
        delta = fn(state) or {}
        illegal = LLM_FORBIDDEN_KEYS & set(delta)
        if illegal:
            raise RuntimeError(
                f"governance violation: LLM node '{fn.__name__}' tried to write {illegal}"
            )
        return delta

    return wrapper


def build_graph(checkpointer=None):
    """Wire and compile the fraud-decision graph."""
    g = StateGraph(FraudState)

    # Deterministic spine.
    g.add_node("ingest", nodes.ingest)
    g.add_node("enrich", nodes.enrich)
    g.add_node("score", nodes.score)
    g.add_node("auto_approve", nodes.auto_approve)
    g.add_node("auto_decline", nodes.auto_decline)
    g.add_node("finalize", nodes.finalize)

    # LLM nodes are wrapped by the governance guard.
    g.add_node("investigate", llm_guard(nodes.investigate))
    g.add_node("draft_sar", llm_guard(nodes.draft_sar))

    # Human-in-the-loop + step-up (interrupt nodes).
    g.add_node("human_gate", nodes.human_gate)
    g.add_node("step_up", nodes.step_up)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "enrich")
    g.add_edge("enrich", "score")

    g.add_conditional_edges(
        "score",
        router.route_after_score,
        {
            router.AUTO_APPROVE: "auto_approve",
            router.AUTO_DECLINE: "auto_decline",
            router.INVESTIGATE: "investigate",
        },
    )

    g.add_edge("auto_approve", "finalize")
    g.add_edge("auto_decline", "draft_sar")

    # Investigate -> human consent -> (approve | step_up | decline/escalate).
    g.add_edge("investigate", "human_gate")
    g.add_conditional_edges(
        "human_gate", router.route_after_human,
        {"finalize": "finalize", "step_up": "step_up", "draft_sar": "draft_sar"},
    )
    g.add_conditional_edges(
        "step_up", router.route_after_step_up,
        {"finalize": "finalize", "draft_sar": "draft_sar"},
    )

    g.add_edge("draft_sar", "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)


@asynccontextmanager
async def open_graph(checkpoints_db: Path | None = None):
    """Yield a compiled graph backed by an AsyncSqliteSaver checkpointer."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    path = checkpoints_db or config.CHECKPOINTS_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        yield build_graph(checkpointer=saver)

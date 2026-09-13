"""FastAPI + WebSocket front door to the fraud graph.

Endpoints:
  POST /run                -> {"transaction": {...}} or {"default": "velocity"};
                              returns a new thread_id (does not run yet).
  WS   /stream/{thread_id} -> runs graph.astream(stream_mode="updates"), pushes
                              one message per node; on an interrupt pushes an
                              {type:"interrupt"} message and waits; on completion
                              pushes {type:"result"} with the final state.
  POST /resume/{thread_id} -> {"value": "..."} feeds the waiting interrupt so the
                              open WebSocket continues streaming.

The graph is compiled once (lifespan) over a shared AsyncSqliteSaver, so a run
started on the WS and resumed via /resume share the same durable thread.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from fraudgraph import config
from fraudgraph.data.generate import BLOCKED, CLEAN, HIGH_VALUE, REFERENCE_NOW, VELOCITY
from fraudgraph.graph.build import build_graph

STATIC_DIR = Path(__file__).parent / "static"
_TS = REFERENCE_NOW.strftime("%Y-%m-%d %H:%M:%S")
_TXN_FIELDS = ("txn_id", "card_id", "device_id", "ip", "merchant_id", "amount", "geo")
PACING_SEC = 0.35    # small delay between node events so the run is watchable
KEEPALIVE_SEC = 8.0  # ping cadence to keep the socket alive during long LLM nodes


def _demo(crafted: dict) -> dict:
    txn = {k: crafted[k] for k in _TXN_FIELDS}
    txn["ts"] = _TS
    return txn


DEFAULTS = {"clean": _demo(CLEAN), "blocklist": _demo(BLOCKED),
            "velocity": _demo(VELOCITY), "high_value": _demo(HIGH_VALUE)}

DEFAULT_META = {
    "clean": "A normal, low-risk purchase — small amount, home city, trusted device, "
             "low-risk merchant. Scores 0, so the engine auto-approves it.",
    "blocklist": "The card, device and IP are all on the confirmed-fraud blocklist, "
                 "at a crypto merchant from Moscow. A deterministic auto-decline "
                 "(the LLM is never even consulted).",
    "velocity": "The same card was used 6 times within an hour — over the limit of 5. "
                "Grey band → the LLM investigates and an analyst is asked to consent.",
    "high_value": "A ₹250,000 purchase that is clean on every risk signal. The rule "
                  "score stays low, but the amount alone routes it to investigation.",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.CHECKPOINTS_DB.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(config.CHECKPOINTS_DB)) as saver:
        app.state.graph = build_graph(checkpointer=saver)
        app.state.pending = {}          # thread_id -> initial transaction
        app.state.resume = {}           # thread_id -> asyncio.Queue for resume values
        yield


app = FastAPI(lifespan=lifespan)


@app.post("/run")
async def run(body: dict):
    """Register a run and return its thread_id. Streaming happens over the WS."""
    if "default" in body:
        txn = DEFAULTS.get(body["default"])
        if txn is None:
            return {"error": f"unknown default {body['default']!r}"}
    else:
        txn = body.get("transaction", body)
    thread_id = str(uuid.uuid4())
    app.state.pending[thread_id] = txn
    app.state.resume[thread_id] = asyncio.Queue()
    return {"thread_id": thread_id, "transaction": txn}


@app.post("/resume/{thread_id}")
async def resume(thread_id: str, body: dict):
    """Feed a human input to the interrupt the WebSocket is waiting on."""
    queue = app.state.resume.get(thread_id)
    if queue is None:
        return {"error": "unknown thread_id"}
    await queue.put(body.get("value", ""))
    return {"ok": True}


def _interrupt_message(update: dict) -> dict:
    payload = update["__interrupt__"][0].value
    return {"type": "interrupt", "kind": payload.get("kind"), "payload": payload}


@app.websocket("/stream/{thread_id}")
async def stream(ws: WebSocket, thread_id: str):
    await ws.accept()
    graph = ws.app.state.graph
    txn = ws.app.state.pending.get(thread_id)
    if txn is None:
        await ws.send_json({"type": "error", "message": "unknown thread_id"})
        await ws.close()
        return

    cfg = {
        "configurable": {"thread_id": thread_id},
        # Context for LangSmith so a trace is legible at a glance.
        "run_name": f"fraudgraph · {txn.get('txn_id', 'txn')}",
        "tags": ["fraudgraph", "web"],
        "metadata": {
            "thread_id": thread_id,
            "txn_id": txn.get("txn_id"),
            "card_id": txn.get("card_id"),
            "amount": txn.get("amount"),
            "merchant_id": txn.get("merchant_id"),
            "geo": txn.get("geo"),
        },
    }
    graph_input = {"transaction": txn}

    # Serialize all sends (the keepalive task and the stream loop share one ws).
    send_lock = asyncio.Lock()

    async def safe_send(msg):
        async with send_lock:
            await ws.send_json(msg)

    async def keepalive():
        # A node like `investigate` can run 20-60s with no stream output; without
        # traffic the proxy drops the idle socket. Send a light ping to hold it.
        while True:
            await asyncio.sleep(KEEPALIVE_SEC)
            await safe_send({"type": "ping"})

    ka_task = asyncio.create_task(keepalive())
    try:
        while True:
            interrupted = False
            # "debug" gives us a node-START event (task) so the UI can highlight a
            # node WHILE it runs (e.g. the ~5s draft_sar LLM call), not only after.
            async for mode, chunk in graph.astream(
                graph_input, cfg, stream_mode=["updates", "debug", "custom"]
            ):
                if mode == "debug":
                    if chunk.get("type") == "task":
                        await safe_send({"type": "node_start", "node": chunk["payload"]["name"]})
                    continue
                if mode == "custom":
                    # fine-grained progress step emitted from inside a node
                    await safe_send({"type": "step", "message": chunk.get("step", "")})
                    continue
                # mode == "updates": node finished (carries the state delta)
                if "__interrupt__" in chunk:
                    await safe_send(_interrupt_message(chunk))
                    interrupted = True
                    continue
                for node, delta in chunk.items():
                    await safe_send({"type": "node", "node": node, "delta": delta or {}})
                    await asyncio.sleep(PACING_SEC)
            if not interrupted:
                break
            # Wait for the human input from POST /resume, then continue streaming.
            value = await ws.app.state.resume[thread_id].get()
            graph_input = Command(resume=value)

        snapshot = await graph.aget_state(cfg)
        await safe_send({"type": "result", "state": snapshot.values})
    except WebSocketDisconnect:
        return
    finally:
        ka_task.cancel()
        ws.app.state.pending.pop(thread_id, None)
        ws.app.state.resume.pop(thread_id, None)


@app.get("/defaults")
async def defaults():
    """Descriptions + payloads for the demo transactions (drives the UI)."""
    return {k: {"description": DEFAULT_META.get(k, ""), "transaction": v}
            for k, v in DEFAULTS.items()}


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

"""Run the fraud graph on a transaction and print the path node by node.

Examples:
    uv run python scripts/run_cli.py --default clean
    uv run python scripts/run_cli.py --default velocity
    uv run python scripts/run_cli.py --txn '{"card_id":"CARD_0001","amount":1200,...}'
    uv run python scripts/run_cli.py --txn path/to/txn.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path

from langgraph.types import Command

from fraudgraph.data.generate import BLOCKED, CLEAN, HIGH_VALUE, REFERENCE_NOW, VELOCITY
from fraudgraph.graph.build import open_graph

_TS = REFERENCE_NOW.strftime("%Y-%m-%d %H:%M:%S")
_TXN_FIELDS = ("txn_id", "card_id", "device_id", "ip", "merchant_id", "amount", "geo")


def _demo(crafted: dict) -> dict:
    txn = {k: crafted[k] for k in _TXN_FIELDS}
    txn["ts"] = _TS
    return txn


DEFAULTS = {
    "clean": _demo(CLEAN),
    "blocklist": _demo(BLOCKED),
    "velocity": _demo(VELOCITY),
    "high_value": _demo(HIGH_VALUE),
}


def _load_txn(args) -> dict:
    if args.default:
        return DEFAULTS[args.default]
    raw = args.txn
    p = Path(raw)
    if p.exists():
        return json.loads(p.read_text())
    return json.loads(raw)


def _fmt_delta(delta: dict) -> str:
    bits = []
    for key in ("risk_band", "rule_score", "ml_score", "decision"):
        if key in delta:
            val = delta[key]
            bits.append(f"{key}={val:.2f}" if isinstance(val, float) else f"{key}={val}")
    if "reasons" in delta:
        bits.append(f"reasons={delta['reasons']}")
    if "evidence" in delta:
        ev = delta["evidence"]
        bits.append(f"blocklist_hit={ev['blocklist']['hit']} velocity={ev['velocity']}")
    if "recommendation" in delta:
        rec = delta["recommendation"]
        bits.append(f"recommend={rec.get('recommend')} (conf {rec.get('confidence')})")
    if "analyst_summary" in delta:
        bits.append("analyst_summary set")
    if "sar_draft" in delta:
        bits.append("sar_draft set")
    return "  ".join(bits)


def _prompt_for_resume(payload: dict):
    """Ask the terminal for the value the interrupted graph is waiting on."""
    kind = payload.get("kind")
    print(f"\n  ⏸  PAUSED for {kind}")
    if kind == "consent":
        rec = payload.get("recommendation") or {}
        print(f"     LLM recommends: {rec.get('recommend')} — {rec.get('rationale')}")
        return input("     decision [approve/decline/step_up/escalate]: ").strip()
    if kind == "otp":
        print(f"     OTP sent to {payload.get('email')}")
        return input("     enter OTP code: ").strip()
    return input("     resume value: ").strip()


async def run(txn: dict) -> dict:
    thread_id = str(uuid.uuid4())
    cfg = {
        "configurable": {"thread_id": thread_id},
        "run_name": f"fraudgraph · {txn.get('txn_id', 'txn')}",
        "tags": ["fraudgraph", "cli"],
        "metadata": {"thread_id": thread_id, "txn_id": txn.get("txn_id"),
                     "card_id": txn.get("card_id"), "amount": txn.get("amount")},
    }
    print(f"\nthread {thread_id}")
    print(f"txn    {txn}\n")
    print("path:")

    final: dict = {}
    async with open_graph() as graph:
        graph_input = {"transaction": txn}
        while True:
            interrupt_payload = None
            async for update in graph.astream(graph_input, cfg, stream_mode="updates"):
                if "__interrupt__" in update:
                    interrupt_payload = update["__interrupt__"][0].value
                    continue
                for node, delta in update.items():
                    print(f"  → {node:<13} {_fmt_delta(delta or {})}")
            if interrupt_payload is None:
                break
            resume_value = _prompt_for_resume(interrupt_payload)
            graph_input = Command(resume=resume_value)
        final = await graph.aget_state(cfg)

    values = final.values
    print("\nresult:")
    print(f"  decision  : {values.get('decision')}")
    print(f"  risk_band : {values.get('risk_band')}")
    print(f"  rule_score: {values.get('rule_score')}")
    print(f"  ml_score  : {values.get('ml_score'):.3f}" if values.get('ml_score') is not None else "  ml_score  : n/a")
    print(f"  reasons   : {values.get('reasons')}")
    if values.get("recommendation"):
        rec = values["recommendation"]
        print(f"  LLM recommends: {rec.get('recommend')} "
              f"(confidence {rec.get('confidence')}) — {rec.get('rationale')}")
    if values.get("analyst_summary"):
        print(f"\n  analyst summary:\n    {values['analyst_summary']}")
    if values.get("sar_draft"):
        print(f"\n  SAR draft:\n    {values['sar_draft']}")
    return values


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the fraud graph on one transaction.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--default", choices=sorted(DEFAULTS), help="use a canned demo transaction")
    group.add_argument("--txn", help="transaction as inline JSON or a path to a JSON file")
    args = ap.parse_args()

    asyncio.run(run(_load_txn(args)))


if __name__ == "__main__":
    main()

# Demo transactions

Four canned inputs, each exercising a distinct path through the graph.

| File | Path | Final decision |
|---|---|---|
| `clean.json` | ingest → enrich → score → **auto_approve** → finalize | approve (automatic) |
| `blocklist.json` | ingest → enrich → score → **auto_decline** → draft_sar → finalize | decline (automatic) |
| `velocity.json` | ingest → enrich → score → **investigate** → human_gate (consent) | analyst decides |
| `high_value.json` | ingest → enrich → score → **investigate** → human_gate → **step_up (OTP)** | approve after OTP |

Run one from the CLI:

```bash
uv run python scripts/run_cli.py --txn demo/transactions/velocity.json
```

Or in the web app (http://localhost:8000), pick it from the dropdown or upload the JSON.

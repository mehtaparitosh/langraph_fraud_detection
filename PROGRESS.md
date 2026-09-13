# FraudGraph — Progress (Phases 0–2 done)

Local-first agentic fraud-detection demo on LangGraph.
**Governing rule: the LLM/ML never authorizes a decision** — a deterministic
engine or a human decides; models only score and (later) explain.

## Status

- [x] **Phase 0 — Scaffold**: `uv` + Python 3.12 project, all deps, `config.py`, `.env`, `.gitignore`.
- [x] **Phase 1 — Data layer**: synthetic data, 6 SQLite tables, 4 lookups, tests.
- [x] **Phase 2 — Scoring engine**: additive rules → score/band/reasons, tiny ML model (XGBoost).
- [x] **Phase 3 — LangGraph skeleton**: deterministic spine + router + `AsyncSqliteSaver` + CLI, no LLM.
- [x] **Phase 4 — LLM investigate + RAG**: tool-calling agent, Chroma RAG, SAR drafter, governance guard.
- [x] **Phase 5 — human gate + step-up (OTP) + email**: `interrupt()` consent + OTP, idempotent SMTP send.
- [x] **Phase 6 — web app**: FastAPI + WebSocket, live node highlighting, consent/OTP panels, result view.
- [ ] Phase 7 — polish + demo script + AWS talk track ← next

## Run it

```bash
uv sync --extra dev
cp .env.example .env                              # keys already set locally
uv run python -m fraudgraph.data.seed             # build data/fraud_demo.db from committed CSV seeds
uv run python -m fraudgraph.scoring.train_model   # train + save data/model.pkl (AUC ~0.998)
uv run python scripts/run_cli.py --default velocity  # run the graph, watch the path node-by-node
uv run pytest                                     # 28 tests, all green
```

## What each piece does

| Area | File | Purpose |
|---|---|---|
| Config | `src/fraudgraph/config.py` | thresholds, band cutoffs, velocity, OTP, paths, provider — nothing hardcoded |
| Data gen | `data/generate.py` | deterministic synthetic data (~2.5k txns) + 4 crafted rows → committed CSVs |
| Seed | `data/seed.py` | create schema + load CSVs into SQLite |
| Lookups | `data/db.py` | `check_blocklist`, `velocity`, `device_reputation`, `merchant_reputation`, `gather_evidence` |
| Features | `scoring/features.py` | evidence → feature dict / ML vector (single source, no train/infer drift) |
| Rules | `scoring/rules.py` | **the only place a score is set** — additive, every point has a reason code |
| Model | `scoring/model.py` + `train_model.py` | tiny XGBoost model; predicts 0–1 `ml_score` as an *input* to the rules |
| Graph | `graph/state.py` `router.py` `nodes.py` `build.py` | `FraudState` contract, deterministic router, nodes, compiled graph + `AsyncSqliteSaver`, LLM guard |
| RAG | `rag/index.py` + `retriever.py` | Chroma + `all-MiniLM-L6-v2`; 16 cases + 2 policies in `data/seed/` |
| LLM | `llm/factory.py` + `prompts.py` | provider switch (Anthropic↔OpenAI); investigate/SAR prompts |
| Tools | `graph/tools.py` | `@tool` wrappers over lookups + `similar_past_cases` |
| HITL | `graph/nodes.py` (`human_gate`, `step_up`) | `interrupt()` for consent + OTP; step-up sets decision |
| Email/OTP | `messenger.py` + `notifications.py` | SMTP send; idempotent OTP make/send/verify (tiny SQLite store) |
| CLI | `scripts/run_cli.py` | run a transaction; interactive consent + OTP prompts on interrupt |
| Web | `app/server.py` + `app/static/` | FastAPI `/run` + WS `/stream` + `/resume`; live graph UI, consent/OTP panels |

## The four crafted transactions (the demo backbone)

| Case | Band | Rule score | Signal | Intended path |
|---|---|---|---|---|
| CLEAN | low | 0 | none | auto-approve |
| VELOCITY | grey | 40 | 6 txns > limit in 60m | investigate → human |
| BLOCKED | high | 100 | blocklist (hard override) | auto-decline |
| HIGH_VALUE | low | 10 | amount over ₹50k only | investigate (by amount) |

Governance highlight: on HIGH_VALUE the **ML model predicts fraud (1.00)** but the
deterministic engine keeps it out of "high" and sends it to a human — the model
is a +15-point input, never the decider.

## Notes

- Data files (`*.db`, `model.pkl`, `chroma/`) are gitignored; the CSV seeds in `data/seed/` are committed and regenerate everything deterministically.
- Email in this project goes through a `messenger.py` (SMTP) module, wired up in Phase 5.

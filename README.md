# FraudGraph

A local-first agentic fraud-detection demo on **LangGraph**, built phase by phase.

> **Governance rule of the whole system: the LLM never authorizes a decision.**
> A deterministic engine or a human sets the decision; the LLM only gathers
> evidence and explains.

## What runs locally vs. leaves the machine

- **Local:** the graph, ML score, SQLite/DuckDB, Chroma vector store, checkpointer, web app.
- **Leaves the machine:** LLM API calls + email (SendGrid). Nothing else.

## Setup

```bash
uv sync --extra dev          # install deps (Python 3.12, managed by uv)
cp .env.example .env         # fill in API keys + demo email
uv run python -m fraudgraph.config   # sanity-check config loads
uv run python -m fraudgraph.data.seed         # build data/fraud_demo.db from committed seeds
uv run python -m fraudgraph.scoring.train_model  # train + save data/model.pkl
uv run python -m fraudgraph.rag.index         # build the Chroma index (downloads MiniLM once)
uv run pytest                                 # run the test suite (35 tests)
```

> Phases 4+ call the LLM (Anthropic/OpenAI, set in `.env`); the deterministic
> paths (auto-approve/decline) and all lookups run with no API key.

> Optional: `brew install libomp` enables XGBoost; without it the trainer
> cleanly falls back to scikit-learn logistic regression (AUC ~0.998).

Run the graph on a canned demo transaction and watch the path:

```bash
uv run python scripts/run_cli.py --default clean       # -> auto_approve
uv run python scripts/run_cli.py --default blocklist   # -> auto_decline
uv run python scripts/run_cli.py --default velocity    # -> investigate (grey)
uv run python scripts/run_cli.py --default high_value  # -> investigate (by amount)
```

Or launch the web app and drive it in the browser:

```bash
uv run uvicorn app.server:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000, pick a demo, and watch the graph run node-by-node;
grey / high-value cases show a consent panel (Approve / Step-up / Decline /
Escalate), and Step-up shows an OTP box wired to a real email.

## Build status

Built in order; a phase doesn't start until the previous phase's "Done when" passes.

- [x] **Phase 0 — Scaffold**: runnable project, deps, config, env, gitignore.
- [x] **Phase 1 — Local data layer**: synthetic data, 6 SQLite tables, 4 lookups + tests.
- [x] **Phase 2 — Deterministic scoring engine**: additive rules, bands, reason codes, tiny ML model.
- [x] **Phase 3 — LangGraph skeleton (deterministic)**: full spine, router, `AsyncSqliteSaver`, CLI, tests.
- [x] **Phase 4 — LLM investigate node + RAG**: tool-calling agent, Chroma RAG, SAR drafter, governance guard.
- [x] **Phase 5 — Human-in-the-loop + step-up (OTP) + email**: `interrupt()` consent + OTP, SMTP messenger.
- [x] **Phase 6 — Web app (FastAPI + WebSocket + frontend)**: live node highlighting, consent/OTP panels, result view.
- [ ] Phase 7 — Polish, demo script, AWS talk track

## Layout

```
src/fraudgraph/   config + data / scoring / rag / llm / graph packages
app/              FastAPI server + static frontend (phase 6)
scripts/          run_cli.py (phase 3+)
tests/            pytest suites
data/seed/        committed CSV seeds, cases, policies
demo/             canned demo transactions
```

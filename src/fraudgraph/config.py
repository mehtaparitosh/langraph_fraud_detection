"""Central configuration for FraudGraph.

Every tunable knob lives here so nothing is hardcoded in the logic modules.
Values come from `.env` where it makes sense (secrets, provider choice) and
have sensible defaults for everything else so the system runs out of the box.

Governance note: none of these knobs let the LLM authorize a decision. The
band cutoffs and thresholds below feed the *deterministic* engine only.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Keep the local embedding/tokenizer stack lean and quiet: no tokenizer fork
# parallelism (avoids warnings + semaphore leaks) and modest thread counts so
# the whole app stays comfortably within ~1GB when the graph runs.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# --- LangSmith tracing ---
# If a LangSmith API key is present, turn tracing on and group this app's runs
# under their own project (override the generic course default). Customise the
# project name with FRAUDGRAPH_LANGSMITH_PROJECT if you like. LangChain reads
# these env vars automatically — no other code change is needed.
if os.getenv("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    os.environ["LANGSMITH_PROJECT"] = os.getenv("FRAUDGRAPH_LANGSMITH_PROJECT", "fraudgraph")

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# config.py -> fraudgraph -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SEED_DIR = DATA_DIR / "seed"

# The three generated (gitignored) files.
DB_PATH = Path(os.getenv("FRAUD_DB_PATH", DATA_DIR / "fraud_demo.db"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", DATA_DIR / "chroma"))
CHECKPOINTS_DB = Path(os.getenv("CHECKPOINTS_DB", DATA_DIR / "checkpoints.db"))

# Trained ML model artifact.
MODEL_PATH = Path(os.getenv("MODEL_PATH", DATA_DIR / "model.pkl"))

# Tiny store for issued OTPs (survives the step-up node re-running on resume).
OTP_DB = Path(os.getenv("OTP_DB", DATA_DIR / "otp.db"))


def _as_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _as_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


# --------------------------------------------------------------------------
# Amount branch
# --------------------------------------------------------------------------
# High-value transactions route to investigation regardless of a low score.
HIGH_VALUE_THRESHOLD = _as_float("HIGH_VALUE_THRESHOLD", 50_000.0)

# --------------------------------------------------------------------------
# Risk bands on the 0-100 deterministic rule score
# --------------------------------------------------------------------------
LOW_MAX = _as_float("LOW_MAX", 30.0)    # score <= LOW_MAX          -> "low"
GREY_MAX = _as_float("GREY_MAX", 70.0)  # LOW_MAX < score <= GREY_MAX -> "grey"
# score > GREY_MAX -> "high"

# --------------------------------------------------------------------------
# Velocity window
# --------------------------------------------------------------------------
VELOCITY_WINDOW_MIN = _as_int("VELOCITY_WINDOW_MIN", 60)
VELOCITY_MAX_TXNS = _as_int("VELOCITY_MAX_TXNS", 5)
# "Extreme" velocity is deterministic enough to auto-decline (vs. investigate).
VELOCITY_EXTREME_TXNS = _as_int("VELOCITY_EXTREME_TXNS", VELOCITY_MAX_TXNS * 3)

# --------------------------------------------------------------------------
# LLM provider (swappable Anthropic <-> OpenAI)
# --------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --------------------------------------------------------------------------
# OTP / step-up
# --------------------------------------------------------------------------
OTP_TTL_SEC = _as_int("OTP_TTL_SEC", 300)
OTP_LENGTH = _as_int("OTP_LENGTH", 6)

# --------------------------------------------------------------------------
# Email / notifications (SMTP, e.g. Gmail app password) — used by messenger.py
# --------------------------------------------------------------------------
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "")
DEMO_USER_EMAIL = os.getenv("DEMO_USER_EMAIL", "")


def band_for_score(score: float) -> str:
    """Map a 0-100 rule score to a risk band using the configured cutoffs."""
    if score <= LOW_MAX:
        return "low"
    if score <= GREY_MAX:
        return "grey"
    return "high"


if __name__ == "__main__":
    # Quick sanity dump: `uv run python -m fraudgraph.config`
    print("FraudGraph config")
    print(f"  PROJECT_ROOT         = {PROJECT_ROOT}")
    print(f"  DB_PATH              = {DB_PATH}")
    print(f"  CHROMA_DIR           = {CHROMA_DIR}")
    print(f"  CHECKPOINTS_DB       = {CHECKPOINTS_DB}")
    print(f"  MODEL_PATH           = {MODEL_PATH}")
    print(f"  HIGH_VALUE_THRESHOLD = {HIGH_VALUE_THRESHOLD}")
    print(f"  LOW_MAX / GREY_MAX   = {LOW_MAX} / {GREY_MAX}")
    print(f"  VELOCITY             = {VELOCITY_MAX_TXNS} txns / {VELOCITY_WINDOW_MIN} min")
    print(f"  LLM_PROVIDER         = {LLM_PROVIDER}")
    print(f"  OTP                  = {OTP_LENGTH} digits, {OTP_TTL_SEC}s TTL")

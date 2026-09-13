"""FraudGraph — local-first agentic fraud-detection demo on LangGraph.

Governance rule of the whole system: the LLM never authorizes a decision.
A deterministic engine or a human sets the decision; the LLM only gathers
evidence and explains.
"""

__version__ = "0.1.0"


def main() -> None:
    """Entry point stub. Real CLI lives in scripts/run_cli.py (phase 3+)."""
    from fraudgraph import config

    print(f"FraudGraph v{__version__} — provider={config.LLM_PROVIDER}")
    print("Scaffold ready. See the build plan for phase-by-phase steps.")

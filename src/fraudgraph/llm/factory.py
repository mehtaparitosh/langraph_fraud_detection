"""Provider-agnostic chat-model factory.

``get_model()`` returns a LangChain chat model chosen by ``LLM_PROVIDER`` in
``.env`` (anthropic | openai), so the whole app can flip providers without a
code change. Small, fast models by default (see config).
"""

from __future__ import annotations

from functools import lru_cache

from fraudgraph import config


@lru_cache(maxsize=4)
def get_model(provider: str | None = None, temperature: float = 0.0):
    """Return a LangChain chat model for the configured (or given) provider."""
    provider = (provider or config.LLM_PROVIDER or "anthropic").lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set in .env")
        return ChatAnthropic(
            model=config.ANTHROPIC_MODEL,
            api_key=config.ANTHROPIC_API_KEY,
            temperature=temperature,
            max_tokens=1024,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set in .env")
        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=temperature,
        )

    raise ValueError(f"unknown LLM_PROVIDER: {provider!r} (expected 'anthropic' or 'openai')")

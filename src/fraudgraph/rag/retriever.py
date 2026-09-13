"""Retrieve similar past cases / policies from the Chroma index.

Auto-builds the index on first use if it is empty, so callers never have to
remember to run the indexer.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fraudgraph import config
from fraudgraph.rag import index as rag_index


@lru_cache(maxsize=4)
def _get_collection(chroma_dir: str | None = None):
    client = rag_index._client(chroma_dir)
    try:
        col = client.get_collection(
            rag_index.COLLECTION, embedding_function=rag_index._embedding_function()
        )
    except Exception:
        rag_index.build_index(Path(chroma_dir) if chroma_dir else None)
        col = client.get_collection(
            rag_index.COLLECTION, embedding_function=rag_index._embedding_function()
        )
    if col.count() == 0:
        rag_index.build_index(Path(chroma_dir) if chroma_dir else None)
    return col


def similar_past_cases(query: str, k: int = 3, chroma_dir: Path | None = None) -> list[dict]:
    """Return the ``k`` most similar corpus entries to ``query``.

    Each item: ``{"id", "source", "kind", "text", "distance"}`` (lower distance
    = more similar). Includes cases and policy docs.
    """
    col = _get_collection(str(chroma_dir) if chroma_dir else None)
    res = col.query(query_texts=[query], n_results=k)
    out = []
    for i, doc in enumerate(res["documents"][0]):
        meta = res["metadatas"][0][i]
        out.append({
            "id": res["ids"][0][i],
            "source": meta.get("source"),
            "kind": meta.get("kind"),
            "text": doc,
            "distance": res["distances"][0][i] if res.get("distances") else None,
        })
    return out

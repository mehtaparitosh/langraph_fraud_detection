"""Build the Chroma vector index from the seed cases + policies.

Fully local: embeddings come from sentence-transformers ``all-MiniLM-L6-v2``
(~90MB, downloaded once, then cached). The index persists under
``config.CHROMA_DIR`` so it survives restarts.

Run:  ``uv run python -m fraudgraph.rag.index``
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fraudgraph import config

COLLECTION = "fraud_knowledge"
EMBED_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _embedding_function():
    # Cached: the SentenceTransformer weights load once per process, not on
    # every retrieval (a per-call reload made the investigate node very slow).
    from chromadb.utils import embedding_functions

    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


@lru_cache(maxsize=4)
def _client(chroma_dir: str | None = None):
    import chromadb

    path = Path(chroma_dir) if chroma_dir else config.CHROMA_DIR
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def _collect_docs(seed_dir: Path | None = None):
    """Read every case + policy markdown file into (id, text, metadata) tuples."""
    seed_dir = seed_dir or config.SEED_DIR
    docs = []
    for kind, sub in (("case", "cases"), ("policy", "policies")):
        for path in sorted((seed_dir / sub).glob("*.md")):
            docs.append((f"{kind}:{path.stem}", path.read_text(), {"kind": kind, "source": path.name}))
    return docs


def build_index(chroma_dir: Path | None = None, seed_dir: Path | None = None) -> int:
    """(Re)build the collection from the seed corpus. Returns the doc count."""
    client = _client(str(chroma_dir) if chroma_dir else None)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, embedding_function=_embedding_function())

    docs = _collect_docs(seed_dir)
    if not docs:
        return 0
    ids, texts, metas = zip(*docs)
    collection.add(ids=list(ids), documents=list(texts), metadatas=list(metas))
    return len(docs)


def main() -> None:
    n = build_index()
    print(f"Indexed {n} documents into '{COLLECTION}' at {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()

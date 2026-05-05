"""Embed chunks with BGE-small and persist to ChromaDB + BM25 (Block 3).

Embedding model: ``BAAI/bge-small-en-v1.5`` (512-token context, ~130MB).
Chosen over ``all-MiniLM-L6-v2`` so the spec's 500-word chunks fit within
the model's input window — see Block 3 Q1 decision (2026-05-04).

Persistence:
    - Vector index:  data/chunk_store/chromadb (Chroma PersistentClient,
      collection "chunks", cosine space)
    - Lexical index: data/chunk_store/bm25.pkl (BM25Okapi over tokenized
      chunk texts, plus chunk_id ↔ text mapping)
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Sequence

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from schemas.records import ChunkRecord


EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "chunks"
DEFAULT_CHROMA_PATH = Path("data/chunk_store/chromadb")
DEFAULT_BM25_PATH = Path("data/chunk_store/bm25.pkl")

_TOKEN_RE = re.compile(r"\b[a-z0-9][a-z0-9'-]*\b")

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------


def embed_chunks(
    chunks: Sequence[ChunkRecord], *, batch_size: int = 32
) -> list[list[float]]:
    """Encode chunk texts with BGE-small. Returns a list of L2-normalized
    embedding vectors (cosine-ready)."""
    model = _get_model()
    texts = [c.text for c in chunks]
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return [list(map(float, v)) for v in embs]


def build_chunk_store(
    chunks: Sequence[ChunkRecord],
    *,
    chroma_path: Path = DEFAULT_CHROMA_PATH,
    bm25_path: Path = DEFAULT_BM25_PATH,
) -> dict[str, int]:
    """Persist both the vector index and the BM25 index.

    Drops the existing Chroma collection (clean rebuild) and overwrites the
    BM25 pickle. Returns a small stats dict for logging.
    """
    chroma_path.mkdir(parents=True, exist_ok=True)
    bm25_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = embed_chunks(chunks)

    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    coll = client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    coll.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "source_document": c.source_document,
                "section": c.section,
                "position_index": c.position_index,
                "word_count": c.word_count,
            }
            for c in chunks
        ],
    )

    tokenized = [_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    payload = {
        "bm25": bm25,
        "chunk_ids": [c.chunk_id for c in chunks],
        "chunk_texts": {c.chunk_id: c.text for c in chunks},
        "tokenize_version": 1,
    }
    with bm25_path.open("wb") as f:
        pickle.dump(payload, f)

    return {
        "n_chunks": len(chunks),
        "n_embeddings": len(embeddings),
        "embed_dim": len(embeddings[0]) if embeddings else 0,
    }


# ---------------------------------------------------------------------------
# Querying (used by the smoke test)
# ---------------------------------------------------------------------------


def vector_search(
    query: str,
    k: int = 5,
    *,
    chroma_path: Path = DEFAULT_CHROMA_PATH,
) -> list[tuple[str, float, str]]:
    """Return ``[(chunk_id, distance, text), ...]`` sorted by ascending
    cosine distance (closer is better)."""
    model = _get_model()
    qemb = model.encode([query], normalize_embeddings=True)
    qvec = [list(map(float, qemb[0]))]
    client = chromadb.PersistentClient(path=str(chroma_path))
    coll = client.get_collection(name=COLLECTION_NAME)
    res = coll.query(query_embeddings=qvec, n_results=k)
    ids = res["ids"][0]
    distances = res["distances"][0]
    docs = res["documents"][0]
    return [(cid, float(d), txt) for cid, d, txt in zip(ids, distances, docs)]


def bm25_search(
    query: str,
    k: int = 5,
    *,
    bm25_path: Path = DEFAULT_BM25_PATH,
) -> list[tuple[str, float, str]]:
    """Return ``[(chunk_id, score, text), ...]`` sorted by descending BM25."""
    with bm25_path.open("rb") as f:
        payload = pickle.load(f)
    bm25: BM25Okapi = payload["bm25"]
    chunk_ids: list[str] = payload["chunk_ids"]
    chunk_texts: dict[str, str] = payload["chunk_texts"]
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)
    paired = sorted(
        zip(chunk_ids, (float(s) for s in scores)), key=lambda x: -x[1]
    )[:k]
    return [(cid, score, chunk_texts[cid]) for cid, score in paired]

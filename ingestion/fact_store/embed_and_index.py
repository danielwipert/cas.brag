"""Embed FactRecords with BGE-small and persist to ChromaDB (Block 4).

The Fact Store is a separate ChromaDB persistent client at
``data/fact_store/chromadb`` with a single collection ``facts``. Per the
build plan, the BM25 index over fact claims is **deferred to Block 5**,
when prose facts join the corpus and a single BM25 build covers both.

Embedding model: ``BAAI/bge-small-en-v1.5`` (same as Chunk Store, Block 3
decision). Cosine space. Metadata includes ``fact_type``, ``period``,
``source_document``, ``source_section``, ``concept_tag``, ``unit``,
``confidence``, plus an optional ``region`` parsed from the fact_id when
present — so Block 7's planner can filter by these without re-embedding.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import chromadb
from sentence_transformers import SentenceTransformer

from schemas.records import FactRecord


EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "facts"
DEFAULT_CHROMA_PATH = Path("data/fact_store/chromadb")

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def _region_from_fact_id(fact_id: str) -> str | None:
    """Extract trailing region tag (UCAN/EMEA/LATAM/APAC) from an XBRL
    regional fact_id. Returns None for aggregate facts."""
    for suffix in ("-UCAN", "-EMEA", "-LATAM", "-APAC"):
        if fact_id.endswith(suffix):
            return suffix.lstrip("-")
    return None


def _metadata_for(fact: FactRecord) -> dict:
    md: dict = {
        "fact_type": fact.fact_type.value,
        "source_document": fact.source_document,
        "source_section": fact.source_section,
        "concept_tag": fact.concept_tag or "",
        "confidence": fact.confidence,
    }
    if fact.period:
        md["period"] = fact.period
    if fact.unit:
        md["unit"] = fact.unit
    if fact.value is not None:
        md["value"] = fact.value
    region = _region_from_fact_id(fact.fact_id)
    if region:
        md["region"] = region
    return md


def embed_facts(
    facts: Sequence[FactRecord], *, batch_size: int = 32
) -> list[list[float]]:
    """Encode each fact's ``claim`` text with BGE-small. Returns L2-
    normalized vectors (cosine-ready)."""
    if not facts:
        return []
    model = _get_model()
    texts = [f.claim for f in facts]
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return [list(map(float, v)) for v in embs]


def build_fact_store(
    facts: Sequence[FactRecord],
    *,
    chroma_path: Path = DEFAULT_CHROMA_PATH,
) -> dict[str, int]:
    """Persist a clean Chroma collection of fact embeddings. Drops the
    existing collection (clean rebuild). Returns a small stats dict."""
    chroma_path.mkdir(parents=True, exist_ok=True)
    embeddings = embed_facts(facts)

    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    coll = client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    coll.add(
        ids=[f.fact_id for f in facts],
        documents=[f.claim for f in facts],
        embeddings=embeddings,
        metadatas=[_metadata_for(f) for f in facts],
    )
    return {
        "n_facts": len(facts),
        "n_embeddings": len(embeddings),
        "embed_dim": len(embeddings[0]) if embeddings else 0,
    }


def vector_search(
    query: str,
    k: int = 5,
    *,
    where: dict | None = None,
    chroma_path: Path = DEFAULT_CHROMA_PATH,
) -> list[tuple[str, float, str, dict]]:
    """Return ``[(fact_id, distance, claim, metadata), ...]`` ascending by
    cosine distance. ``where`` is a Chroma metadata filter (e.g.
    ``{"period": "2024Q3"}``)."""
    model = _get_model()
    qemb = model.encode([query], normalize_embeddings=True)
    qvec = [list(map(float, qemb[0]))]
    client = chromadb.PersistentClient(path=str(chroma_path))
    coll = client.get_collection(name=COLLECTION_NAME)
    res = coll.query(
        query_embeddings=qvec,
        n_results=k,
        where=where,
    )
    ids = res["ids"][0]
    distances = res["distances"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return [
        (cid, float(d), doc, dict(meta or {}))
        for cid, d, doc, meta in zip(ids, distances, docs, metas)
    ]

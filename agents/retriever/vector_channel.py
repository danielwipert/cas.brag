"""Block 8: semantic vector retrieval channel.

Embeds the slot's ``sub_question`` with the same model used to build the
indexes (BAAI/bge-small-en-v1.5; the spec calls for MiniLM but Block 3
swapped to BGE-small to fit 500-word chunks within the 512-token
context — see project_brag memory). Queries the Chunk Store and/or
Fact Store Chroma collections per the slot's ``target_layer`` and
returns ``ChannelCandidate``s with the cosine similarity as their
score (= 1 - cosine distance).
"""
from __future__ import annotations

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from agents.retriever.period_filter import (
    ChannelCandidate,
    equivalent_period_strings,
    equivalent_source_documents,
    period_from_document_id,
    source_document_from_chunk_id,
)
from schemas.enums import CandidateSource, TargetLayer


EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

CHUNK_CHROMA_PATH = Path("data/chunk_store/chromadb")
CHUNK_COLLECTION = "chunks"

FACT_CHROMA_PATH = Path("data/fact_store/chromadb")
FACT_COLLECTION = "facts"

_model: SentenceTransformer | None = None
_clients: dict[str, chromadb.PersistentClient] = {}


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def _get_client(path: Path) -> chromadb.PersistentClient:
    key = str(path)
    if key not in _clients:
        _clients[key] = chromadb.PersistentClient(path=key)
    return _clients[key]


def _facts_where_clause(period_filter: str | None) -> dict | None:
    """Chroma ``where`` clause for the facts collection that restricts
    candidates to those whose intrinsic ``period`` is equivalent to
    ``period_filter`` OR whose ``source_document``'s doc-derived period
    is equivalent (rescues prose facts that carry ``period=None`` but
    were extracted from a period-anchored document).

    Returns ``None`` when ``period_filter`` is None — caller skips the
    ``where`` argument so the query is unconstrained."""
    if period_filter is None:
        return None
    eq_periods = sorted(equivalent_period_strings(period_filter))
    eq_docs = sorted(equivalent_source_documents(period_filter))
    clauses: list[dict] = []
    if eq_periods:
        clauses.append({"period": {"$in": eq_periods}})
    if eq_docs:
        clauses.append({"source_document": {"$in": eq_docs}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$or": clauses}


def _chunks_where_clause(period_filter: str | None) -> dict | None:
    """Chroma ``where`` clause for chunks: chunks have no ``period``
    field, only ``source_document``. Constrain to chunks from docs
    whose doc-derived period is equivalent to the filter."""
    if period_filter is None:
        return None
    eq_docs = sorted(equivalent_source_documents(period_filter))
    if not eq_docs:
        return None
    return {"source_document": {"$in": eq_docs}}


def _query_facts(
    qvec: list[list[float]],
    k: int,
    period_filter: str | None = None,
) -> list[ChannelCandidate]:
    client = _get_client(FACT_CHROMA_PATH)
    coll = client.get_collection(name=FACT_COLLECTION)
    where = _facts_where_clause(period_filter)
    kwargs: dict = {"query_embeddings": qvec, "n_results": k}
    if where is not None:
        kwargs["where"] = where
    res = coll.query(**kwargs)
    out: list[ChannelCandidate] = []
    ids = res["ids"][0]
    distances = res["distances"][0]
    metas = res["metadatas"][0]
    for cid, dist, meta in zip(ids, distances, metas):
        meta = meta or {}
        out.append(
            ChannelCandidate(
                candidate_id=cid,
                source=CandidateSource.fact,
                score=1.0 - float(dist),  # cosine similarity
                source_document=str(meta.get("source_document", "")),
                period=meta.get("period") or None,
            )
        )
    return out


def _query_chunks(
    qvec: list[list[float]],
    k: int,
    period_filter: str | None = None,
) -> list[ChannelCandidate]:
    client = _get_client(CHUNK_CHROMA_PATH)
    coll = client.get_collection(name=CHUNK_COLLECTION)
    where = _chunks_where_clause(period_filter)
    kwargs: dict = {"query_embeddings": qvec, "n_results": k}
    if where is not None:
        kwargs["where"] = where
    res = coll.query(**kwargs)
    out: list[ChannelCandidate] = []
    ids = res["ids"][0]
    distances = res["distances"][0]
    for cid, dist in zip(ids, distances):
        doc_id = source_document_from_chunk_id(cid)
        out.append(
            ChannelCandidate(
                candidate_id=cid,
                source=CandidateSource.chunk,
                score=1.0 - float(dist),
                source_document=doc_id,
                period=period_from_document_id(doc_id),
            )
        )
    return out


def vector_search(
    query: str,
    target_layer: TargetLayer,
    k: int,
    period_filter: str | None = None,
) -> list[ChannelCandidate]:
    """Embed ``query`` and return up to ``k`` ChannelCandidate results
    per requested layer. For ``target_layer='both'``, runs both layer
    queries and returns the concatenation (RRF handles deduplication
    and ranking downstream).

    When ``period_filter`` is set, a Chroma ``where`` clause restricts
    the candidate pool to the period-equivalent subset BEFORE the
    top-K cutoff — without this, narrow period queries often had zero
    matches in the top-K because off-period content drowned out the
    relevant facts. See ``equivalent_period_strings`` /
    ``equivalent_source_documents`` for the equivalence rules.
    """
    model = _get_model()
    qemb = model.encode([query], normalize_embeddings=True)
    qvec = [list(map(float, qemb[0]))]
    out: list[ChannelCandidate] = []
    if target_layer in (TargetLayer.fact_store, TargetLayer.both):
        out.extend(_query_facts(qvec, k, period_filter))
    if target_layer in (TargetLayer.chunk_store, TargetLayer.both):
        out.extend(_query_chunks(qvec, k, period_filter))
    return out

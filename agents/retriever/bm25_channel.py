"""Block 8: lexical BM25 retrieval channel.

Uses the same tokenizer the indexes were built with (see
``ingestion.chunker.embed_and_index`` and
``scripts.build_corpus_combined_fact_store``). Loads both BM25 pickles
once and caches them.

The slot's ``key_terms`` are joined with whitespace into a single
query string before tokenization, so a multi-term list like
``["Netflix operating income Q3 2024", "operating income 2024Q3"]``
contributes every token via BM25's bag-of-words scoring.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from agents.retriever.period_filter import (
    ChannelCandidate,
    period_from_document_id,
    source_document_from_chunk_id,
)
from schemas.enums import CandidateSource, TargetLayer


CHUNK_BM25_PATH = Path("data/chunk_store/bm25.pkl")
FACT_BM25_PATH = Path("data/fact_store/bm25.pkl")

# Tokenizer must match ingestion/chunker/embed_and_index.py and
# scripts/build_corpus_combined_fact_store.py exactly. Both use the
# same pattern; we re-declare it here so this module is standalone.
_TOKEN_RE = re.compile(r"\b[a-z0-9][a-z0-9'-]*\b")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class _BM25Index:
    bm25: BM25Okapi
    ids: list[str]
    # Optional metadata captured at index time so period lookup doesn't
    # need a Chroma round-trip on the BM25 path. For the fact store
    # we don't have this and fall back to a Chroma lookup at retrieve
    # time; for chunks we derive from the chunk_id directly.
    fact_periods: dict[str, str | None] | None = None
    fact_source_documents: dict[str, str] | None = None


_cache: dict[str, _BM25Index] = {}


def _load_chunk_bm25() -> _BM25Index:
    if "chunks" in _cache:
        return _cache["chunks"]
    with CHUNK_BM25_PATH.open("rb") as fh:
        payload: dict[str, Any] = pickle.load(fh)
    idx = _BM25Index(bm25=payload["bm25"], ids=payload["chunk_ids"])
    _cache["chunks"] = idx
    return idx


def _load_fact_bm25() -> _BM25Index:
    if "facts" in _cache:
        return _cache["facts"]
    with FACT_BM25_PATH.open("rb") as fh:
        payload: dict[str, Any] = pickle.load(fh)
    idx = _BM25Index(bm25=payload["bm25"], ids=payload["fact_ids"])
    _cache["facts"] = idx
    return idx


def _topk(idx: _BM25Index, query_tokens: list[str], k: int) -> list[tuple[str, float]]:
    if not query_tokens:
        return []
    scores = idx.bm25.get_scores(query_tokens)
    pairs = sorted(zip(idx.ids, scores), key=lambda p: -p[1])[:k]
    return [(cid, float(s)) for cid, s in pairs if s > 0]


def _fact_metadata_lookup(
    ids: list[str],
) -> tuple[dict[str, str], dict[str, str | None]]:
    """Fetch source_document and period for a list of fact IDs by
    hitting the facts Chroma collection. Used only for BM25 hits since
    the BM25 pickle doesn't carry metadata."""
    if not ids:
        return {}, {}
    from agents.retriever.vector_channel import _get_client, FACT_CHROMA_PATH, FACT_COLLECTION

    client = _get_client(FACT_CHROMA_PATH)
    coll = client.get_collection(name=FACT_COLLECTION)
    res = coll.get(ids=ids, include=["metadatas"])
    got_ids = res["ids"]
    metas = res["metadatas"]
    src_map: dict[str, str] = {}
    period_map: dict[str, str | None] = {}
    for cid, meta in zip(got_ids, metas):
        meta = meta or {}
        src_map[cid] = str(meta.get("source_document", ""))
        period_map[cid] = meta.get("period") or None
    return src_map, period_map


def bm25_search(
    key_terms: list[str],
    target_layer: TargetLayer,
    k: int,
) -> list[ChannelCandidate]:
    """Run BM25 over the slot's ``key_terms`` against the requested
    layer(s). Returns up to ``k`` per-layer hits, concatenated for
    ``target_layer='both'``."""
    query_text = " ".join(key_terms or [])
    tokens = _tokenize(query_text)

    out: list[ChannelCandidate] = []

    if target_layer in (TargetLayer.fact_store, TargetLayer.both):
        idx = _load_fact_bm25()
        pairs = _topk(idx, tokens, k)
        if pairs:
            fact_ids = [cid for cid, _ in pairs]
            src_map, period_map = _fact_metadata_lookup(fact_ids)
            for cid, score in pairs:
                out.append(
                    ChannelCandidate(
                        candidate_id=cid,
                        source=CandidateSource.fact,
                        score=score,
                        source_document=src_map.get(cid, ""),
                        period=period_map.get(cid),
                    )
                )

    if target_layer in (TargetLayer.chunk_store, TargetLayer.both):
        idx = _load_chunk_bm25()
        pairs = _topk(idx, tokens, k)
        for cid, score in pairs:
            doc_id = source_document_from_chunk_id(cid)
            out.append(
                ChannelCandidate(
                    candidate_id=cid,
                    source=CandidateSource.chunk,
                    score=score,
                    source_document=doc_id,
                    period=period_from_document_id(doc_id),
                )
            )

    return out

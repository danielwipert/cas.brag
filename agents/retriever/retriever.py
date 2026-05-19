"""Block 8: Retriever orchestrator.

``retrieve(slot, ...)`` is the public entry point: it runs both
channels, applies period filtering, performs memory exclusion, fuses
via RRF, and returns a fully-populated ``RetrievalRecord``.

Per-tier K (channel input cap) and N (fused output cap) come from
spec §3.4 / §3.2:

    Simple   K=10  N=8
    Standard K=15  N=12
    Complex  K=20  N=16

Memory exclusion happens BEFORE RRF: the channels' raw outputs are
filtered for excluded IDs first, then ranked-and-fused. Excluded IDs
are returned on the record for the Memory Ledger to consume.
"""
from __future__ import annotations

from typing import Iterable

from agents.retriever.bm25_channel import bm25_search
from agents.retriever.period_filter import (
    ChannelCandidate,
    filter_by_period,
)
from agents.retriever.rrf import rrf_fuse
from agents.retriever.vector_channel import vector_search
from schemas.enums import ComplexityTier, PassOrigin
from schemas.records import EvidenceSlot, RetrievalCandidate, RetrievalRecord


# Per-tier channel K (cap on each channel's raw output) and final N
# (cap on the fused, period-filtered candidate set). Spec §3.4 + §3.2.
_K_BY_TIER: dict[ComplexityTier, int] = {
    ComplexityTier.simple: 10,
    ComplexityTier.standard: 15,
    ComplexityTier.complex: 20,
}
_N_BY_TIER: dict[ComplexityTier, int] = {
    ComplexityTier.simple: 8,
    ComplexityTier.standard: 12,
    ComplexityTier.complex: 16,
}


def _default_retrieval_id(slot_id: str, iteration: int) -> str:
    return f"R1_{slot_id}_iter{iteration}"


def _drop_excluded(
    candidates: list[ChannelCandidate],
    excluded: frozenset[str],
) -> list[ChannelCandidate]:
    if not excluded:
        return candidates
    return [c for c in candidates if c.candidate_id not in excluded]


def _build_fused_candidates(
    vector_hits: list[ChannelCandidate],
    bm25_hits: list[ChannelCandidate],
    n: int,
) -> list[RetrievalCandidate]:
    """Run RRF across the two channels and assemble RetrievalCandidate
    rows, attaching the original per-channel scores when available."""
    vec_score_by_id = {c.candidate_id: c.score for c in vector_hits}
    bm25_score_by_id = {c.candidate_id: c.score for c in bm25_hits}
    # ChannelCandidate carries the source type; first occurrence wins.
    source_by_id: dict[str, ChannelCandidate] = {}
    for c in vector_hits:
        source_by_id.setdefault(c.candidate_id, c)
    for c in bm25_hits:
        source_by_id.setdefault(c.candidate_id, c)

    rankings: list[list[str]] = [
        [c.candidate_id for c in vector_hits],
        [c.candidate_id for c in bm25_hits],
    ]
    fused = rrf_fuse(rankings)[:n]

    out: list[RetrievalCandidate] = []
    for f in fused:
        meta = source_by_id[f.candidate_id]
        out.append(
            RetrievalCandidate(
                candidate_id=f.candidate_id,
                source=meta.source,
                rrf_score=round(f.rrf_score, 6),
                vector_score=(
                    round(vec_score_by_id[f.candidate_id], 6)
                    if f.candidate_id in vec_score_by_id else None
                ),
                bm25_score=(
                    round(bm25_score_by_id[f.candidate_id], 6)
                    if f.candidate_id in bm25_score_by_id else None
                ),
            )
        )
    return out


def retrieve(
    slot: EvidenceSlot,
    *,
    complexity_tier: ComplexityTier,
    iteration: int = 1,
    pass_origin: PassOrigin = PassOrigin.verifier_loop,
    excluded_ids: Iterable[str] = (),
    retrieval_id: str | None = None,
) -> RetrievalRecord:
    """Run the full hybrid retrieval pipeline for one ``EvidenceSlot``.

    The returned ``RetrievalRecord`` is schema-valid and carries
    ``pass_origin``, ``period_filter``, the fused candidate set, and
    the list of IDs excluded via the ``excluded_ids`` argument."""
    k = _K_BY_TIER[complexity_tier]
    n = _N_BY_TIER[complexity_tier]
    excluded = frozenset(excluded_ids)

    # Channel 1: semantic vector. Period filter is applied INSIDE the
    # channel (Block 19) via a Chroma where-clause so the top-K is
    # taken over the period-equivalent subset, not globally; otherwise
    # off-period content dominates the top-K and the filter empties
    # the candidate set.
    vec_hits = vector_search(
        slot.sub_question, slot.target_layer, k,
        period_filter=slot.period_filter,
    )
    # Channel 2: lexical BM25. Same period-aware top-K.
    bm25_hits = bm25_search(
        list(slot.key_terms), slot.target_layer, k,
        period_filter=slot.period_filter,
    )

    # Defensive: the channels already constrain to the period-equivalent
    # subset, so this is a no-op for the constrained path. Keep it as a
    # safety net for any future caller that bypasses the channel-level
    # filter, and for cases where the channel returned candidates with
    # period=None whose doc-fallback needs evaluation.
    vec_hits = filter_by_period(vec_hits, slot.period_filter)
    bm25_hits = filter_by_period(bm25_hits, slot.period_filter)

    # Memory exclusion before fusion so excluded items don't displace
    # ranks of items the Verifier hasn't seen yet.
    vec_hits = _drop_excluded(vec_hits, excluded)
    bm25_hits = _drop_excluded(bm25_hits, excluded)

    candidates = _build_fused_candidates(vec_hits, bm25_hits, n)

    return RetrievalRecord(
        retrieval_id=retrieval_id or _default_retrieval_id(slot.slot_id, iteration),
        slot_id=slot.slot_id,
        iteration=iteration,
        pass_origin=pass_origin,
        vector_query=slot.sub_question,
        bm25_terms=list(slot.key_terms),
        period_filter=slot.period_filter,
        candidates=candidates,
        memory_exclusions=sorted(excluded),
    )

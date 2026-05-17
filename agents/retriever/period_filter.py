"""Block 8: post-retrieval period filtering.

Spec §3.4: period filtering is applied AFTER vector/BM25 retrieval, not
as a query-time constraint, to avoid distorting similarity scores.

For facts, the period is carried directly on the FactRecord (and on the
Chroma metadata). For chunks, no period field exists — we derive it
from the source_document id, which encodes the document's primary
reporting period:

    nflx-10q-2024-q3      -> "2024Q3"
    nflx-10k-2024         -> "FY2024"
    nflx-q3-2024-letter   -> "2024Q3"
    nflx-q3-2024-transcript -> "2024Q3"

A 10-K contains comparative FY2023/FY2022 references but is "anchored"
to its primary FY; queries that need the older FY should match against
the 10-K-{older-year} document instead. This v1 implementation accepts
that trade-off rather than attempting per-chunk period parsing.

When a period_filter is set and a candidate has no derivable period
(or its period mismatches), the candidate is dropped. Period-anchored
queries should not silently absorb unbounded claims.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from schemas.enums import CandidateSource


_TRANSCRIPT_LETTER_RE = re.compile(
    r"^nflx-(q[1-4])-(\d{4})-(?:letter|transcript)$", re.IGNORECASE
)
_TENQ_RE = re.compile(r"^nflx-10q-(\d{4})-(q[1-4])$", re.IGNORECASE)
_TENK_RE = re.compile(r"^nflx-10k-(\d{4})$", re.IGNORECASE)


@dataclass(frozen=True)
class ChannelCandidate:
    """A candidate returned by one of the retrieval channels. ``score``
    is the channel-native score (cosine similarity for vector,
    BM25 score for lexical)."""

    candidate_id: str
    source: CandidateSource
    score: float
    source_document: str
    period: str | None


def period_from_document_id(document_id: str) -> str | None:
    """Return the primary reporting period for a Netflix document id,
    or None if the format is unrecognized."""
    if not document_id:
        return None
    m = _TRANSCRIPT_LETTER_RE.match(document_id)
    if m:
        quarter = m.group(1).upper()  # "Q3"
        year = m.group(2)
        return f"{year}{quarter}"
    m = _TENQ_RE.match(document_id)
    if m:
        year = m.group(1)
        quarter = m.group(2).upper()
        return f"{year}{quarter}"
    m = _TENK_RE.match(document_id)
    if m:
        return f"FY{m.group(1)}"
    return None


def source_document_from_chunk_id(chunk_id: str) -> str:
    """Chunk IDs have the form ``{doc_id}__{section}__chunk_{n}``."""
    return chunk_id.split("__", 1)[0]


def filter_by_period(
    candidates: list[ChannelCandidate],
    period_filter: str | None,
) -> list[ChannelCandidate]:
    """Keep only candidates whose period matches ``period_filter``.

    Match rules:

    * ``period_filter is None`` — return everything unchanged.
    * Candidate's intrinsic period equals the filter — keep.
    * Candidate has ``period=None`` AND its ``source_document``'s
      derived period equals the filter — keep. This is load-bearing
      for fact types that are temporally unbounded by extraction
      design (risk_disclosure, strategic_claim, accounting_policy)
      but were asserted in a period-anchored document.

    Chunks already carry a doc-derived period from the channels, so
    the document-fallback branch is a no-op for them in practice."""
    if period_filter is None:
        return candidates
    out: list[ChannelCandidate] = []
    for c in candidates:
        if c.period == period_filter:
            out.append(c)
            continue
        if c.period is None and c.source_document:
            doc_period = period_from_document_id(c.source_document)
            if doc_period == period_filter:
                out.append(c)
    return out

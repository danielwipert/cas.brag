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

# Block 19: period equivalence for Netflix's calendar fiscal year

Netflix files on a calendar-year fiscal calendar (FY ends 12-31), so
XBRL instants at the year/quarter end are the natural pair for the
corresponding duration period:

    FY{Y}    <->  {Y}-12-31         (FY end)
    {Y}Q1    <->  {Y}-03-31         (Q1 end)
    {Y}Q2    <->  {Y}-06-30         (Q2 end)
    {Y}Q3    <->  {Y}-09-30         (Q3 end)
    {Y}Q4    <->  {Y}-12-31         (Q4 end)

The Planner emits ``2024-12-31`` for balance-sheet questions and
``FY2024`` for income-statement questions; the same 10-K contains
both kinds of facts. Without equivalence, an instant-style filter
drops every prose fact in the matching FY10-K (their doc-fallback
period is ``FY2024``), and an FY-style filter drops the instant
XBRL facts that live at ``2024-12-31``. Treating the pair as
equivalent at retrieval time lets both flow through; the Verifier
then filters by sub_question relevance, which is the right place
to enforce semantic narrowness.

``FY{Y}-guidance`` is never equivalent to anything else — it's a
forward-looking marker, not a historical period.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from schemas.enums import CandidateSource
from schemas.period import Period, parse_period


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


# Calendar-quarter end (month, day) for Netflix's fiscal calendar.
_QUARTER_END_MD: dict[int, tuple[int, int]] = {
    1: (3, 31),
    2: (6, 30),
    3: (9, 30),
    4: (12, 31),
}


def _try_parse(p: str | None) -> Period | None:
    if not p:
        return None
    try:
        return parse_period(p)
    except ValueError:
        return None


def periods_equivalent(a: str | None, b: str | None) -> bool:
    """Return True if ``a`` and ``b`` denote the same Netflix reporting
    anchor, treating FY <-> year-end-instant and quarter <-> quarter-end-
    instant as equivalent. See module docstring for the equivalence
    table. Returns False if either side is None or unparseable."""
    if a is None or b is None:
        return False
    if a == b:
        return True
    pa, pb = _try_parse(a), _try_parse(b)
    if pa is None or pb is None:
        return False
    if pa.year != pb.year:
        return False
    kinds = {pa.kind, pb.kind}
    # fy_guidance is forward-looking; never equivalent to a historical period.
    if "fy_guidance" in kinds:
        return False
    # FY{Y} <-> {Y}-12-31
    if kinds == {"fiscal_year", "instant"}:
        instant = pa if pa.kind == "instant" else pb
        assert instant.instant is not None
        return (instant.instant.month, instant.instant.day) == (12, 31)
    # {Y}Q{N} <-> quarter-end instant
    if kinds == {"quarter", "instant"}:
        quarter = pa if pa.kind == "quarter" else pb
        instant = pa if pa.kind == "instant" else pb
        assert quarter.quarter is not None and instant.instant is not None
        return (instant.instant.month, instant.instant.day) == _QUARTER_END_MD[quarter.quarter]
    return False


def equivalent_period_strings(period_filter: str | None) -> frozenset[str]:
    """Set of period strings that a candidate's intrinsic ``period``
    field could carry while still being equivalent to ``period_filter``
    under ``periods_equivalent``.

    Used by the channels to build a Chroma ``where`` clause / BM25 post-
    filter so retrieval ranks candidates within the period-equivalent
    subset, rather than ranking globally and discarding non-matches
    afterward (which often empties the top-K)."""
    if period_filter is None:
        return frozenset()
    p = _try_parse(period_filter)
    if p is None:
        return frozenset({period_filter})
    if p.kind == "fy_guidance":
        return frozenset({period_filter})
    out: set[str] = {period_filter}
    if p.kind == "fiscal_year":
        out.add(f"{p.year}-12-31")
    elif p.kind == "instant":
        assert p.instant is not None
        m, d = p.instant.month, p.instant.day
        if (m, d) == (12, 31):
            out.add(f"FY{p.year}")
            out.add(f"{p.year}Q4")
        elif (m, d) == (3, 31):
            out.add(f"{p.year}Q1")
        elif (m, d) == (6, 30):
            out.add(f"{p.year}Q2")
        elif (m, d) == (9, 30):
            out.add(f"{p.year}Q3")
    elif p.kind == "quarter":
        assert p.quarter is not None
        month, day = _QUARTER_END_MD[p.quarter]
        out.add(f"{p.year}-{month:02d}-{day:02d}")
    return frozenset(out)


def equivalent_source_documents(period_filter: str | None) -> frozenset[str]:
    """Netflix corpus doc IDs whose ``period_from_document_id`` would be
    equivalent to ``period_filter``. Used by the channels to constrain
    on ``source_document`` for candidates whose intrinsic period is
    None (prose facts) or for chunks (no period metadata).

    Returns an empty set for None / unparseable / fy_guidance filters."""
    if period_filter is None:
        return frozenset()
    p = _try_parse(period_filter)
    if p is None or p.kind == "fy_guidance":
        return frozenset()
    year = p.year
    doc_periods: set[str] = set()
    if p.kind == "fiscal_year":
        doc_periods.add(f"FY{year}")
    elif p.kind == "instant":
        assert p.instant is not None
        m, d = p.instant.month, p.instant.day
        if (m, d) == (12, 31):
            doc_periods.update({f"FY{year}", f"{year}Q4"})
        elif (m, d) == (3, 31):
            doc_periods.add(f"{year}Q1")
        elif (m, d) == (6, 30):
            doc_periods.add(f"{year}Q2")
        elif (m, d) == (9, 30):
            doc_periods.add(f"{year}Q3")
    elif p.kind == "quarter":
        doc_periods.add(f"{year}Q{p.quarter}")
    docs: set[str] = set()
    for dp in doc_periods:
        if dp.startswith("FY"):
            docs.add(f"nflx-10k-{dp[2:]}")
        else:
            m = re.match(r"^(\d{4})Q([1-4])$", dp)
            if m:
                y, q = m.group(1), m.group(2)
                docs.add(f"nflx-10q-{y}-q{q}")
                docs.add(f"nflx-q{q}-{y}-letter")
                docs.add(f"nflx-q{q}-{y}-transcript")
    return frozenset(docs)


def filter_by_period(
    candidates: list[ChannelCandidate],
    period_filter: str | None,
) -> list[ChannelCandidate]:
    """Keep only candidates whose period matches ``period_filter`` under
    the Netflix-fiscal-calendar equivalence (see ``periods_equivalent``).

    Match rules:

    * ``period_filter is None`` — return everything unchanged.
    * Candidate's intrinsic period is equivalent to the filter — keep.
    * Candidate has ``period=None`` AND its ``source_document``'s
      derived period is equivalent to the filter — keep. Load-bearing
      for narrative fact types (risk_disclosure, strategic_claim,
      accounting_policy) that have no extracted period but were
      asserted in a period-anchored document.

    Chunks already carry a doc-derived period from the channels, so
    the document-fallback branch is a no-op for them in practice."""
    if period_filter is None:
        return candidates
    out: list[ChannelCandidate] = []
    for c in candidates:
        if periods_equivalent(c.period, period_filter):
            out.append(c)
            continue
        if c.period is None and c.source_document:
            doc_period = period_from_document_id(c.source_document)
            if periods_equivalent(doc_period, period_filter):
                out.append(c)
    return out

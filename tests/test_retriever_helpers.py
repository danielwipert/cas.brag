"""Unit tests for the Block 8 retriever helpers (period derivation + RRF).

The vector and BM25 channels are integration-tested via the smoke
script; only the pure-logic helpers are unit-tested here."""
from __future__ import annotations

from agents.retriever.period_filter import (
    ChannelCandidate,
    filter_by_period,
    period_from_document_id,
    periods_equivalent,
    source_document_from_chunk_id,
)
from agents.retriever.rrf import rrf_fuse
from schemas.enums import CandidateSource


# ---------------------------------------------------------------------------
# Document id -> period
# ---------------------------------------------------------------------------


def test_letter_period() -> None:
    assert period_from_document_id("nflx-q3-2024-letter") == "2024Q3"


def test_transcript_period() -> None:
    assert period_from_document_id("nflx-q1-2026-transcript") == "2026Q1"


def test_tenq_period() -> None:
    assert period_from_document_id("nflx-10q-2024-q3") == "2024Q3"


def test_tenk_period() -> None:
    assert period_from_document_id("nflx-10k-2024") == "FY2024"


def test_unknown_returns_none() -> None:
    assert period_from_document_id("") is None
    assert period_from_document_id("foo-bar") is None


def test_source_document_from_chunk_id() -> None:
    assert source_document_from_chunk_id(
        "nflx-10q-2024-q3__notes_to_financial_statements__chunk_5"
    ) == "nflx-10q-2024-q3"
    assert source_document_from_chunk_id(
        "nflx-q3-2024-letter__letter_body__chunk_0"
    ) == "nflx-q3-2024-letter"


# ---------------------------------------------------------------------------
# filter_by_period
# ---------------------------------------------------------------------------


def _c(cid: str, period: str | None) -> ChannelCandidate:
    return ChannelCandidate(
        candidate_id=cid,
        source=CandidateSource.fact,
        score=1.0,
        source_document="x",
        period=period,
    )


def test_filter_passes_through_when_none() -> None:
    cs = [_c("A", "2024Q3"), _c("B", None)]
    assert filter_by_period(cs, None) == cs


def test_filter_drops_mismatched_period() -> None:
    cs = [_c("A", "2024Q3"), _c("B", "2024Q2"), _c("C", "2024Q3")]
    out = filter_by_period(cs, "2024Q3")
    assert [c.candidate_id for c in out] == ["A", "C"]


def test_filter_drops_none_period_when_filter_set() -> None:
    # Default _c() helper sets source_document="x" which has no derivable
    # period — so the document-fallback branch doesn't rescue B here.
    cs = [_c("A", "2024Q3"), _c("B", None)]
    out = filter_by_period(cs, "2024Q3")
    assert [c.candidate_id for c in out] == ["A"]


def _c_with_doc(cid: str, period: str | None, source_document: str) -> ChannelCandidate:
    return ChannelCandidate(
        candidate_id=cid,
        source=CandidateSource.fact,
        score=1.0,
        source_document=source_document,
        period=period,
    )


def test_filter_doc_period_fallback_rescues_none_period_fact() -> None:
    # A risk_disclosure fact with period=None but extracted from
    # nflx-10k-2022 should be kept when period_filter="FY2022".
    cs = [
        _c_with_doc("A", "FY2022", "nflx-10k-2022"),
        _c_with_doc("B", None, "nflx-10k-2022"),       # rescued
        _c_with_doc("C", None, "nflx-10k-2021"),       # dropped: wrong year
        _c_with_doc("D", None, "no-recognized-doc"),   # dropped: not parseable
    ]
    out = filter_by_period(cs, "FY2022")
    assert [c.candidate_id for c in out] == ["A", "B"]


# ---------------------------------------------------------------------------
# Block 19: period equivalence (FY <-> year-end instant, Q <-> Q-end instant)
# ---------------------------------------------------------------------------


def test_periods_equivalent_exact_match() -> None:
    assert periods_equivalent("FY2024", "FY2024")
    assert periods_equivalent("2024-12-31", "2024-12-31")
    assert periods_equivalent("2024Q1", "2024Q1")


def test_periods_equivalent_fy_to_year_end_instant() -> None:
    assert periods_equivalent("FY2024", "2024-12-31")
    assert periods_equivalent("2024-12-31", "FY2024")  # symmetric
    assert periods_equivalent("FY2019", "2019-12-31")


def test_periods_equivalent_quarter_to_quarter_end_instant() -> None:
    assert periods_equivalent("2024Q1", "2024-03-31")
    assert periods_equivalent("2024Q2", "2024-06-30")
    assert periods_equivalent("2024Q3", "2024-09-30")
    assert periods_equivalent("2024Q4", "2024-12-31")
    # Symmetric
    assert periods_equivalent("2024-03-31", "2024Q1")


def test_periods_equivalent_negatives() -> None:
    # Different years
    assert not periods_equivalent("FY2024", "2023-12-31")
    # Wrong end-of-quarter date
    assert not periods_equivalent("2024Q1", "2024-04-30")
    # FY does NOT match Q4 (different durations even though they share an end)
    assert not periods_equivalent("FY2024", "2024Q4")
    # FY-guidance is forward-looking; never equivalent to a historical period
    assert not periods_equivalent("FY2024-guidance", "FY2024")
    assert not periods_equivalent("FY2024-guidance", "2024-12-31")
    # ...but exact-match on FY-guidance still works (handled before the rule)
    assert periods_equivalent("FY2024-guidance", "FY2024-guidance")
    # None handling
    assert not periods_equivalent(None, "FY2024")
    assert not periods_equivalent("FY2024", None)
    assert not periods_equivalent(None, None)
    # Unparseable
    assert not periods_equivalent("garbage", "FY2024")


def test_filter_matches_fy_filter_against_year_end_instant_fact() -> None:
    # XBRL instant fact at 2024-12-31 should be retrieved by an FY2024 filter.
    cs = [
        _c_with_doc("A", "2024-12-31", "nflx-10k-2024"),  # XBRL instant
        _c_with_doc("B", "FY2024", "nflx-10k-2024"),       # XBRL duration
        _c_with_doc("C", "2024-09-30", "nflx-10k-2024"),  # Q3-end, not equivalent to FY2024
    ]
    out = filter_by_period(cs, "FY2024")
    assert [c.candidate_id for c in out] == ["A", "B"]


def test_filter_matches_instant_filter_against_fy_doc_fallback() -> None:
    # The Q4 case: Planner emits "2024-12-31" for a balance-sheet question.
    # Prose facts in the 2024 10-K have period=None and doc-fallback FY2024.
    # Under Block 19, those prose facts should be rescued.
    cs = [
        _c_with_doc("A", None, "nflx-10k-2024"),       # rescued via doc fallback
        _c_with_doc("B", "2024-12-31", "nflx-10k-2024"),  # exact match
        _c_with_doc("C", None, "nflx-10k-2023"),       # wrong year, dropped
    ]
    out = filter_by_period(cs, "2024-12-31")
    assert [c.candidate_id for c in out] == ["A", "B"]


def test_filter_matches_quarter_filter_against_quarter_end_instant() -> None:
    cs = [
        _c_with_doc("A", "2024-03-31", "nflx-10q-2024-q1"),  # Q1-end instant
        _c_with_doc("B", "2024-06-30", "nflx-10q-2024-q2"),  # Q2-end, dropped
    ]
    out = filter_by_period(cs, "2024Q1")
    assert [c.candidate_id for c in out] == ["A"]


def test_filter_does_not_conflate_fy_with_q4() -> None:
    # FY2024 (full-year duration) must NOT match a Q4 2024 (quarter
    # duration) fact — they're semantically different periods for
    # income-statement items.
    cs = [
        _c_with_doc("A", "FY2024", "nflx-10k-2024"),
        _c_with_doc("B", "2024Q4", "nflx-q4-2024-letter"),  # dropped
    ]
    out = filter_by_period(cs, "FY2024")
    assert [c.candidate_id for c in out] == ["A"]


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


def test_rrf_single_ranking_sorts_by_position() -> None:
    out = rrf_fuse([["A", "B", "C"]])
    assert [f.candidate_id for f in out] == ["A", "B", "C"]
    # Strictly decreasing.
    assert all(out[i].rrf_score > out[i + 1].rrf_score for i in range(len(out) - 1))


def test_rrf_dedupes_across_channels() -> None:
    # A appears at rank 1 in both lists; B at rank 2 in both; C at rank 3 in
    # one only. A should beat B should beat C.
    out = rrf_fuse([["A", "B", "C"], ["A", "B"]])
    assert [f.candidate_id for f in out] == ["A", "B", "C"]


def test_rrf_one_channel_only() -> None:
    # An item present in just one channel still appears, with a smaller
    # score than items in both.
    out = rrf_fuse([["A", "B"], ["B"]])
    ids = [f.candidate_id for f in out]
    assert ids == ["B", "A"]


def test_rrf_score_formula() -> None:
    # Single ranking with one item: score = 1/(60+1).
    out = rrf_fuse([["X"]])
    assert abs(out[0].rrf_score - 1.0 / 61.0) < 1e-9


def test_rrf_empty_rankings_returns_empty() -> None:
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []

"""Tests for the XBRL ingestion path (build plan Block 4)."""
from __future__ import annotations

from datetime import date

from ingestion.xbrl.concept_filter import (
    classify_dimensions,
    human_label,
    is_canonical,
    statement_section,
)
from ingestion.xbrl.build_fact_records import (
    _canonical_period,
    _format_anchor,
    _format_for_claim,
    _human_period,
    xbrl_to_fact_record,
)
from ingestion.xbrl.parse import XBRLFact


def _qtd_fact(**overrides) -> XBRLFact:
    base = dict(
        concept="us-gaap:Revenues",
        fact_id_raw="f-30",
        context_id="c-3",
        unit="USD",
        decimals=-3,
        raw_value="9824703000",
        numeric_value=9824703000.0,
        period_kind="qtd",
        period_end=date(2024, 9, 30),
        period_start=date(2024, 7, 1),
        instant=None,
        dimensions={},
    )
    base.update(overrides)
    return XBRLFact(**base)


def _instant_fact(**overrides) -> XBRLFact:
    base = dict(
        concept="us-gaap:CashAndCashEquivalentsAtCarryingValue",
        fact_id_raw="f-100",
        context_id="c-2",
        unit="USD",
        decimals=-3,
        raw_value="7800000000",
        numeric_value=7800000000.0,
        period_kind="instant",
        period_end=date(2024, 9, 30),
        period_start=None,
        instant=date(2024, 9, 30),
        dimensions={},
    )
    base.update(overrides)
    return XBRLFact(**base)


# ---------------------------------------------------------------------------
# concept_filter
# ---------------------------------------------------------------------------


def test_canonical_concepts_match_retention_set():
    assert is_canonical("us-gaap:Revenues")
    assert is_canonical("us-gaap:NetIncomeLoss")
    assert is_canonical("nflx:ContentAssetsNet")
    assert not is_canonical("us-gaap:NotARealConcept")
    assert not is_canonical("dei:DocumentPeriodEndDate")


def test_human_labels_present_for_all_retained():
    for concept in (
        "us-gaap:Revenues",
        "us-gaap:OperatingIncomeLoss",
        "us-gaap:NetCashProvidedByUsedInOperatingActivities",
        "nflx:NumberOfPaidMemberships",
    ):
        assert human_label(concept)
        assert statement_section(concept)


def test_classify_dimensions_aggregate():
    assert classify_dimensions({}) == ("aggregate", None)


def test_classify_dimensions_streaming_only_collapses_to_aggregate():
    assert classify_dimensions(
        {"srt:ProductOrServiceAxis": "nflx:StreamingMember"}
    ) == ("aggregate", None)


def test_classify_dimensions_geographic_kept():
    dims = {
        "srt:ProductOrServiceAxis": "nflx:StreamingMember",
        "srt:StatementGeographicalAxis": "nflx:UnitedStatesAndCanadaMember",
    }
    assert classify_dimensions(dims) == ("regional", "UCAN")


def test_classify_dimensions_other_axes_dropped():
    dims = {"us-gaap:DebtInstrumentAxis": "nflx:FivePointEightSevenFivePercentSeniorNotesMember"}
    assert classify_dimensions(dims) == ("drop", None)


def test_classify_dimensions_unknown_geographic_member_dropped():
    # Unrecognized geographic member: drop.
    dims = {"srt:StatementGeographicalAxis": "us-gaap:AntarcticaMember"}
    assert classify_dimensions(dims) == ("drop", None)


# ---------------------------------------------------------------------------
# Period formatting
# ---------------------------------------------------------------------------


def test_canonical_period_qtd():
    assert _canonical_period(_qtd_fact()) == "2024Q3"


def test_canonical_period_instant():
    assert _canonical_period(_instant_fact()) == "2024-09-30"


def test_canonical_period_fy():
    f = _qtd_fact(period_kind="fy", period_end=date(2024, 12, 31), period_start=date(2024, 1, 1))
    assert _canonical_period(f) == "FY2024"


def test_human_period_qtd_says_q3_2024():
    assert _human_period(_qtd_fact()) == "Q3 2024"


def test_human_period_instant_says_as_of():
    assert "September 30, 2024" in _human_period(_instant_fact())


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------


def test_anchor_usd_thousands_precision():
    # 9,824,703,000 USD reported with decimals=-3 -> "$9,824,703" in HTML.
    assert _format_anchor(_qtd_fact()) == "$9,824,703"


def test_anchor_eps():
    # EPS lives at full precision (decimals=2 typically).
    f = _qtd_fact(
        concept="us-gaap:EarningsPerShareBasic",
        unit="USDPerShare",
        decimals=2,
        raw_value="5.40",
        numeric_value=5.40,
    )
    assert _format_anchor(f) == "$5.40"


def test_claim_value_uses_compact_billions():
    assert _format_for_claim(_qtd_fact()) == "$9.82 billion"


def test_claim_value_negative_handled():
    f = _qtd_fact(numeric_value=-150_000_000.0, raw_value="-150000000")
    assert _format_for_claim(f).startswith("-$")


# ---------------------------------------------------------------------------
# End-to-end: xbrl_to_fact_record
# ---------------------------------------------------------------------------


def test_xbrl_to_fact_record_aggregate_revenue():
    rec = xbrl_to_fact_record(
        _qtd_fact(),
        source_document="nflx-10q-2024-q3",
        filing_date=date(2024, 10, 18),
    )
    assert rec is not None
    assert rec.fact_id == "F-XBRL-nflx-10q-2024-q3-us-gaap-Revenues-2024Q3"
    assert rec.period == "2024Q3"
    assert rec.value == 9824703000.0
    assert rec.unit == "USD"
    assert rec.concept_tag == "us-gaap:Revenues"
    assert rec.confidence == 1.00
    assert rec.assertion_date == date(2024, 10, 18)
    assert rec.verbatim_anchor == "$9,824,703"
    assert "Q3 2024" in rec.claim
    assert "$9.82 billion" in rec.claim
    assert rec.fact_type.value == "financial_metric"


def test_xbrl_to_fact_record_regional_includes_region_in_claim_and_id():
    f = _qtd_fact(
        numeric_value=4322476000.0,
        raw_value="4322476000",
        dimensions={
            "srt:ProductOrServiceAxis": "nflx:StreamingMember",
            "srt:StatementGeographicalAxis": "nflx:UnitedStatesAndCanadaMember",
        },
    )
    rec = xbrl_to_fact_record(
        f,
        source_document="nflx-10q-2024-q3",
        filing_date=date(2024, 10, 18),
    )
    assert rec is not None
    assert rec.fact_id.endswith("-2024Q3-UCAN")
    assert "UCAN" in rec.claim


def test_xbrl_to_fact_record_drops_ytd():
    f = _qtd_fact(
        period_kind="ytd_9m",
        period_start=date(2024, 1, 1),
        numeric_value=28754453000.0,
        raw_value="28754453000",
    )
    assert xbrl_to_fact_record(
        f,
        source_document="nflx-10q-2024-q3",
        filing_date=date(2024, 10, 18),
    ) is None


def test_xbrl_to_fact_record_drops_non_canonical():
    f = _qtd_fact(concept="us-gaap:NotARealConcept")
    assert xbrl_to_fact_record(
        f,
        source_document="nflx-10q-2024-q3",
        filing_date=date(2024, 10, 18),
    ) is None


def test_xbrl_to_fact_record_drops_dimensional_outside_policy():
    f = _qtd_fact(
        dimensions={
            "us-gaap:DebtInstrumentAxis": "nflx:SomeNoteMember",
        },
    )
    assert xbrl_to_fact_record(
        f,
        source_document="nflx-10q-2024-q3",
        filing_date=date(2024, 10, 18),
    ) is None

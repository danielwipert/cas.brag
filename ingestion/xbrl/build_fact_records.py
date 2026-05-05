"""Convert ``XBRLFact`` (raw parse) -> ``FactRecord`` (BRAG schema), applying
the Block 4 retention and dimension policy.

Decision recap (Block 4, 2026-05-04):
- Q1: Drop YTD facts (6M and 9M durations). Keep instants + QTD + FY.
- Q2: Keep aggregate + four geographic regions only (UCAN/EMEA/LATAM/APAC).
- Q4: ``verbatim_anchor`` = formatted display value (e.g. ``"$9,824,703"``)
  matching the rendered HTML table cell.

Note on value semantics: in the XBRL instance, the numeric text is the
canonical value in the unit specified by ``unitRef`` (e.g. dollars when
unit=USD). The ``decimals`` attribute is *precision* metadata, not a
scaling factor. For example, ``<us-gaap:Revenues unitRef="usd"
decimals="-3">9824703000</us-gaap:Revenues>`` means $9,824,703,000 reported
to thousands precision — NOT 9824703000 × 1000. This matches the XBRL 2.1
specification. (The build plan's note about multiplying by 10^|decimals|
is incorrect and would inflate every reported value by 1000x; we honor
the spec instead.)
"""
from __future__ import annotations

from datetime import date

from ingestion.xbrl.concept_filter import (
    classify_dimensions,
    human_label,
    is_canonical,
    statement_section,
)
from ingestion.xbrl.parse import XBRLFact
from schemas.enums import FactType
from schemas.records import FactRecord


# Period kinds we accept (dropped: ytd_6m, ytd_9m, unknown).
_RETAINED_PERIOD_KINDS: frozenset[str] = frozenset({"instant", "qtd", "fy"})

_MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


# ---------------------------------------------------------------------------
# Period formatting
# ---------------------------------------------------------------------------


def _quarter(d: date) -> int:
    return (d.month - 1) // 3 + 1


def _canonical_period(fact: XBRLFact) -> str | None:
    """Return the BRAG canonical period string for a fact's context."""
    if fact.period_kind == "instant" and fact.instant is not None:
        return fact.instant.isoformat()
    if fact.period_kind == "qtd" and fact.period_end is not None:
        return f"{fact.period_end.year}Q{_quarter(fact.period_end)}"
    if fact.period_kind == "fy" and fact.period_end is not None:
        return f"FY{fact.period_end.year}"
    return None


def _human_period(fact: XBRLFact) -> str:
    """Human-readable period rendering for inclusion in ``claim`` text."""
    if fact.period_kind == "instant" and fact.instant is not None:
        i = fact.instant
        return f"as of {_MONTH_NAMES[i.month - 1]} {i.day}, {i.year}"
    if fact.period_kind == "qtd" and fact.period_end is not None:
        return f"Q{_quarter(fact.period_end)} {fact.period_end.year}"
    if fact.period_kind == "fy" and fact.period_end is not None:
        return f"fiscal year {fact.period_end.year}"
    return ""


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------


def _displayed_scale(decimals: int | None) -> int:
    """Return the scale factor by which the canonical value is divided to
    produce the rendered HTML table cell. decimals=-3 means rendered in
    thousands; decimals=-6 in millions. decimals>=0 (or None) means rendered
    full-precision."""
    if decimals is None or decimals >= 0:
        return 1
    return 10 ** (-decimals)


_COUNT_UNITS: frozenset[str] = frozenset(
    {"shares", "pure", "membership", "segment"}
)


def _format_anchor(fact: XBRLFact) -> str:
    """Format the value as it appears in the rendered HTML table cell.

    Examples (for unit=USD, decimals=-3):
        9824703000      -> "$9,824,703"
        2321101000      -> "$2,321,101"
    For EPS (unit=USDPerShare, decimals=2):
        5.40            -> "$5.40"
    For shares / memberships (count-like units, decimals=-3):
        428000000       -> "428,000"
        84803000        -> "84,803"
    """
    value = fact.numeric_value
    unit = fact.unit
    if unit == "USD":
        displayed = value / _displayed_scale(fact.decimals)
        return f"${displayed:,.0f}"
    if unit == "USDPerShare":
        return f"${value:.2f}"
    if unit in _COUNT_UNITS:
        displayed = value / _displayed_scale(fact.decimals)
        if displayed == int(displayed):
            return f"{int(displayed):,}"
        return f"{displayed:,.2f}"
    return f"{value:,.0f} {unit}"


def _format_for_claim(fact: XBRLFact) -> str:
    """Analyst-friendly compact rendering for claim text. Independent of
    the rendered table-cell scale; uses absolute magnitude buckets.

    Examples:
        9.82B revenue   -> "$9.82 billion"
        2.32B opCF      -> "$2.32 billion"
        5.40 EPS        -> "$5.40 per share"
        428M shares     -> "428.0 million shares"
    """
    value = fact.numeric_value
    unit = fact.unit
    if unit == "USD":
        sign = "-" if value < 0 else ""
        v = abs(value)
        if v >= 1e9:
            return f"{sign}${v / 1e9:.2f} billion"
        if v >= 1e6:
            return f"{sign}${v / 1e6:.1f} million"
        if v >= 1e3:
            return f"{sign}${v / 1e3:,.0f} thousand"
        return f"{sign}${v:,.2f}"
    if unit == "USDPerShare":
        return f"${value:.2f} per share"
    if unit == "shares":
        v = abs(value)
        if v >= 1e6:
            return f"{value / 1e6:.1f} million shares"
        return f"{value:,.0f} shares"
    if unit in _COUNT_UNITS:
        # Memberships, segments, and other dimensionless whole-unit counts.
        if abs(value) >= 1e6:
            return f"{value / 1e6:.1f} million"
        if abs(value) >= 1e3:
            return f"{value:,.0f}"
        return f"{value:,.0f}"
    return f"{value:,.0f}"


# ---------------------------------------------------------------------------
# Fact ID
# ---------------------------------------------------------------------------


def _concept_slug(concept: str) -> str:
    return concept.replace(":", "-")


def _build_fact_id(*, document_id: str, concept: str, period: str, region: str | None) -> str:
    parts = ["F-XBRL", document_id, _concept_slug(concept), period]
    if region:
        parts.append(region)
    return "-".join(parts)


# ---------------------------------------------------------------------------
# Claim templating
# ---------------------------------------------------------------------------


def _is_balance_sheet_unit(fact: XBRLFact) -> bool:
    return fact.period_kind == "instant"


def _build_claim(
    *,
    label: str,
    region: str | None,
    fact: XBRLFact,
) -> str:
    period_phrase = _human_period(fact)
    value_phrase = _format_for_claim(fact)
    is_instant = _is_balance_sheet_unit(fact)
    region_prefix = f"{region} " if region else ""

    if is_instant:
        return (
            f"Netflix's {region_prefix}{label.lower()} {period_phrase} "
            f"was {value_phrase}."
        )
    return (
        f"Netflix's {region_prefix}{label.lower()} for {period_phrase} "
        f"was {value_phrase}."
    )


# ---------------------------------------------------------------------------
# Public conversion
# ---------------------------------------------------------------------------


def xbrl_to_fact_record(
    fact: XBRLFact,
    *,
    source_document: str,
    filing_date: date,
) -> FactRecord | None:
    """Convert one ``XBRLFact`` to a ``FactRecord``, applying the Block 4
    retention policy. Returns None if the fact is filtered out (non-canonical
    concept, dropped period kind, or out-of-policy dimensions)."""
    if not is_canonical(fact.concept):
        return None
    if fact.period_kind not in _RETAINED_PERIOD_KINDS:
        return None

    classification, region = classify_dimensions(fact.dimensions)
    if classification == "drop":
        return None

    period = _canonical_period(fact)
    if period is None:
        return None

    label = human_label(fact.concept)
    section = statement_section(fact.concept)
    anchor = _format_anchor(fact)
    claim = _build_claim(label=label, region=region, fact=fact)
    fact_id = _build_fact_id(
        document_id=source_document,
        concept=fact.concept,
        period=period,
        region=region,
    )

    return FactRecord(
        fact_id=fact_id,
        claim=claim,
        asserter="Netflix",
        source_document=source_document,
        source_section=section,
        verbatim_anchor=anchor,
        fact_type=FactType.financial_metric,
        period=period,
        value=fact.numeric_value,
        unit=fact.unit,
        concept_tag=fact.concept,
        assertion_date=filing_date,
        confidence=1.00,
    )

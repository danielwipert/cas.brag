"""Offline XBRL instance parser (build plan Block 4, decision Q3 = lxml-only).

Reads a Netflix XBRL instance document (e.g.,
``data/raw/nflx-10q-2024-q3.xbrl.xml``) and yields one ``XBRLFact`` per
numeric fact tag. Period classification (instant / QTD / 6M-YTD / 9M-YTD /
FY) is computed at parse time and surfaced on the fact record so downstream
filters (``build_fact_records``) can drop YTD facts cleanly.

We do not load the schema (.xsd) or any linkbases — labels and
statement-section assignments live in ``concept_filter.py`` instead. This
keeps ingestion fully offline and deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from lxml import etree


# XBRL standard namespace (the root <xbrl> element).
_NS_XBRLI = "http://www.xbrl.org/2003/instance"
_NS_XBRLDI = "http://xbrl.org/2006/xbrldi"
_NS_XLINK = "http://www.w3.org/1999/xlink"

# Element tags we handle directly. lxml stores tags as Clark notation:
# "{ns}localname".
_TAG_CONTEXT = f"{{{_NS_XBRLI}}}context"
_TAG_UNIT = f"{{{_NS_XBRLI}}}unit"
_TAG_PERIOD = f"{{{_NS_XBRLI}}}period"
_TAG_INSTANT = f"{{{_NS_XBRLI}}}instant"
_TAG_START = f"{{{_NS_XBRLI}}}startDate"
_TAG_END = f"{{{_NS_XBRLI}}}endDate"
_TAG_SEGMENT = f"{{{_NS_XBRLI}}}segment"
_TAG_ENTITY = f"{{{_NS_XBRLI}}}entity"
_TAG_MEASURE = f"{{{_NS_XBRLI}}}measure"
_TAG_DIVIDE = f"{{{_NS_XBRLI}}}divide"
_TAG_UNIT_NUM = f"{{{_NS_XBRLI}}}unitNumerator"
_TAG_UNIT_DEN = f"{{{_NS_XBRLI}}}unitDenominator"
_TAG_EXPLICIT_MEMBER = f"{{{_NS_XBRLDI}}}explicitMember"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class XBRLContext:
    context_id: str
    instant: date | None
    start_date: date | None
    end_date: date | None
    # axis_tag -> member_tag, e.g. {"srt:StatementGeographicalAxis":
    # "nflx:UnitedStatesAndCanadaMember"}
    dimensions: dict[str, str] = field(default_factory=dict)

    @property
    def is_instant(self) -> bool:
        return self.instant is not None

    @property
    def duration_days(self) -> int | None:
        if self.start_date is None or self.end_date is None:
            return None
        return (self.end_date - self.start_date).days + 1


@dataclass(frozen=True)
class XBRLFact:
    """One numeric XBRL fact, parsed from an instance element.

    ``period_kind`` is one of ``"instant"``, ``"qtd"``, ``"ytd_6m"``,
    ``"ytd_9m"``, ``"fy"``, or ``"unknown"`` — computed from the context.
    """

    concept: str  # e.g. "us-gaap:Revenues" (prefixed form, not Clark)
    fact_id_raw: str  # the <... id="f-30"> attribute, for traceability
    context_id: str
    unit: str  # e.g. "USD", "shares", "USDPerShare"
    decimals: int | None  # XBRL decimals attribute (e.g. -3 for thousands)
    raw_value: str  # original string in the XML
    numeric_value: float  # parsed
    period_kind: str
    period_end: date | None
    period_start: date | None  # None for instant
    instant: date | None  # None for duration
    dimensions: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class XBRLInstance:
    """Parsed XBRL instance document. Iterate ``facts`` to get every numeric
    fact; lookup ``contexts`` and ``units`` by id for cross-reference."""

    document_period_end: date | None
    cik: str | None
    contexts: dict[str, XBRLContext]
    units: dict[str, str]
    facts: list[XBRLFact]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _qname_to_prefixed(qname: str, nsmap: dict[str | None, str]) -> str:
    """Convert a Clark-notation tag ``{ns}local`` to ``prefix:local`` using
    the element's nsmap. If the namespace has no prefix, return ``local``."""
    if "}" not in qname:
        return qname
    ns, local = qname[1:].split("}", 1)
    for prefix, uri in nsmap.items():
        if uri == ns and prefix:
            return f"{prefix}:{local}"
    return local


def _parse_date(text: str | None) -> date | None:
    if not text:
        return None
    try:
        return date.fromisoformat(text.strip())
    except ValueError:
        return None


def _classify_period(context: XBRLContext) -> str:
    """QTD vs YTD-6M vs YTD-9M vs FY heuristic by duration length, with a
    tolerance to absorb day-boundary edges (e.g., 89 vs 92 vs 93 days)."""
    if context.is_instant:
        return "instant"
    days = context.duration_days
    if days is None:
        return "unknown"
    if 88 <= days <= 95:
        return "qtd"
    if 178 <= days <= 186:
        return "ytd_6m"
    if 268 <= days <= 275:
        return "ytd_9m"
    if 360 <= days <= 370:
        return "fy"
    return "unknown"


def _read_context(elem: etree._Element) -> XBRLContext:
    period = elem.find(_TAG_PERIOD)
    instant_node = period.find(_TAG_INSTANT) if period is not None else None
    start_node = period.find(_TAG_START) if period is not None else None
    end_node = period.find(_TAG_END) if period is not None else None

    dimensions: dict[str, str] = {}
    entity = elem.find(_TAG_ENTITY)
    segment = entity.find(_TAG_SEGMENT) if entity is not None else None
    if segment is not None:
        for child in segment:
            if child.tag == _TAG_EXPLICIT_MEMBER:
                axis = child.get("dimension", "").strip()
                member = (child.text or "").strip()
                if axis and member:
                    dimensions[axis] = member

    return XBRLContext(
        context_id=elem.get("id", ""),
        instant=_parse_date(instant_node.text) if instant_node is not None else None,
        start_date=_parse_date(start_node.text) if start_node is not None else None,
        end_date=_parse_date(end_node.text) if end_node is not None else None,
        dimensions=dimensions,
    )


def _read_unit(elem: etree._Element) -> str:
    """Return a canonical unit string: USD, shares, USDPerShare, or whatever
    measure(s) the unit declares."""
    direct_measures = elem.findall(_TAG_MEASURE)
    if direct_measures:
        return _measure_label(direct_measures[0].text)
    divide = elem.find(_TAG_DIVIDE)
    if divide is not None:
        num = divide.find(_TAG_UNIT_NUM)
        den = divide.find(_TAG_UNIT_DEN)
        num_m = num.find(_TAG_MEASURE) if num is not None else None
        den_m = den.find(_TAG_MEASURE) if den is not None else None
        if num_m is not None and den_m is not None:
            return f"{_measure_label(num_m.text)}Per{_measure_label(den_m.text)}"
    return "unknown"


def _measure_label(measure: str | None) -> str:
    if not measure:
        return "unknown"
    measure = measure.strip()
    if ":" in measure:
        _, local = measure.split(":", 1)
    else:
        local = measure
    # Common SEC measures: iso4217:USD, xbrli:shares, etc.
    if local.lower() == "usd":
        return "USD"
    if local.lower() == "shares":
        return "shares"
    return local


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def load_xbrl_instance(path: str | Path) -> XBRLInstance:
    """Parse an XBRL instance XML file. Returns the full parsed object — do
    not call this in a hot path; for one filing it's cheap."""
    path = Path(path)
    tree = etree.parse(str(path))
    root = tree.getroot()

    contexts: dict[str, XBRLContext] = {}
    units: dict[str, str] = {}
    facts: list[XBRLFact] = []
    document_period_end: date | None = None
    cik: str | None = None

    for child in root:
        if child.tag == _TAG_CONTEXT:
            ctx = _read_context(child)
            if ctx.context_id:
                contexts[ctx.context_id] = ctx
                # Pull CIK from the first context that has one.
                if cik is None:
                    entity = child.find(_TAG_ENTITY)
                    if entity is not None:
                        ident = entity.find(f"{{{_NS_XBRLI}}}identifier")
                        if ident is not None and ident.text:
                            cik = ident.text.strip()
        elif child.tag == _TAG_UNIT:
            uid = child.get("id", "")
            if uid:
                units[uid] = _read_unit(child)

    # Second pass: facts. Anything with a contextRef and not a context/unit.
    for child in root:
        if child.tag in (_TAG_CONTEXT, _TAG_UNIT):
            continue
        ctx_ref = child.get("contextRef")
        if not ctx_ref:
            continue
        concept = _qname_to_prefixed(child.tag, child.nsmap)
        unit_ref = child.get("unitRef") or ""
        decimals_raw = child.get("decimals")
        try:
            decimals = int(decimals_raw) if decimals_raw not in (None, "INF") else None
        except ValueError:
            decimals = None

        raw_value = (child.text or "").strip()
        if not raw_value:
            continue
        try:
            numeric_value = float(raw_value)
        except ValueError:
            # Non-numeric facts (TextBlock, dei strings) live in the same
            # element space; skip them — only the retention set matters.
            continue

        ctx = contexts.get(ctx_ref)
        if ctx is None:
            continue

        unit = units.get(unit_ref, "unknown") if unit_ref else "pure"
        period_kind = _classify_period(ctx)

        facts.append(
            XBRLFact(
                concept=concept,
                fact_id_raw=child.get("id", ""),
                context_id=ctx_ref,
                unit=unit,
                decimals=decimals,
                raw_value=raw_value,
                numeric_value=numeric_value,
                period_kind=period_kind,
                period_end=ctx.end_date if not ctx.is_instant else ctx.instant,
                period_start=ctx.start_date,
                instant=ctx.instant,
                dimensions=dict(ctx.dimensions),
            )
        )

        # Surface DocumentPeriodEndDate for downstream metadata. Stored as
        # a date, so we parse the value if the concept matches.
        if concept == "dei:DocumentPeriodEndDate" and document_period_end is None:
            document_period_end = _parse_date(raw_value)

    # The dei:DocumentPeriodEndDate fact is non-numeric, so the loop above
    # won't capture it. Pull it directly from any element with that tag.
    if document_period_end is None:
        for elem in root.iter():
            tag = _qname_to_prefixed(elem.tag, elem.nsmap)
            if tag == "dei:DocumentPeriodEndDate":
                document_period_end = _parse_date(elem.text or "")
                break

    return XBRLInstance(
        document_period_end=document_period_end,
        cik=cik,
        contexts=contexts,
        units=units,
        facts=facts,
    )


def iter_facts(instance: XBRLInstance):
    """Trivial iterator over an instance's facts. Provided for symmetry with
    the build plan's API; equivalent to ``iter(instance.facts)``."""
    yield from instance.facts

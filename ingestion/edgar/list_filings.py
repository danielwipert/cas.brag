"""SEC EDGAR filing enumeration for Netflix (Block 6a).

Lists every Netflix 10-K, 10-Q, and 8-K with Item 2.02 (earnings-release
shareholder letter) filed within a date window. This is the SEC half of
Block 6's document-manifest step.

Each returned dict is the canonical manifest entry shape:

    {
        "document_id":        "nflx-10q-2024-q3",
        "source":             "edgar",
        "form":               "10-Q" | "10-K" | "8-K",
        "document_kind":      None | "letter",
        "accession":          "0001065280-24-000093",
        "filing_date":        "2024-10-18",
        "period_of_report":   "2024-09-30",
        "fiscal_period":      "2024Q3" | "FY2024",
    }
"""
from __future__ import annotations

from datetime import date

import edgar

from ingestion.edgar.fetch import NETFLIX_CIK, _ensure_configured
from ingestion.normalize import _quarter_for_date, assign_document_id


# Netflix releases earnings in mid-Jan (Q4+FY), mid-Apr (Q1), mid-Jul (Q2),
# mid-Oct (Q3). Map an 8-K Item 2.02 period_of_report month to the
# fiscal quarter being announced and the year offset relative to that date.
_8K_MONTH_TO_QUARTER: dict[int, tuple[int, int]] = {
    1: (4, -1),  # Jan release covers prior-year Q4
    2: (4, -1),  # very rare slip into Feb
    4: (1, 0),
    7: (2, 0),
    10: (3, 0),
}


def _8k_announces_period(period_of_report: date) -> tuple[int, int] | None:
    """Map an 8-K Item 2.02 period_of_report to (fiscal_year, quarter).
    Returns None if the date falls in a month Netflix doesn't release in."""
    info = _8K_MONTH_TO_QUARTER.get(period_of_report.month)
    if info is None:
        return None
    qtr, year_offset = info
    return period_of_report.year + year_offset, qtr


def _has_item_202(filing: edgar.Filing) -> bool:
    """Test for Item 2.02 disclosure. ``Filing.items`` is a comma-separated
    string ('2.02,9.01') or a list — accept both shapes."""
    raw = getattr(filing, "items", None)
    if raw is None:
        return False
    if isinstance(raw, str):
        return "2.02" in raw
    try:
        return any("2.02" in str(item) for item in raw)
    except TypeError:
        return False


def list_netflix_filings(
    *,
    filed_on_or_after: str = "2016-05-06",
    filed_on_or_before: str = "2026-05-06",
) -> list[dict]:
    """Enumerate Netflix 10-K, 10-Q, and 8-K-Item-2.02 filings within the
    inclusive date window. Amended filings (10-K/A, 10-Q/A) supersede the
    originals — only the most recently filed entry per (form_base, period)
    is retained, matching ``find_netflix_filing``'s behavior."""
    _ensure_configured()
    company = edgar.Company(NETFLIX_CIK)
    out: list[dict] = []

    # ``get_filings(form="10-K")`` also returns 10-K/A amendments. Collect
    # everything per (form_base, period_of_report). Selection rule:
    #   1. Prefer original filings over amendments (/A) — amendments are
    #      almost always partial (Item 11 exec-comp corrections, etc.) and
    #      contain only the amended item, not the full document text. The
    #      original 10-K/10-Q is the canonical narrative for BRAG ingest.
    #   2. Within the same form-class (both original or both amendment),
    #      keep the most recently filed entry.
    periodic: dict[tuple[str, str], dict] = {}
    for form_base in ("10-K", "10-Q"):
        for f in company.get_filings(form=form_base):
            fd = str(f.filing_date)
            if fd < filed_on_or_after or fd > filed_on_or_before:
                continue
            por_str = str(f.period_of_report)
            try:
                por_date = date.fromisoformat(por_str)
            except ValueError:
                continue
            doc_id = assign_document_id(form=form_base, period=por_date)
            qtr = _quarter_for_date(por_date)
            fiscal_period = (
                f"{por_date.year}Q{qtr}"
                if form_base == "10-Q"
                else f"FY{por_date.year}"
            )
            entry = {
                "document_id": doc_id,
                "source": "edgar",
                "form": str(f.form),  # keeps "10-K/A" when applicable
                "document_kind": None,
                "accession": f.accession_number,
                "filing_date": fd,
                "period_of_report": por_str,
                "fiscal_period": fiscal_period,
            }
            key = (form_base, por_str)
            prior = periodic.get(key)
            if prior is None:
                periodic[key] = entry
                continue
            prior_amend = prior["form"].endswith("/A")
            this_amend = entry["form"].endswith("/A")
            if prior_amend and not this_amend:
                periodic[key] = entry  # original wins over the amendment we had
            elif prior_amend == this_amend and fd > prior["filing_date"]:
                periodic[key] = entry  # same class, latest filing_date wins
    out.extend(periodic.values())

    # 8-K letter entries can collide on document_id when multiple Item 2.02
    # filings land in the same earnings month (e.g. Jan 2019 had a Jan-3
    # 8-K plus the Jan-17 actual earnings release — both fall in fiscal
    # period 2018Q4). Mirror the periodic-form rule: keep the latest
    # filing_date per (document_id). letter_unmapped entries get a
    # per-accession document_id and never collide, so they pass through.
    letters: dict[str, dict] = {}
    for f in company.get_filings(form="8-K"):
        fd = str(f.filing_date)
        if fd < filed_on_or_after or fd > filed_on_or_before:
            continue
        if not _has_item_202(f):
            continue
        try:
            por_date = date.fromisoformat(str(f.period_of_report))
        except (ValueError, TypeError):
            por_date = date.fromisoformat(fd)
        announced = _8k_announces_period(por_date)
        if announced is None:
            out.append(
                {
                    "document_id": f"nflx-8k-{f.accession_number}",
                    "source": "edgar",
                    "form": "8-K",
                    "document_kind": "letter_unmapped",
                    "accession": f.accession_number,
                    "filing_date": fd,
                    "period_of_report": str(f.period_of_report),
                    "fiscal_period": None,
                }
            )
            continue
        year, qtr = announced
        doc_id = assign_document_id(
            form="8-K", period=f"{year}Q{qtr}", document_kind="letter"
        )
        entry = {
            "document_id": doc_id,
            "source": "edgar",
            "form": "8-K",
            "document_kind": "letter",
            "accession": f.accession_number,
            "filing_date": fd,
            "period_of_report": str(f.period_of_report),
            "fiscal_period": f"{year}Q{qtr}",
        }
        prior = letters.get(doc_id)
        if prior is None or fd > prior["filing_date"]:
            letters[doc_id] = entry
    out.extend(letters.values())

    out.sort(key=lambda d: (d["filing_date"], d["form"], d["document_id"]))
    return out

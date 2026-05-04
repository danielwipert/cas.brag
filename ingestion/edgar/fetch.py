"""SEC EDGAR document acquisition (Stage I1, build plan Block 2).

Wraps edgartools to provide BRAG-shaped helpers:
    - configure() — sets the SEC-required User-Agent.
    - find_netflix_filing(form, period) — look up a specific Netflix filing.
    - fetch_filing(cik, accession_number) — direct retrieval by accession.
    - extract_sections(filing) — return {section_name -> raw_text} per spec §2.3.
    - get_xbrl_instance(filing) — return the XBRL object (or None).
    - extract_exhibit_991(filing) — pull the shareholder letter body from an 8-K.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import edgar

NETFLIX_CIK = "0001065280"

# Spec D25 enumerates retained sections in 10-K vocabulary (Item 1, 1A, 7, 7A).
# 10-Q uses different item numbers for the same content; the mapping is fixed:
#   10-K Item 1   (Business)              -> not present in 10-Q
#   10-K Item 1A  (Risk Factors)          -> 10-Q Part II Item 1A (only when updated)
#   10-K Item 7   (MD&A)                  -> 10-Q Part I  Item 2
#   10-K Item 7A  (Quant/Qual disclosure) -> 10-Q Part I  Item 3
# Item 1 in 10-Q (Financial Statements) is XBRL territory; we extract only the
# narrative footnotes from it via marker search in _extract_footnote_narrative.
TENK_RETAINED_ITEMS = ("Item 1", "Item 1A", "Item 7", "Item 7A")
TENQ_RETAINED_ITEMS = ("Item 2", "Item 3", "Item 1A")

_CONFIGURED = False


def configure(user_agent: str | None = None) -> str:
    """Set the SEC-required User-Agent. Idempotent.

    Resolution order: explicit arg > EDGAR_USER_AGENT env > EDGAR_IDENTITY env >
    a development fallback derived from the git committer email.
    """
    global _CONFIGURED
    if user_agent is None:
        user_agent = (
            os.environ.get("EDGAR_USER_AGENT")
            or os.environ.get("EDGAR_IDENTITY")
            or "cas.brag wipertds@gmail.com"
        )
    edgar.set_identity(user_agent)
    _CONFIGURED = True
    return user_agent


def _ensure_configured() -> None:
    if not _CONFIGURED:
        configure()


def find_netflix_filing(
    *,
    form: str,
    period_of_report: str,
) -> edgar.Filing:
    """Find a single Netflix filing by form + period_of_report (YYYY-MM-DD).

    Raises LookupError if no matching filing exists.
    """
    _ensure_configured()
    company = edgar.Company(NETFLIX_CIK)
    filings = company.get_filings(form=form)
    matches = [f for f in filings if str(f.period_of_report) == period_of_report]
    if not matches:
        raise LookupError(
            f"no {form} filing for Netflix with period_of_report={period_of_report}"
        )
    if len(matches) > 1:
        # Prefer the most recently filed (handles amended filings).
        matches.sort(key=lambda f: f.filing_date, reverse=True)
    return matches[0]


def find_netflix_8k_letter(*, filed_on_or_after: str, filed_on_or_before: str) -> edgar.Filing:
    """Find Netflix's 8-K shareholder letter (Item 2.02) within a date range.

    The letter body lives in Exhibit 99.1 of an 8-K announcing earnings results.
    Both dates are inclusive, ISO format (YYYY-MM-DD).
    """
    _ensure_configured()
    company = edgar.Company(NETFLIX_CIK)
    filings = company.get_filings(form="8-K")

    candidates: list[edgar.Filing] = []
    for f in filings:
        fd = str(f.filing_date)
        if fd < filed_on_or_after or fd > filed_on_or_before:
            continue
        # Item 2.02 announcements have the press release / shareholder letter as 99.1.
        # Heuristic: look for Item 2.02 in the filing's items list.
        items = []
        try:
            items = list(getattr(f, "items", []) or [])
        except Exception:
            items = []
        if any("2.02" in str(item) for item in items):
            candidates.append(f)

    if not candidates:
        # Fallback: any 8-K in the window. Some filings may not surface items
        # cleanly via the property; we'll pick the most recent and confirm via
        # exhibits at extraction time.
        candidates = [
            f for f in filings
            if filed_on_or_after <= str(f.filing_date) <= filed_on_or_before
        ]
    if not candidates:
        raise LookupError(
            f"no Netflix 8-K filed between {filed_on_or_after} and {filed_on_or_before}"
        )
    candidates.sort(key=lambda f: f.filing_date, reverse=True)
    return candidates[0]


def fetch_filing(cik: str, accession_number: str) -> edgar.Filing:
    _ensure_configured()
    return edgar.get_by_accession_number(accession_number)


def get_xbrl_instance(filing: edgar.Filing):
    """Return the XBRL data object for a filing, or None if not present."""
    try:
        return filing.xbrl()
    except Exception:
        return None


def save_primary_html(filing: edgar.Filing, dest: Path) -> Path:
    """Save the filing's primary document HTML to disk, return the path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    html = filing.html()
    if html is None:
        raise ValueError(f"filing {filing.accession_number} has no HTML primary doc")
    dest.write_text(html, encoding="utf-8")
    return dest


def save_xbrl(filing: edgar.Filing, dest: Path) -> Path | None:
    """If the filing has an XBRL instance, save the raw XBRL XML to disk."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    # The simplest reliable XBRL bytes: find the .xml attachment matching the instance.
    try:
        for att in filing.attachments:
            name = (getattr(att, "document", "") or "").lower()
            if name.endswith("_htm.xml") or name.endswith(".xml") and "xbrl" in name:
                content = att.content
                if isinstance(content, bytes):
                    dest.write_bytes(content)
                else:
                    dest.write_text(str(content), encoding="utf-8")
                return dest
    except Exception:
        pass
    return None


def extract_sections(filing: edgar.Filing) -> dict[str, str]:
    """Return {section_name: raw_text} for the form-specific retained sections.

    For 10-K and 10-Q this uses edgartools' typed report objects (TenK/TenQ).
    Each filing exposes a Sections mapping; we read `sections[item].text`.
    Items not present in the filing (e.g., 10-Q without a Risk Factors update)
    are silently skipped.

    For 8-K shareholder letters, use `extract_exhibit_991` instead — letters
    aren't item-shaped.
    """
    form = filing.form.upper()
    obj = filing.obj()

    if form == "10-K":
        wanted = TENK_RETAINED_ITEMS
    elif form == "10-Q":
        wanted = TENQ_RETAINED_ITEMS
    else:
        raise ValueError(
            f"extract_sections only supports 10-K/10-Q; got {form}. "
            "Use extract_exhibit_991 for 8-K shareholder letters."
        )

    sections: dict[str, str] = {}
    sections_map = getattr(obj, "sections", None)
    available_keys: set[str] = set()
    if sections_map is not None and hasattr(sections_map, "keys"):
        available_keys = set(sections_map.keys())

    for item in wanted:
        if item not in available_keys:
            continue
        try:
            section = sections_map[item]
            text_attr = getattr(section, "text", None)
            text = text_attr() if callable(text_attr) else text_attr
        except Exception:
            text = None
        if text and len(str(text).strip()) > 0:
            sections[item] = str(text)

    # Footnote narrative — search the full document text for the standard
    # "Notes to ... Financial Statements" marker. Block 3's chunker is
    # responsible for further filtering tabular content out of this block.
    try:
        notes_text = _extract_footnote_narrative(filing)
        if notes_text:
            sections["Notes to Financial Statements"] = notes_text
    except Exception:
        pass

    return sections


def _extract_footnote_narrative(filing: edgar.Filing) -> str:
    """Best-effort: pull the document text after the financial statements
    so footnote narrative is captured. Block 3's chunker handles the section
    boundary; here we just return what's available.
    """
    try:
        text = filing.text() or ""
    except Exception:
        text = ""
    if not text:
        return ""
    markers = [
        "Notes to Condensed Consolidated Financial Statements",
        "Notes to Consolidated Financial Statements",
        "NOTES TO CONDENSED CONSOLIDATED FINANCIAL STATEMENTS",
        "NOTES TO CONSOLIDATED FINANCIAL STATEMENTS",
    ]
    for m in markers:
        idx = text.find(m)
        if idx >= 0:
            return text[idx:]
    return ""


# Markers identifying the start of the financial-statement appendix at the
# bottom of every Netflix shareholder letter. The text-rendered tables below
# these markers are mangled (column truncation), but the values they contain
# are sourced from XBRL anyway (Block 4), so we drop the appendix from the
# Chunk Store and keep only the analytical narrative.
_LETTER_APPENDIX_MARKERS = (
    "Consolidated Statements of Operations",
    "Consolidated Balance Sheet",
    "Consolidated Statements of Cash Flows",
    "Reconciliation of Free Cash Flow",
    "Reconciliation of Non-GAAP",
)


def _trim_letter_appendix(text: str) -> str:
    """Cut the text at the first appendix-table marker, then strip trailing
    standalone header lines (page numbers, "Netflix, Inc.") that sit between
    the prose and the appendix."""
    earliest = len(text)
    for marker in _LETTER_APPENDIX_MARKERS:
        idx = text.find(marker)
        if 0 <= idx < earliest:
            earliest = idx
    if earliest >= len(text):
        return text

    body = text[:earliest]
    # Walk back through trailing lines, stripping orphan header/page markers.
    lines = body.splitlines()
    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        # Page numbers (e.g., "12"), the company header, or all-cap section breaks.
        if last.isdigit() or last in ("Netflix, Inc.", "Netflix, Inc"):
            lines.pop()
            continue
        break
    return "\n".join(lines).rstrip()


def extract_exhibit_991(filing: edgar.Filing) -> str:
    """Pull the shareholder letter body from an 8-K. Returns the letter text.

    Strategy:
        1. Find the attachment whose document name contains 'ex99' or 'ex-99'
           or whose description mentions the shareholder letter.
        2. Return its rendered text content.
    """
    if filing.form.upper().lstrip("/") not in ("8-K", "8-K/A"):
        raise ValueError(f"extract_exhibit_991 expects 8-K, got {filing.form}")

    candidates: list[Any] = []
    try:
        atts = list(filing.attachments)
    except Exception:
        atts = []

    for att in atts:
        name = str(getattr(att, "document", "") or "").lower()
        desc = str(getattr(att, "description", "") or "").lower()
        if "ex99" in name or "ex-99" in name or "99.1" in name or "99_1" in name:
            candidates.append(att)
        elif "shareholder" in desc or "letter" in desc or "press release" in desc:
            candidates.append(att)

    if not candidates:
        raise LookupError(
            f"no Exhibit 99.1 attachment found on 8-K {filing.accession_number}"
        )

    # Prefer the first candidate matching ex99-prefixed names.
    candidates.sort(
        key=lambda a: 0 if "ex99" in str(getattr(a, "document", "")).lower() else 1
    )
    att = candidates[0]

    # Attachments often expose .text() / .markdown() / .html() / .content
    for method in ("text", "markdown"):
        fn = getattr(att, method, None)
        if callable(fn):
            try:
                out = fn()
                if out:
                    return _trim_letter_appendix(str(out))
            except Exception:
                continue

    raw = getattr(att, "content", None)
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            text = raw.decode("latin-1", errors="replace")
        return _trim_letter_appendix(text)
    if raw:
        return _trim_letter_appendix(str(raw))

    raise ValueError(f"unable to extract text from Exhibit 99.1 on {filing.accession_number}")

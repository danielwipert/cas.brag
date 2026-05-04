"""Text normalization and document_id assignment (Stage I1, build plan Block 2)."""
from __future__ import annotations

import re
import unicodedata
from datetime import date


_WS_RE = re.compile(r"[ \t ]+")
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n[ \t]*\n+")
_NULL_BYTE_RE = re.compile(r"[\x00\x0b\x0c]")
_HTML_ARTIFACT_RE = re.compile(r"&(nbsp|amp|lt|gt|quot|apos|#160);", re.IGNORECASE)
_HTML_ENTITY_MAP = {
    "&nbsp;": " ",
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&apos;": "'",
    "&#160;": " ",
}


def normalize_text(text: str) -> str:
    """Strip HTML artifacts, NFC-normalize, standardize whitespace and quotes.

    Idempotent — calling twice yields the same result as calling once.
    """
    if not text:
        return ""

    s = unicodedata.normalize("NFC", text)

    # Smart quotes -> straight (preserves searchability of GAAP-style strings).
    s = (
        s.replace("‘", "'")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")  # en dash
        .replace("—", "-")  # em dash
        .replace("…", "...")  # ellipsis
    )

    # Common HTML entity escapes that survived an imperfect HTML-to-text pass.
    for k, v in _HTML_ENTITY_MAP.items():
        s = s.replace(k, v)
        s = s.replace(k.lower(), v)
        s = s.replace(k.upper(), v)

    # Drop control chars except newline and tab.
    s = _NULL_BYTE_RE.sub("", s)

    # Normalize CRLF / CR to LF.
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse runs of horizontal whitespace.
    s = _WS_RE.sub(" ", s)

    # Trim trailing whitespace on each line.
    s = "\n".join(line.rstrip() for line in s.split("\n"))

    # Collapse 3+ consecutive blank lines to 2.
    s = _BLANK_LINE_RE.sub("\n\n", s)

    return s.strip()


def _quarter_for_date(d: date) -> int:
    return (d.month - 1) // 3 + 1


def assign_document_id(
    *,
    form: str,
    period: date | str,
    document_kind: str | None = None,
) -> str:
    """Return the canonical BRAG document_id.

    Pattern: nflx-{type}-{period_token}-{form_marker}, simplified to one of:
        nflx-10k-{YYYY}                   for 10-K
        nflx-10q-{YYYY}-q{N}              for 10-Q
        nflx-q{N}-{YYYY}-letter           for 8-K Exhibit 99.1 (shareholder letter)
        nflx-q{N}-{YYYY}-transcript       for q4cdn earnings transcripts

    Examples used in build plan:
        nflx-10q-2024-q3, nflx-q4-2023-letter, nflx-q1-2024-transcript
    """
    if isinstance(period, str):
        # Accept canonical period strings. Parsing is loose here on purpose —
        # callers should already be passing a date or a known canonical form.
        if period.startswith("FY"):
            year = int(period[2:6])
            qtr = None
        elif "Q" in period:
            year, qtr = period.split("Q")
            year, qtr = int(year), int(qtr)
        else:
            d = date.fromisoformat(period)
            year, qtr = d.year, _quarter_for_date(d)
    else:
        year, qtr = period.year, _quarter_for_date(period)

    form_lower = form.lower().replace("/a", "")  # 10-Q/A normalizes to 10-q

    if form_lower in ("10-k", "10k"):
        return f"nflx-10k-{year}"
    if form_lower in ("10-q", "10q"):
        if qtr is None:
            raise ValueError("10-Q document_id requires a quarter")
        return f"nflx-10q-{year}-q{qtr}"
    if form_lower in ("8-k", "8k"):
        if document_kind == "letter":
            if qtr is None:
                raise ValueError("8-K letter document_id requires a quarter")
            return f"nflx-q{qtr}-{year}-letter"
        # Generic 8-K (not a shareholder letter) — uncommon for our corpus.
        return f"nflx-8k-{year}-q{qtr}" if qtr else f"nflx-8k-{year}"
    if form_lower in ("transcript", "earnings_transcript"):
        if qtr is None:
            raise ValueError("transcript document_id requires a quarter")
        return f"nflx-q{qtr}-{year}-transcript"

    raise ValueError(f"unsupported form/document_kind combination: form={form!r}, kind={document_kind!r}")

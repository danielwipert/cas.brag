"""Netflix earnings transcript acquisition from s22.q4cdn.com (build plan Block 2).

Netflix publishes pre-recorded earnings interview transcripts as PDFs on its
investor relations document host. URL pattern:

    https://s22.q4cdn.com/959853165/files/doc_financials/{year}/{quarter}/{filename}.pdf

The filename component is not fully predictable (Netflix has used several
naming conventions over the years), so this module attempts a list of known
candidates and falls back to discovery via HEAD probe.
"""
from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

import requests
from pypdf import PdfReader

Q4CDN_BASE = "https://s22.q4cdn.com/959853165/files/doc_financials"

# Candidate filename templates Netflix has used. Probed in order; first 200 wins.
_FILENAME_PATTERNS: tuple[str, ...] = (
    "FINAL-Q{q}-{yy}-Earnings-Interview-Transcript.pdf",
    "FINAL-Q{q}-{yyyy}-Earnings-Interview-Transcript.pdf",
    "Q{q}-{yy}-Earnings-Interview-Transcript.pdf",
    "Q{q}-{yyyy}-Earnings-Interview-Transcript.pdf",
    "Q{q}{yy}-Earnings-Interview-Transcript.pdf",
    "{yyyy}-Q{q}-Earnings-Interview-Transcript.pdf",
    "Earnings-Interview-Transcript-Q{q}-{yyyy}.pdf",
    "Final-Earnings-Interview-Q{q}-{yyyy}.pdf",
    "NFLX-Q{q}-{yyyy}-Earnings-Interview.pdf",
)

_USER_AGENT = "cas.brag/0.1 (Netflix transcript ingest; +https://github.com/danielwipert/cas.brag)"
_TIMEOUT = 30


def _candidate_urls(year: int, quarter: int) -> list[str]:
    yy = str(year)[-2:]
    yyyy = str(year)
    urls: list[str] = []
    for tmpl in _FILENAME_PATTERNS:
        fname = tmpl.format(q=quarter, yy=yy, yyyy=yyyy)
        urls.append(f"{Q4CDN_BASE}/{year}/q{quarter}/{fname}")
    return urls


def discover_transcript_url(year: int, quarter: int) -> str:
    """Return the first candidate URL that responds 200 to a HEAD request."""
    headers = {"User-Agent": _USER_AGENT}
    last_error: Exception | None = None
    for url in _candidate_urls(year, quarter):
        try:
            r = requests.head(url, headers=headers, timeout=_TIMEOUT, allow_redirects=True)
            if r.status_code == 200:
                return url
        except requests.RequestException as e:
            last_error = e
    raise LookupError(
        f"no q4cdn transcript candidate URL responded 200 for {year}Q{quarter}; "
        f"tried {len(_FILENAME_PATTERNS)} patterns. Last error: {last_error!r}"
    )


def fetch_transcript(year: int, quarter: int, *, url: str | None = None) -> bytes:
    """Download a Netflix earnings transcript PDF. Returns the raw PDF bytes."""
    if url is None:
        url = discover_transcript_url(year, quarter)
    headers = {"User-Agent": _USER_AGENT}
    r = requests.get(url, headers=headers, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.content


def save_transcript(pdf_bytes: bytes, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(pdf_bytes)
    return dest


# Speakers commonly present on Netflix transcripts. Pattern matching is loose
# because some transcripts use different cap conventions.
_KNOWN_SPEAKERS = (
    "Ted Sarandos",
    "Spencer Neumann",
    "Greg Peters",
    "Reed Hastings",
    "David Wells",
    "Wilmot Reed Hastings",
)
_SPEAKER_RE = re.compile(
    r"(?P<name>" + "|".join(re.escape(s) for s in _KNOWN_SPEAKERS) + r")\s*[:\-—]",
    re.IGNORECASE,
)


def extract_transcript_text(pdf_bytes: bytes) -> str:
    """Extract text from a Netflix transcript PDF, preserving speaker turns.

    Conventions:
        - Each speaker turn is separated by a blank line.
        - Speaker names are normalized to "Name:" prefix on their own line.
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    raw = "\n".join(pages)

    # Insert blank lines before speaker labels so chunkers can detect turns.
    # Use a positive lookbehind-friendly transformation: prepend "\n\n" before
    # any known speaker name occurrence that isn't already at line start with
    # blank line above.
    def _insert_blank(match: re.Match[str]) -> str:
        return "\n\n" + match.group(0).strip() + " "

    out = _SPEAKER_RE.sub(_insert_blank, raw)
    # Collapse any 3+ newlines back to exactly 2.
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()

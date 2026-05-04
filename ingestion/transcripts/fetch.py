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


# Speakers commonly present on Netflix transcripts. q4cdn-hosted earnings
# transcripts are S&P Global Market Intelligence typesets and use formal
# names as speaker labels ("Theodore A. Sarandos") rather than the casual
# forms used in shareholder letters ("Ted Sarandos"). We canonicalize to
# the casual form on extraction so the asserter field is consistent across
# transcript-sourced and letter-sourced facts (spec §2.5 examples use the
# casual form).
_FORMAL_TO_CASUAL: dict[str, str] = {
    "Theodore A. Sarandos": "Ted Sarandos",
    "Theodore Anthony Sarandos": "Ted Sarandos",
    "Spencer Adam Neumann": "Spencer Neumann",
    "Spence Neumann": "Spencer Neumann",
    "Gregory K. Peters": "Greg Peters",
    "Wilmot Reed Hastings": "Reed Hastings",
    "Wilmot Reed Hastings Jr": "Reed Hastings",
    "David B. Wells": "David Wells",
}

_CASUAL_NAMES: tuple[str, ...] = (
    "Ted Sarandos",
    "Spencer Neumann",
    "Greg Peters",
    "Reed Hastings",
    "David Wells",
    "Spencer Wang",  # Netflix VP of IR — moderates the Q&A
)

# Match formal forms first (longer strings), then casual forms.
_ALL_NAMES = tuple(_FORMAL_TO_CASUAL.keys()) + _CASUAL_NAMES
_SPEAKER_RE = re.compile(
    r"(?P<name>" + "|".join(re.escape(s) for s in _ALL_NAMES) + r")",
)


def _canonicalize_name(name: str) -> str:
    return _FORMAL_TO_CASUAL.get(name, name)


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

    # Replace formal-form speaker labels with canonical "Casual Name:" prefix,
    # each on its own line with a blank line above so the chunker can detect
    # speaker turns and the LLM extractor can attribute facts to the right
    # asserter.
    def _label(match: re.Match[str]) -> str:
        return f"\n\n{_canonicalize_name(match.group('name'))}:\n"

    out = _SPEAKER_RE.sub(_label, raw)
    # Collapse any 3+ newlines back to exactly 2.
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()

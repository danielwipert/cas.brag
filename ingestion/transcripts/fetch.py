"""Netflix earnings transcript acquisition from s22.q4cdn.com (build plan Block 2).

Netflix publishes pre-recorded earnings interview transcripts as PDFs on its
investor relations document host. Two URL layouts have been observed:

    /files/doc_financials/{year}/q{N}/...        (older — pre-~2024)
    /files/doc_events/{YYYY}/{Mon}/{DD}/...      (current — 2024+)

The current layout's filename and path both embed the actual earnings-
release date, which we don't know without external info. Pass
``release_date`` (e.g. the matching 8-K Item 2.02 filing date) to
``discover_transcript_url`` to enable the dated patterns.
"""
from __future__ import annotations

import re
from datetime import date
from io import BytesIO
from pathlib import Path

import requests
from pypdf import PdfReader

# Date-free URL templates (older layout). Probed when no release_date is known.
_URL_TEMPLATES_DATEFREE: tuple[str, ...] = (
    "https://s22.q4cdn.com/959853165/files/doc_financials/{year}/q{q}/FINAL-Q{q}-{yy}-Earnings-Interview-Transcript.pdf",
    "https://s22.q4cdn.com/959853165/files/doc_financials/{year}/q{q}/FINAL-Q{q}-{yyyy}-Earnings-Interview-Transcript.pdf",
    "https://s22.q4cdn.com/959853165/files/doc_financials/{year}/q{q}/Q{q}-{yy}-Earnings-Interview-Transcript.pdf",
    "https://s22.q4cdn.com/959853165/files/doc_financials/{year}/q{q}/Q{q}-{yyyy}-Earnings-Interview-Transcript.pdf",
    "https://s22.q4cdn.com/959853165/files/doc_financials/{year}/q{q}/Q{q}{yy}-Earnings-Interview-Transcript.pdf",
    "https://s22.q4cdn.com/959853165/files/doc_financials/{year}/q{q}/{yyyy}-Q{q}-Earnings-Interview-Transcript.pdf",
    "https://s22.q4cdn.com/959853165/files/doc_financials/{year}/q{q}/Earnings-Interview-Transcript-Q{q}-{yyyy}.pdf",
    "https://s22.q4cdn.com/959853165/files/doc_financials/{year}/q{q}/Final-Earnings-Interview-Q{q}-{yyyy}.pdf",
    "https://s22.q4cdn.com/959853165/files/doc_financials/{year}/q{q}/NFLX-Q{q}-{yyyy}-Earnings-Interview.pdf",
)

# Dated URL templates (current layout). Probed first when release_date is provided.
# Available context vars: {year}, {q}, {yy}, {yyyy}, {date}, {mon}, {dd}.
_URL_TEMPLATES_DATED: tuple[str, ...] = (
    "https://s22.q4cdn.com/959853165/files/doc_events/{yyyy}/{mon}/{dd}/netflix-inc-usa_earnings-call_{date}_english-1.pdf",
    "https://s22.q4cdn.com/959853165/files/doc_events/{yyyy}/{mon}/{dd}/netflix-inc-usa_earnings-call_{date}_english.pdf",
)

_USER_AGENT = "cas.brag/0.1 (Netflix transcript ingest; +https://github.com/danielwipert/cas.brag)"
_TIMEOUT = 30


def _candidate_urls(
    year: int, quarter: int, release_date: date | None = None
) -> list[str]:
    yy = str(year)[-2:]
    yyyy = str(year)
    urls: list[str] = []
    if release_date is not None:
        ctx = {
            "year": year,
            "q": quarter,
            "yy": yy,
            "yyyy": yyyy,
            "date": release_date.isoformat(),
            "mon": release_date.strftime("%b"),
            "dd": f"{release_date.day:02d}",
        }
        for tmpl in _URL_TEMPLATES_DATED:
            urls.append(tmpl.format(**ctx))
    for tmpl in _URL_TEMPLATES_DATEFREE:
        urls.append(tmpl.format(year=year, q=quarter, yy=yy, yyyy=yyyy))
    return urls


def discover_transcript_url(
    year: int, quarter: int, *, release_date: date | None = None
) -> str:
    """Return the first candidate URL that responds 200 to a HEAD request.

    When ``release_date`` is supplied (typically the 8-K Item 2.02 filing
    date for the corresponding earnings release), the current
    ``/doc_events/{YYYY}/{Mon}/{DD}/`` layout is probed first."""
    headers = {"User-Agent": _USER_AGENT}
    last_error: Exception | None = None
    candidates = _candidate_urls(year, quarter, release_date=release_date)
    for url in candidates:
        try:
            r = requests.head(url, headers=headers, timeout=_TIMEOUT, allow_redirects=True)
            if r.status_code == 200:
                return url
        except requests.RequestException as e:
            last_error = e
    raise LookupError(
        f"no q4cdn transcript candidate URL responded 200 for {year}Q{quarter}; "
        f"tried {len(candidates)} patterns. Last error: {last_error!r}"
    )


def fetch_transcript(
    year: int,
    quarter: int,
    *,
    url: str | None = None,
    release_date: date | None = None,
) -> bytes:
    """Download a Netflix earnings transcript PDF. Returns the raw PDF bytes."""
    if url is None:
        url = discover_transcript_url(year, quarter, release_date=release_date)
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

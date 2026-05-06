"""Netflix earnings-transcript discovery for the full corpus (Block 6a).

For each (year, quarter) in the configured window, probe the q4cdn
filename patterns via ``discover_transcript_url``. Returns one entry per
quarter with a ``status`` field that distinguishes discovered URLs from
quarters needing manual URL entry.

Each returned dict:

    {
        "document_id":   "nflx-q1-2024-transcript",
        "source":        "q4cdn",
        "form":          "transcript",
        "document_kind": "transcript",
        "fiscal_period": "2024Q1",
        "url":           "https://s22.q4cdn.com/.../...pdf" | None,
        "status":        "discovered" | "missing",
    }

Note: 2024+ Netflix transcripts have started appearing under
``/doc_events/{YYYY}/{Mon}/{DD}/`` rather than the
``/doc_financials/{year}/q{N}/`` pattern hardcoded in
``discover_transcript_url``. Quarters that fail discovery here will
need their URL added by hand to the manifest before Block 6b runs.
"""
from __future__ import annotations

from datetime import date
from typing import Iterable

from ingestion.normalize import assign_document_id
from ingestion.transcripts.fetch import discover_transcript_url


def _quarters_in_range(
    start_year: int, start_quarter: int, end_year: int, end_quarter: int
) -> Iterable[tuple[int, int]]:
    y, q = start_year, start_quarter
    while (y, q) <= (end_year, end_quarter):
        yield (y, q)
        q += 1
        if q > 4:
            q = 1
            y += 1


def list_netflix_transcripts(
    *,
    start_year: int = 2016,
    start_quarter: int = 1,
    end_year: int = 2026,
    end_quarter: int = 1,
    release_dates: dict[tuple[int, int], date] | None = None,
    progress: bool = True,
) -> list[dict]:
    """Probe the q4cdn host for every quarter in the window.

    ``release_dates`` maps (year, quarter) -> the actual earnings-release
    date for that quarter. Pass it (typically derived from 8-K Item 2.02
    filing dates) to enable the current ``/doc_events/{YYYY}/{Mon}/{DD}/``
    URL layout, which embeds the release date in the path."""
    release_dates = release_dates or {}
    out: list[dict] = []
    quarters = list(_quarters_in_range(start_year, start_quarter, end_year, end_quarter))
    for i, (y, q) in enumerate(quarters, start=1):
        doc_id = assign_document_id(form="transcript", period=f"{y}Q{q}")
        rd = release_dates.get((y, q))
        if progress:
            tag = f" (rd={rd.isoformat()})" if rd else ""
            print(f"  [{i}/{len(quarters)}] {y}Q{q}{tag} ... ", end="", flush=True)
        try:
            url = discover_transcript_url(y, q, release_date=rd)
            status = "discovered"
            if progress:
                print("ok")
        except LookupError:
            url = None
            status = "missing"
            if progress:
                print("MISSING")
        out.append(
            {
                "document_id": doc_id,
                "source": "q4cdn",
                "form": "transcript",
                "document_kind": "transcript",
                "fiscal_period": f"{y}Q{q}",
                "url": url,
                "status": status,
            }
        )
    return out

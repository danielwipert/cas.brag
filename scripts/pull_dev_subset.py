"""Pull the three Block 2 dev-subset documents from SEC EDGAR + q4cdn.

Run from repo root:
    python -m scripts.pull_dev_subset

Outputs:
    data/raw/nflx-10q-2024-q3.html       (primary 10-Q HTML)
    data/raw/nflx-10q-2024-q3.xbrl.xml   (XBRL instance, when available)
    data/raw/nflx-q4-2023-letter.html    (Exhibit 99.1 raw)
    data/raw/nflx-q1-2024-transcript.pdf (PDF)
    data/dev_subset/nflx-10q-2024-q3/{section}.txt
    data/dev_subset/nflx-q4-2023-letter/letter_body.txt
    data/dev_subset/nflx-q1-2024-transcript/transcript.txt
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ingestion.edgar import fetch as edgar_fetch
from ingestion.transcripts import fetch as transcript_fetch
from ingestion.normalize import assign_document_id, normalize_text


RAW_DIR = REPO_ROOT / "data" / "raw"
DEV_DIR = REPO_ROOT / "data" / "dev_subset"


def _load_dotenv() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _safe_section_filename(section: str) -> str:
    return (
        section.lower()
        .replace(" ", "_")
        .replace("&", "and")
        .replace(".", "")
        .replace("/", "_")
    )


def pull_q3_2024_10q() -> dict:
    print("\n[1/3] Q3 2024 10-Q")
    filing = edgar_fetch.find_netflix_filing(form="10-Q", period_of_report="2024-09-30")
    print(f"  found accession={filing.accession_number} filed={filing.filing_date}")

    doc_id = assign_document_id(form="10-Q", period=date(2024, 9, 30))
    raw_html = RAW_DIR / f"{doc_id}.html"
    edgar_fetch.save_primary_html(filing, raw_html)
    print(f"  saved primary HTML -> {raw_html.relative_to(REPO_ROOT)}")

    xbrl_dest = RAW_DIR / f"{doc_id}.xbrl.xml"
    saved_xbrl = edgar_fetch.save_xbrl(filing, xbrl_dest)
    if saved_xbrl:
        print(f"  saved XBRL          -> {saved_xbrl.relative_to(REPO_ROOT)}")
    else:
        print("  XBRL save: not located via attachments (will retry in Block 4)")

    # Section extraction.
    sections = edgar_fetch.extract_sections(filing)
    out_dir = DEV_DIR / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, int]] = []
    for section_name, raw_text in sections.items():
        if not raw_text:
            continue
        text = normalize_text(raw_text)
        if not text:
            continue
        fname = _safe_section_filename(section_name) + ".txt"
        path = out_dir / fname
        path.write_text(text, encoding="utf-8")
        written.append((section_name, len(text.split())))
    print(f"  extracted {len(written)} sections:")
    for section, wc in written:
        print(f"    - {section}: {wc} words")

    return {
        "document_id": doc_id,
        "form": "10-Q",
        "filing_date": str(filing.filing_date),
        "period_of_report": str(filing.period_of_report),
        "accession": filing.accession_number,
        "sections": dict(written),
        "xbrl_saved": bool(saved_xbrl),
    }


def pull_q4_2023_letter() -> dict:
    print("\n[2/3] Q4 2023 shareholder letter")
    # Q4 2023 results were filed in late January 2024 as an 8-K with Item 2.02.
    filing = edgar_fetch.find_netflix_8k_letter(
        filed_on_or_after="2024-01-01",
        filed_on_or_before="2024-02-15",
    )
    print(f"  found accession={filing.accession_number} filed={filing.filing_date}")

    doc_id = assign_document_id(
        form="8-K", period=date(2023, 12, 31), document_kind="letter"
    )
    body_html = edgar_fetch.extract_exhibit_991(filing)
    raw_path = RAW_DIR / f"{doc_id}.html"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(body_html, encoding="utf-8")
    print(f"  saved Exhibit 99.1 raw -> {raw_path.relative_to(REPO_ROOT)}")

    text = normalize_text(body_html)
    out_dir = DEV_DIR / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "letter_body.txt"
    out_path.write_text(text, encoding="utf-8")
    word_count = len(text.split())
    print(f"  normalized -> {out_path.relative_to(REPO_ROOT)}  ({word_count} words)")

    return {
        "document_id": doc_id,
        "form": "8-K (Exhibit 99.1)",
        "filing_date": str(filing.filing_date),
        "period_of_report": str(filing.period_of_report),
        "accession": filing.accession_number,
        "word_count": word_count,
    }


def pull_q1_2024_transcript() -> dict:
    print("\n[3/3] Q1 2024 transcript")
    doc_id = assign_document_id(form="transcript", period=date(2024, 3, 31))

    # Allow URL override via env var (.env: NFLX_Q1_2024_TRANSCRIPT_URL=...).
    override_url = os.environ.get("NFLX_Q1_2024_TRANSCRIPT_URL") or None

    try:
        pdf_bytes = transcript_fetch.fetch_transcript(2024, 1, url=override_url)
    except LookupError as e:
        print(f"  SKIPPED — {e}")
        print("  Set NFLX_Q1_2024_TRANSCRIPT_URL in .env to the q4cdn PDF link "
              "(see https://ir.netflix.net/financials/quarterly-earnings/) and re-run.")
        return {
            "document_id": doc_id,
            "form": "transcript",
            "status": "skipped_missing_url",
            "error": str(e),
        }

    raw_path = RAW_DIR / f"{doc_id}.pdf"
    transcript_fetch.save_transcript(pdf_bytes, raw_path)
    print(f"  saved PDF        -> {raw_path.relative_to(REPO_ROOT)}  ({len(pdf_bytes):,} bytes)")

    text = transcript_fetch.extract_transcript_text(pdf_bytes)
    text = normalize_text(text)
    out_dir = DEV_DIR / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "transcript.txt"
    out_path.write_text(text, encoding="utf-8")
    word_count = len(text.split())
    print(f"  normalized       -> {out_path.relative_to(REPO_ROOT)}  ({word_count} words)")

    return {
        "document_id": doc_id,
        "form": "transcript",
        "source": "s22.q4cdn.com",
        "word_count": word_count,
    }


def main() -> None:
    _load_dotenv()
    edgar_fetch.configure()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DEV_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "q3_2024_10q": pull_q3_2024_10q(),
        "q4_2023_letter": pull_q4_2023_letter(),
        "q1_2024_transcript": pull_q1_2024_transcript(),
    }

    summary_path = REPO_ROOT / "data" / "logs" / "dev_subset_pull.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to {summary_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

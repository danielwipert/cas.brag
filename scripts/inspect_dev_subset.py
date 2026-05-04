"""One-off inspection (build plan Block 2 done-when criterion).

Prints, for each dev-subset document:
    - document_id
    - type / form
    - filing_date / period_of_report
    - section_count, total_word_count
    - first 200 chars of each section

Run:  python -m scripts.inspect_dev_subset
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


SUMMARY_PATH = REPO_ROOT / "data" / "logs" / "dev_subset_pull.json"
DEV_DIR = REPO_ROOT / "data" / "dev_subset"


def _read_pull_summary() -> dict:
    if not SUMMARY_PATH.exists():
        return {}
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def _inspect_document(doc_dir: Path) -> dict:
    if not doc_dir.is_dir():
        return {"present": False}
    sections: list[tuple[str, str]] = []
    for path in sorted(doc_dir.glob("*.txt")):
        sections.append((path.stem, path.read_text(encoding="utf-8")))
    total_words = sum(len(text.split()) for _, text in sections)
    return {
        "present": True,
        "section_count": len(sections),
        "total_word_count": total_words,
        "sections": sections,
    }


def _print_doc(label: str, doc_id: str, meta: dict, fs: dict) -> None:
    print("=" * 78)
    print(f"  {label}: {doc_id}")
    print("=" * 78)
    if not fs.get("present"):
        print(f"  (no extracted content at {DEV_DIR / doc_id})")
        return

    print(f"  form           : {meta.get('form', '?')}")
    print(f"  filing_date    : {meta.get('filing_date', '?')}")
    print(f"  period         : {meta.get('period_of_report', meta.get('period', '?'))}")
    if "accession" in meta:
        print(f"  accession      : {meta['accession']}")
    print(f"  section_count  : {fs['section_count']}")
    print(f"  total_words    : {fs['total_word_count']:,}")
    print()
    for section_name, text in fs["sections"]:
        words = len(text.split())
        head = text.replace("\n", " ").strip()[:200]
        print(f"  ── {section_name}  ({words:,} words)")
        print(f"     {head!r}")
        print()


def main() -> None:
    summary = _read_pull_summary()

    plan = [
        ("[1/3] Q3 2024 10-Q",          "nflx-10q-2024-q3",     summary.get("q3_2024_10q", {})),
        ("[2/3] Q4 2023 letter",        "nflx-q4-2023-letter",  summary.get("q4_2023_letter", {})),
        ("[3/3] Q1 2024 transcript",    "nflx-q1-2024-transcript", summary.get("q1_2024_transcript", {})),
    ]
    for label, doc_id, meta in plan:
        fs = _inspect_document(DEV_DIR / doc_id)
        _print_doc(label, doc_id, meta, fs)

    if summary.get("q1_2024_transcript", {}).get("status") == "skipped_missing_url":
        print()
        print("Note: Q1 2024 transcript was skipped — URL unknown.")
        print("      Set NFLX_Q1_2024_TRANSCRIPT_URL in .env, then re-run pull_dev_subset.")


if __name__ == "__main__":
    main()

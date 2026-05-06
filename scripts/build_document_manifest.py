"""Build the corpus document manifest (Block 6a).

Enumerates Netflix SEC filings (10-K, 10-Q, 8-K Item 2.02) and
earnings-call transcripts in the build-plan window (May 2016 – May 2026)
and writes them to ``data/raw/document_manifest.json``.

Run from the repo root::

    python -m scripts.build_document_manifest
    python -m scripts.build_document_manifest --skip-transcripts   # SEC only
    python -m scripts.build_document_manifest --skip-filings       # transcripts only

Design notes:
- Filings discovery is fast (a couple of EDGAR queries).
- Transcript discovery is ~37 quarters * ~9 HEAD probes = a few minutes.
- ``status: "missing"`` quarters need a URL added by hand before Block 6b.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

from ingestion.edgar.list_filings import list_netflix_filings
from ingestion.transcripts.list_transcripts import list_netflix_transcripts


MANIFEST_PATH = Path("data/raw/document_manifest.json")
TRANSCRIPTS_DIR = Path("data/raw/transcripts")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--skip-filings",
        action="store_true",
        help="Skip SEC filings discovery (use existing manifest entries).",
    )
    p.add_argument(
        "--skip-transcripts",
        action="store_true",
        help="Skip transcript discovery (use existing manifest entries).",
    )
    p.add_argument(
        "--filed-on-or-after",
        default="2016-05-06",
        help="Inclusive lower bound on SEC filing_date.",
    )
    p.add_argument(
        "--filed-on-or-before",
        default="2026-05-06",
        help="Inclusive upper bound on SEC filing_date.",
    )
    return p.parse_args()


def _existing_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"filings": [], "transcripts": []}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _summarize_filings(filings: list[dict]) -> None:
    by_kind: Counter[str] = Counter()
    for f in filings:
        if f["form"] == "8-K" and f["document_kind"] == "letter":
            by_kind["8-K letter"] += 1
        elif f["form"] == "8-K":
            by_kind["8-K (unmapped)"] += 1
        else:
            by_kind[f["form"]] += 1
    print(f"  by kind: {dict(by_kind)}")


def main() -> None:
    args = parse_args()
    existing = _existing_manifest()

    if args.skip_filings:
        filings = existing.get("filings", [])
        print(f"Reusing {len(filings)} existing filing entries.")
    else:
        print(
            f"Discovering SEC filings filed {args.filed_on_or_after} to "
            f"{args.filed_on_or_before}..."
        )
        filings = list_netflix_filings(
            filed_on_or_after=args.filed_on_or_after,
            filed_on_or_before=args.filed_on_or_before,
        )
        print(f"  {len(filings)} filings found")
        _summarize_filings(filings)

    if args.skip_transcripts:
        transcripts = existing.get("transcripts", [])
        print(f"Reusing {len(transcripts)} existing transcript entries.")
    else:
        # Use 8-K Item 2.02 filing dates as the release dates for the
        # ``/doc_events/{YYYY}/{Mon}/{DD}/`` URL layout. Each 8-K letter
        # entry has a fiscal_period like "2024Q1" and a filing_date
        # like "2024-04-18".
        release_dates: dict[tuple[int, int], date] = {}
        for f in filings:
            if f.get("document_kind") != "letter" or not f.get("fiscal_period"):
                continue
            fp = f["fiscal_period"]
            if "Q" not in fp:
                continue
            try:
                yr, qtr = fp.split("Q")
                rd = date.fromisoformat(f["filing_date"])
                release_dates[(int(yr), int(qtr))] = rd
            except (ValueError, KeyError):
                continue
        print(
            f"\nDiscovering earnings transcripts (release dates known for "
            f"{len(release_dates)} quarters)..."
        )
        transcripts = list_netflix_transcripts(release_dates=release_dates)
        discovered = sum(1 for t in transcripts if t["status"] == "discovered")
        missing = [t for t in transcripts if t["status"] == "missing"]
        print(
            f"  {len(transcripts)} quarters checked, "
            f"{discovered} discovered, {len(missing)} missing"
        )
        if missing:
            print("  Missing quarters (need manual URL entry):")
            for t in missing:
                print(f"    - {t['document_id']}")

    # Wire up local transcript files. Any PDF in data/raw/transcripts/ named
    # like {document_id}.pdf becomes the source of truth for that quarter,
    # regardless of whether the URL was also discovered. Block 6b reads
    # local_path when present and falls back to url otherwise.
    n_local = 0
    if TRANSCRIPTS_DIR.exists():
        local_pdfs = {p.stem: p for p in TRANSCRIPTS_DIR.glob("*.pdf")}
        for t in transcripts:
            local = local_pdfs.get(t["document_id"])
            if local is None:
                t["local_path"] = None
                continue
            t["local_path"] = str(local).replace("\\", "/")
            t["status"] = "local"
            n_local += 1
    else:
        for t in transcripts:
            t.setdefault("local_path", None)

    manifest = {
        "filings": filings,
        "transcripts": transcripts,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest -> {MANIFEST_PATH}")
    n_url_only = sum(1 for t in transcripts if t.get("status") == "discovered" and not t.get("local_path"))
    n_missing = sum(1 for t in transcripts if t.get("status") not in ("discovered", "local"))
    print(
        f"Total: {len(filings)} filings + {len(transcripts)} transcripts "
        f"({n_local} local, {n_url_only} url-only, {n_missing} missing) "
        f"= {len(filings) + len(transcripts)} documents"
    )


if __name__ == "__main__":
    main()

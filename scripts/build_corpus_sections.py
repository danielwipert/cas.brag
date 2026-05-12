"""Extract per-section normalized text for every document in the manifest
(Block 6c, stage 1).

Reads ``data/raw/document_manifest.json`` and writes one ``.txt`` file per
section under ``data/corpus/{document_id}/``. The chunker in
``ingestion/chunker/section_aware.py`` consumes this tree.

Layout:
    data/corpus/nflx-10k-2024/item_1.txt        (Business)
    data/corpus/nflx-10k-2024/item_1a.txt       (Risk Factors)
    data/corpus/nflx-10k-2024/item_7.txt        (MD&A)
    data/corpus/nflx-10k-2024/item_7a.txt       (Quant disclosure)
    data/corpus/nflx-10k-2024/notes_to_financial_statements.txt
    data/corpus/nflx-q4-2023-letter/letter_body.txt
    data/corpus/nflx-q1-2024-transcript/transcript.txt

Strategy by source:
    - 10-K / 10-K/A / 10-Q: refetch the typed Filing object (edgartools)
      and call extract_sections(); this is the only path that gives us
      the per-item sections cleanly.
    - 8-K letter: the saved .html on disk is already the letter body
      (output of extract_exhibit_991 during Block 6b acquisition), so
      we just normalize it.
    - Transcript: load the PDF bytes, extract with the same helper Block 2
      validated on the dev subset.

Idempotent: skips any document whose target dir exists and is non-empty,
unless --force. ``--only doc_id1,doc_id2`` restricts the run to specific
documents for spot-checks.

Run from repo root::

    python -m scripts.build_corpus_sections
    python -m scripts.build_corpus_sections --force
    python -m scripts.build_corpus_sections --only nflx-10k-2024,nflx-q4-2023-letter
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ingestion.edgar import fetch as edgar_fetch
from ingestion.transcripts import fetch as transcript_fetch
from ingestion.normalize import normalize_text


MANIFEST_PATH = REPO_ROOT / "data" / "raw" / "document_manifest.json"
CORPUS_DIR = REPO_ROOT / "data" / "corpus"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-extract sections even if the corpus dir is already populated.",
    )
    p.add_argument(
        "--only",
        default="",
        help="Comma-separated document_ids to limit the run to.",
    )
    return p.parse_args()


def _load_dotenv() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _safe_section_filename(section: str) -> str:
    """Same convention pull_dev_subset used so file stems match SKIP_SECTIONS."""
    return (
        section.lower()
        .replace(" ", "_")
        .replace("&", "and")
        .replace(".", "")
        .replace("/", "_")
    )


def _doc_dir(document_id: str) -> Path:
    return CORPUS_DIR / document_id


def _already_extracted(document_id: str) -> bool:
    d = _doc_dir(document_id)
    return d.exists() and any(d.glob("*.txt"))


def _write_section(doc_dir: Path, section: str, raw_text: str) -> int:
    """Write one section .txt under doc_dir. Returns word count, or 0 if empty."""
    text = normalize_text(raw_text or "")
    if not text:
        return 0
    fname = _safe_section_filename(section) + ".txt"
    (doc_dir / fname).write_text(text, encoding="utf-8")
    return len(text.split())


def _extract_periodic(entry: dict, doc_dir: Path) -> dict[str, int]:
    """10-K / 10-K/A / 10-Q: refetch filing and extract typed sections."""
    filing = edgar_fetch.fetch_filing(
        cik=edgar_fetch.NETFLIX_CIK, accession_number=entry["accession"]
    )
    sections = edgar_fetch.extract_sections(filing)
    written: dict[str, int] = {}
    for section_name, raw_text in sections.items():
        wc = _write_section(doc_dir, section_name, raw_text)
        if wc:
            written[_safe_section_filename(section_name)] = wc
    return written


def _extract_letter(entry: dict, doc_dir: Path) -> dict[str, int]:
    """8-K letter: the saved .html is already the letter body."""
    src = REPO_ROOT / entry["local_path"]
    body = src.read_text(encoding="utf-8")
    wc = _write_section(doc_dir, "letter_body", body)
    return {"letter_body": wc} if wc else {}


def _extract_transcript(entry: dict, doc_dir: Path) -> dict[str, int]:
    """Transcript: PDF bytes -> extract_transcript_text -> normalize."""
    src = REPO_ROOT / entry["local_path"]
    pdf_bytes = src.read_bytes()
    text = transcript_fetch.extract_transcript_text(pdf_bytes)
    wc = _write_section(doc_dir, "transcript", text)
    return {"transcript": wc} if wc else {}


def main() -> None:
    args = parse_args()
    _load_dotenv()
    edgar_fetch.configure()

    if not MANIFEST_PATH.exists():
        raise SystemExit(f"{MANIFEST_PATH} not found.")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    docs: list[tuple[str, dict, str]] = []  # (kind, entry, doc_id)
    for f in manifest.get("filings", []):
        if f.get("document_kind") == "letter_unmapped":
            continue
        if not f.get("local_path"):
            continue
        if f["form"] in ("10-K", "10-K/A", "10-Q"):
            docs.append(("periodic", f, f["document_id"]))
        elif f["form"] == "8-K" and f.get("document_kind") == "letter":
            docs.append(("letter", f, f["document_id"]))
    for t in manifest.get("transcripts", []):
        if not t.get("local_path"):
            continue
        docs.append(("transcript", t, t["document_id"]))

    only: set[str] = set()
    if args.only:
        only = {s.strip() for s in args.only.split(",") if s.strip()}
        unknown = only - {d for _, _, d in docs}
        if unknown:
            raise SystemExit(f"--only mentions unknown document_ids: {sorted(unknown)}")

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    total = len(docs)
    n_skipped = 0
    n_filtered = 0
    n_done = 0
    failures: list[tuple[str, str]] = []
    summary: dict[str, dict[str, int]] = {}

    for i, (kind, entry, doc_id) in enumerate(docs, start=1):
        prefix = f"[{i:>3}/{total}] {doc_id:<32} {kind:<10}"

        if only and doc_id not in only:
            n_filtered += 1
            continue

        if not args.force and _already_extracted(doc_id):
            print(f"{prefix} skip (already extracted)")
            n_skipped += 1
            continue

        doc_dir = _doc_dir(doc_id)
        doc_dir.mkdir(parents=True, exist_ok=True)
        try:
            if kind == "periodic":
                written = _extract_periodic(entry, doc_dir)
            elif kind == "letter":
                written = _extract_letter(entry, doc_dir)
            elif kind == "transcript":
                written = _extract_transcript(entry, doc_dir)
            else:
                raise RuntimeError(f"unhandled kind: {kind}")
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            failures.append((doc_id, msg))
            print(f"{prefix} FAILED  {msg}")
            continue

        if not written:
            print(f"{prefix} EMPTY (no sections written)")
            failures.append((doc_id, "no sections produced"))
            continue

        wc_total = sum(written.values())
        sect_list = ", ".join(f"{k}({v}w)" for k, v in written.items())
        print(f"{prefix} {wc_total:>7}w  [{sect_list}]")
        summary[doc_id] = written
        n_done += 1

    print()
    print("=== Summary ===")
    print(f"  extracted:        {n_done}")
    print(f"  skipped (exists): {n_skipped}")
    if only:
        print(f"  skipped (filter): {n_filtered}")
    print(f"  failed:           {len(failures)}")
    for doc_id, msg in failures:
        print(f"    - {doc_id}: {msg}")

    log_path = REPO_ROOT / "data" / "logs" / "corpus_sections_build.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps({"per_document_words": summary, "failures": failures}, indent=2),
        encoding="utf-8",
    )
    print(f"\nLog -> {log_path.relative_to(REPO_ROOT)}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()

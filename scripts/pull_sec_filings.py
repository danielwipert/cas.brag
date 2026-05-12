"""Acquire all SEC filings listed in the document manifest (Block 6b).

Reads ``data/raw/document_manifest.json`` and downloads every filing
entry's primary document (and XBRL where applicable) into ``data/raw/``.
Writes ``local_path`` (and ``xbrl_local_path`` for 10-K/10-Q) back into
the manifest after each filing, so a crash mid-run leaves consistent
state and a rerun resumes cleanly.

Layout:
    data/raw/{document_id}.html       primary HTML (10-K/10-Q) or
                                       Exhibit 99.1 body (8-K letter)
    data/raw/{document_id}.xbrl.xml   XBRL instance (10-K/10-Q only,
                                       when available via attachments)

Run from repo root::

    python -m scripts.pull_sec_filings              # pull all missing
    python -m scripts.pull_sec_filings --force      # redownload everything
    python -m scripts.pull_sec_filings --dry-run    # list, don't fetch
    python -m scripts.pull_sec_filings --only nflx-10k-2023,nflx-10q-2024-q3

Skips entries with ``document_kind == "letter_unmapped"`` — those are
8-K Item 2.02 filings in non-earnings months and need manual review
before they belong in the corpus.
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


MANIFEST_PATH = REPO_ROOT / "data" / "raw" / "document_manifest.json"
RAW_DIR = REPO_ROOT / "data" / "raw"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--force",
        action="store_true",
        help="Redownload filings even if local_path is set and the file exists.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be fetched without downloading.",
    )
    p.add_argument(
        "--only",
        default="",
        help="Comma-separated document_ids to limit the run to (for testing).",
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


def _save_manifest_atomic(manifest: dict) -> None:
    """Write to a sibling .tmp file then rename — never leave a half-written
    manifest if a process is killed mid-write."""
    tmp = MANIFEST_PATH.with_suffix(MANIFEST_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(MANIFEST_PATH)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _local_path_for(entry: dict) -> Path:
    return RAW_DIR / f"{entry['document_id']}.html"


def _xbrl_path_for(entry: dict) -> Path:
    return RAW_DIR / f"{entry['document_id']}.xbrl.xml"


def _already_local(entry: dict) -> bool:
    lp = entry.get("local_path")
    if not lp:
        return False
    return (REPO_ROOT / lp).exists()


def _adopt_existing_files(manifest: dict) -> int:
    """For any filing entry without local_path but whose canonical HTML
    (and XBRL, where applicable) exist on disk, write local_path back.
    This keeps Block 2 dev-subset artifacts from being re-downloaded.
    Returns the number of entries adopted."""
    adopted = 0
    for entry in manifest.get("filings", []):
        if entry.get("document_kind") == "letter_unmapped":
            continue
        canonical_html = _local_path_for(entry)
        if entry.get("local_path") or not canonical_html.exists():
            continue
        entry["local_path"] = _rel(canonical_html)
        entry["status"] = "local"
        if entry["form"] in ("10-K", "10-K/A", "10-Q"):
            canonical_xbrl = _xbrl_path_for(entry)
            if canonical_xbrl.exists():
                entry["xbrl_local_path"] = _rel(canonical_xbrl)
        adopted += 1
    return adopted


def _fetch_periodic(entry: dict) -> tuple[Path, Path | None]:
    """Download a 10-K, 10-K/A, or 10-Q. Returns (html_path, xbrl_path_or_None)."""
    filing = edgar_fetch.fetch_filing(
        cik=edgar_fetch.NETFLIX_CIK, accession_number=entry["accession"]
    )
    html_path = _local_path_for(entry)
    edgar_fetch.save_primary_html(filing, html_path)
    xbrl_path = edgar_fetch.save_xbrl(filing, _xbrl_path_for(entry))
    return html_path, xbrl_path


def _fetch_letter(entry: dict) -> Path:
    """Download an 8-K shareholder letter (Exhibit 99.1 body)."""
    filing = edgar_fetch.fetch_filing(
        cik=edgar_fetch.NETFLIX_CIK, accession_number=entry["accession"]
    )
    body_html = edgar_fetch.extract_exhibit_991(filing)
    html_path = _local_path_for(entry)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(body_html, encoding="utf-8")
    return html_path


def main() -> None:
    args = parse_args()
    _load_dotenv()
    edgar_fetch.configure()

    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"{MANIFEST_PATH} not found — run scripts/build_document_manifest.py first."
        )

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    filings = manifest.get("filings", [])
    if not filings:
        raise SystemExit("Manifest has no filings.")

    only: set[str] = set()
    if args.only:
        only = {s.strip() for s in args.only.split(",") if s.strip()}
        unknown = only - {f["document_id"] for f in filings}
        if unknown:
            raise SystemExit(f"--only mentions unknown document_ids: {sorted(unknown)}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    adopted = _adopt_existing_files(manifest)
    if adopted:
        _save_manifest_atomic(manifest)
        print(f"Adopted {adopted} existing filing(s) from data/raw/ (local_path written).")

    total = len(filings)
    n_skipped_already_local = 0
    n_skipped_unmapped = 0
    n_skipped_filtered = 0
    n_fetched_periodic = 0
    n_fetched_letter = 0
    n_xbrl_saved = 0
    failures: list[tuple[str, str]] = []

    for i, entry in enumerate(filings, start=1):
        doc_id = entry["document_id"]
        form = entry["form"]
        kind = entry.get("document_kind")
        prefix = f"[{i:>2}/{total}] {doc_id:<26} {form:<6}"

        if only and doc_id not in only:
            n_skipped_filtered += 1
            continue

        if kind == "letter_unmapped":
            print(f"{prefix} SKIP letter_unmapped (manual review)")
            n_skipped_unmapped += 1
            continue

        if not args.force and _already_local(entry):
            print(f"{prefix} skip (already local: {entry['local_path']})")
            n_skipped_already_local += 1
            continue

        if args.dry_run:
            target = _local_path_for(entry)
            extra = " +xbrl" if form in ("10-K", "10-K/A", "10-Q") else ""
            print(f"{prefix} DRY-RUN would fetch -> {_rel(target)}{extra}")
            continue

        try:
            if kind == "letter":
                html_path = _fetch_letter(entry)
                entry["local_path"] = _rel(html_path)
                entry.pop("xbrl_local_path", None)
                entry.pop("fetch_error", None)
                entry["status"] = "local"
                print(f"{prefix} -> {entry['local_path']}")
                n_fetched_letter += 1
            elif form in ("10-K", "10-K/A", "10-Q"):
                html_path, xbrl_path = _fetch_periodic(entry)
                entry["local_path"] = _rel(html_path)
                if xbrl_path is not None:
                    entry["xbrl_local_path"] = _rel(xbrl_path)
                    n_xbrl_saved += 1
                else:
                    entry.pop("xbrl_local_path", None)
                entry.pop("fetch_error", None)
                entry["status"] = "local"
                xbrl_note = f"  +xbrl ({_rel(xbrl_path)})" if xbrl_path else "  (no xbrl)"
                print(f"{prefix} -> {entry['local_path']}{xbrl_note}")
                n_fetched_periodic += 1
            else:
                print(f"{prefix} SKIP unhandled form/kind (form={form} kind={kind!r})")
                continue
        except Exception as e:  # broad catch: continue on per-filing failure
            msg = f"{type(e).__name__}: {e}"
            entry["status"] = "fetch_failed"
            entry["fetch_error"] = msg
            failures.append((doc_id, msg))
            print(f"{prefix} FAILED  {msg}")

        # Atomic write after each entry (success or failure) for crash safety.
        _save_manifest_atomic(manifest)

    print()
    print("=== Summary ===")
    print(f"  fetched periodic (10-K/Q): {n_fetched_periodic}")
    print(f"  fetched letters (8-K):     {n_fetched_letter}")
    print(f"  XBRL files saved:          {n_xbrl_saved}")
    print(f"  skipped (already local):   {n_skipped_already_local}")
    print(f"  skipped (letter_unmapped): {n_skipped_unmapped}")
    if only:
        print(f"  skipped (not in --only):   {n_skipped_filtered}")
    if failures:
        print(f"  FAILURES ({len(failures)}):")
        for doc_id, msg in failures:
            print(f"    - {doc_id}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()

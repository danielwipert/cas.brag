"""Download transcript PDFs for entries the manifest already has URLs for.

Reads ``data/raw/document_manifest.json`` and, for every transcript with
``status == 'discovered'``, writes the PDF to
``data/raw/transcripts/{document_id}.pdf``. Idempotent — skips any file
that already exists. Quarters with ``status == 'missing'`` (no URL) are
listed at the end so you know what to grab by hand.

Run from the repo root::

    python -m scripts.download_discovered_transcripts
    python -m scripts.download_discovered_transcripts --force   # re-download
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


MANIFEST_PATH = Path("data/raw/document_manifest.json")
DEST_DIR = Path("data/raw/transcripts")
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the destination file already exists.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"{MANIFEST_PATH} not found — run scripts/build_document_manifest.py first."
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    transcripts = manifest.get("transcripts", [])
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    discovered = [t for t in transcripts if t.get("status") == "discovered" and t.get("url")]
    missing = [t for t in transcripts if t.get("status") != "discovered"]

    print(f"{len(discovered)} transcripts have URLs; {len(missing)} need manual download.")
    n_pulled = n_skipped = n_failed = 0
    headers = {"User-Agent": _USER_AGENT}
    for t in discovered:
        dest = DEST_DIR / f"{t['document_id']}.pdf"
        if dest.exists() and not args.force:
            n_skipped += 1
            continue
        url = t["url"]
        try:
            r = requests.get(url, headers=headers, timeout=60)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  FAIL  {t['document_id']}  {e}")
            n_failed += 1
            continue
        dest.write_bytes(r.content)
        print(f"  ok    {t['document_id']}  ({len(r.content):>8d} bytes)")
        n_pulled += 1

    print(
        f"\nDownloaded: {n_pulled}   Skipped (exist): {n_skipped}   "
        f"Failed: {n_failed}"
    )
    if missing:
        print(f"\nStill needs manual download to {DEST_DIR}/{{document_id}}.pdf:")
        for t in missing:
            print(f"  {t['document_id']}  ({t['fiscal_period']})")


if __name__ == "__main__":
    main()

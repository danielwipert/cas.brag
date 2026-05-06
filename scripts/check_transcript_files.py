"""Report transcript-file status across the corpus.

Reads ``data/raw/document_manifest.json`` and groups every transcript
into one of:

  local       — PDF exists at data/raw/transcripts/{document_id}.pdf
  url-only    — manifest has a URL but no local file yet
                (run scripts.download_discovered_transcripts to pull these)
  missing     — neither a URL nor a local file; needs manual download

Run from the repo root::

    python -m scripts.check_transcript_files
    python -m scripts.check_transcript_files --missing-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


MANIFEST_PATH = Path("data/raw/document_manifest.json")
TRANSCRIPTS_DIR = Path("data/raw/transcripts")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--missing-only",
        action="store_true",
        help="Print only the quarters that still need a manual download.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"{MANIFEST_PATH} not found.")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    transcripts = manifest.get("transcripts", [])

    # Re-derive state from disk so it's accurate even if the manifest is stale.
    local_files = {p.stem for p in TRANSCRIPTS_DIR.glob("*.pdf")} if TRANSCRIPTS_DIR.exists() else set()

    local: list[dict] = []
    url_only: list[dict] = []
    missing: list[dict] = []
    for t in transcripts:
        if t["document_id"] in local_files:
            local.append(t)
        elif t.get("url"):
            url_only.append(t)
        else:
            missing.append(t)

    if not args.missing_only:
        print(f"Total transcripts: {len(transcripts)}")
        print(f"  local file present:   {len(local):>3d}")
        print(f"  URL only (downloadable): {len(url_only):>3d}")
        print(f"  missing (manual):     {len(missing):>3d}")
        if local:
            print("\nLocal:")
            for t in local:
                print(f"  {t['document_id']}")
        if url_only:
            print("\nURL only (run scripts.download_discovered_transcripts):")
            for t in url_only:
                print(f"  {t['document_id']}")

    if missing:
        print(f"\nMissing — paste each PDF as data/raw/transcripts/{{document_id}}.pdf:")
        for t in missing:
            print(f"  {t['document_id']}  (release ~{t['fiscal_period']})")
    elif not args.missing_only:
        print("\nAll transcripts accounted for.")


if __name__ == "__main__":
    main()

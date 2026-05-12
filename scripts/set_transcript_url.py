"""Manually set a transcript URL in the document manifest.

Some Netflix earnings transcripts (mainly some 2024+ quarters under
``/doc_events/``) have UUID-based filenames that cannot be derived from
the (year, quarter) tuple. Use this script to paste the URL by hand.

Run from repo root::

    python -m scripts.set_transcript_url 2024Q2 \\
        "https://s22.q4cdn.com/959853165/files/doc_events/2024/Jul/18/netflix-inc-usa-d815629a-440e-4f6b-9f5e-831213418dd0.pdf"

Updates ``data/raw/document_manifest.json`` in place: sets the matching
quarter's ``url``, marks ``status='discovered'``, leaves everything else
alone. Does a HEAD probe first so you find out about typos immediately.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests


MANIFEST_PATH = Path("data/raw/document_manifest.json")
_USER_AGENT = "cas.brag/0.1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "fiscal_period",
        help="Quarter to set, e.g. 2024Q2, 2025Q3.",
    )
    p.add_argument("url", help="The full transcript URL.")
    p.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the HEAD probe (use when offline or the host blocks HEAD).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"{MANIFEST_PATH} not found — run scripts/build_document_manifest.py first."
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    if not args.no_verify:
        r = requests.head(
            args.url,
            timeout=15,
            allow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        )
        if r.status_code != 200:
            print(
                f"!! HEAD {args.url} returned {r.status_code} {r.reason}. "
                f"Pass --no-verify if you want to set it anyway.",
                file=sys.stderr,
            )
            raise SystemExit(2)

    target = None
    for t in manifest["transcripts"]:
        if t["fiscal_period"] == args.fiscal_period:
            target = t
            break
    if target is None:
        raise SystemExit(
            f"No transcript entry with fiscal_period={args.fiscal_period!r}. "
            f"Available: "
            + ", ".join(t["fiscal_period"] for t in manifest["transcripts"])
        )

    prior_url = target.get("url")
    target["url"] = args.url
    target["status"] = "discovered"
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Updated {target['document_id']}:")
    print(f"  prior url: {prior_url!r}")
    print(f"  new url:   {args.url}")
    print(f"  status:    discovered")


if __name__ == "__main__":
    main()

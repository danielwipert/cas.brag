"""Apply the GAAP-keyword post-filter to an existing dev_prose_facts.jsonl.

This avoids re-running the LLM extraction. Reads
``data/fact_store/dev_prose_facts.jsonl``, applies the filter, and writes
the survivors back to the same path. The original is preserved at
``data/fact_store/dev_prose_facts.unfiltered.jsonl`` so you can compare.

Run from the repo root::

    python -m scripts.filter_dev_prose_facts
    python -m scripts.filter_dev_prose_facts --dry-run     # show drops without writing
"""
from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

from ingestion.prose.extract import (
    is_gaap_leakage,
    load_facts_jsonl,
    post_filter_facts,
    save_facts_jsonl,
)


JSONL_PATH = Path("data/fact_store/dev_prose_facts.jsonl")
BACKUP_PATH = Path("data/fact_store/dev_prose_facts.unfiltered.jsonl")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which facts would be dropped without writing.",
    )
    p.add_argument(
        "--show",
        type=int,
        default=10,
        help="How many sample drops to print (default 10).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not JSONL_PATH.exists():
        raise SystemExit(f"Missing {JSONL_PATH}.")

    facts = load_facts_jsonl(JSONL_PATH)
    print(f"Loaded {len(facts)} prose facts from {JSONL_PATH}")

    kept, drops = post_filter_facts(facts)
    n_dropped = len(facts) - len(kept)
    drop_reason_counter = Counter(
        is_gaap_leakage(f.claim, f.fact_type.value)
        for f in facts
        if is_gaap_leakage(f.claim, f.fact_type.value)
    )

    print(f"Drops by reason: {dict(drop_reason_counter)}")
    print(f"Kept: {len(kept)}  /  Dropped: {n_dropped}")

    # Show sample drops grouped by fact_type so we can sanity-check the filter.
    dropped = [f for f in facts if is_gaap_leakage(f.claim, f.fact_type.value)]
    if dropped and args.show > 0:
        print(f"\nFirst {min(args.show, len(dropped))} drops:")
        for i, f in enumerate(dropped[: args.show], 1):
            reason = is_gaap_leakage(f.claim, f.fact_type.value)
            print(f"  [{i}] {f.fact_id}  fact_type={f.fact_type.value}  reason={reason}")
            print(f"        claim: {f.claim}")

    if args.dry_run:
        print("\n--dry-run: no files written.")
        return

    if not BACKUP_PATH.exists():
        shutil.copy2(JSONL_PATH, BACKUP_PATH)
        print(f"\nBacked up unfiltered facts -> {BACKUP_PATH}")
    save_facts_jsonl(kept, JSONL_PATH)
    print(f"Filtered facts -> {JSONL_PATH}")


if __name__ == "__main__":
    main()

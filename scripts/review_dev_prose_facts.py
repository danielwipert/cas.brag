"""Manual-review helper for Block 5.

Samples N prose facts (default 20) from data/fact_store/dev_prose_facts.jsonl,
finds each fact's source chunk via the chunker, and prints a side-by-side
view (fact JSON + the verbatim_anchor highlighted in the chunk) so a human
reviewer can sign off on each.

Build plan §Block 5 manual review checklist:
  - verbatim_anchor exists character-exact in the source chunk text
  - asserter is correct (Netflix for filings, named exec for transcripts)
  - fact_type classification is appropriate
  - period is correct when present
  - assertion_date matches the document
  - confidence looks reasonable (high for direct quotes, lower for paraphrases)
  - all 6 prose fact types represented across the 20 reviewed

Run from the repo root::

    python -m scripts.review_dev_prose_facts                # 20 random facts
    python -m scripts.review_dev_prose_facts --n 30
    python -m scripts.review_dev_prose_facts --seed 42      # reproducible sample
    python -m scripts.review_dev_prose_facts --type strategic_claim
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from ingestion.chunker.section_aware import (
    chunk_document,
    iter_dev_subset_documents,
)
from ingestion.prose.extract import load_facts_jsonl
from schemas.records import ChunkRecord, FactRecord


PROSE_JSONL = Path("data/fact_store/dev_prose_facts.jsonl")
DEV_SUBSET_ROOT = Path("data/dev_subset")


def _all_chunks_by_id() -> dict[str, ChunkRecord]:
    by_id: dict[str, ChunkRecord] = {}
    for doc in iter_dev_subset_documents(DEV_SUBSET_ROOT):
        for ch in chunk_document(doc):
            by_id[ch.chunk_id] = ch
    return by_id


def _find_chunk_for_fact(
    fact: FactRecord, chunks: list[ChunkRecord]
) -> ChunkRecord | None:
    """Locate the chunk whose text contains the verbatim_anchor and that
    matches (source_document, source_section). Returns None if not found."""
    for ch in chunks:
        if ch.source_document != fact.source_document:
            continue
        if ch.section != fact.source_section:
            continue
        if fact.verbatim_anchor in ch.text:
            return ch
    return None


def _print_fact(i: int, fact: FactRecord, chunk: ChunkRecord | None) -> None:
    print(f"\n{'=' * 78}")
    print(f"FACT {i}/{fact.fact_id}")
    print(f"{'=' * 78}")
    print(f"  fact_type:         {fact.fact_type.value}")
    print(f"  asserter:          {fact.asserter}")
    print(f"  period:            {fact.period}")
    print(f"  assertion_date:    {fact.assertion_date}")
    print(f"  source_document:   {fact.source_document}")
    print(f"  source_section:    {fact.source_section}")
    print(f"  value/unit:        {fact.value} {fact.unit or ''}".rstrip())
    print(f"  confidence:        {fact.confidence}")
    print(f"  CLAIM:             {fact.claim}")
    print(f"  ANCHOR:            {fact.verbatim_anchor!r}")
    if chunk is None:
        print("  CHUNK CONTEXT:     <chunk not found in dev subset>")
        return
    text = chunk.text
    pos = text.find(fact.verbatim_anchor)
    if pos < 0:
        print("  ANCHOR-IN-CHUNK:   <NOT FOUND -- THIS IS A BUG>")
        return
    end = pos + len(fact.verbatim_anchor)
    pre = text[max(0, pos - 200) : pos]
    post = text[end : end + 200]
    hit = text[pos:end]
    safe = lambda s: s.encode("ascii", "replace").decode("ascii")
    print(f"  ANCHOR-IN-CHUNK:   ...{safe(pre)}>>>{safe(hit)}<<<{safe(post)}...")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=20, help="Sample size (default 20).")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )
    p.add_argument(
        "--type",
        default=None,
        help="Filter to a single fact_type (e.g. strategic_claim).",
    )
    p.add_argument(
        "--doc",
        default=None,
        help="Filter to a single source_document (e.g. nflx-q4-2023-letter).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not PROSE_JSONL.exists():
        raise SystemExit(f"Missing {PROSE_JSONL}. Run Block 5 extraction first.")
    facts = load_facts_jsonl(PROSE_JSONL)
    if args.type:
        facts = [f for f in facts if f.fact_type.value == args.type]
    if args.doc:
        facts = [f for f in facts if f.source_document == args.doc]
    if not facts:
        raise SystemExit("No facts match the filter.")

    rng = random.Random(args.seed)
    n = min(args.n, len(facts))
    sample = rng.sample(facts, n)

    chunk_index = _all_chunks_by_id()
    chunks_list = list(chunk_index.values())

    print(f"Sampled {n} of {len(facts)} prose facts (seed={args.seed}).")
    type_counts = Counter(f.fact_type.value for f in sample)
    print(f"Sample type distribution: {dict(type_counts)}")
    expected_types = {
        "operational_metric",
        "forward_guidance",
        "strategic_claim",
        "causal_explanation",
        "risk_disclosure",
        "accounting_policy",
    }
    missing = expected_types - set(type_counts.keys())
    if missing:
        print(
            f"  NOTE: sample missing types {sorted(missing)} — "
            "increase --n or rerun with a different --seed for full coverage."
        )

    for i, fact in enumerate(sample, start=1):
        chunk = _find_chunk_for_fact(fact, chunks_list)
        _print_fact(i, fact, chunk)

    print(f"\n{'=' * 78}")
    print(f"END OF SAMPLE ({n} facts)")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()

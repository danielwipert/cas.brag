"""Block 5 runner: prose fact extraction on the dev subset.

For each document in ``data/dev_subset/``, runs the section-aware chunker
and feeds each chunk through the OpenRouter prose-fact extractor (DeepSeek
Chat). Validates each fact's verbatim_anchor against the chunk text, drops
any that fail, mints deterministic ``F-PROSE-{6-digit}`` IDs, and writes
the survivors to ``data/fact_store/dev_prose_facts.jsonl``.

Run from the repo root::

    python -m scripts.build_dev_prose_facts                # full run
    python -m scripts.build_dev_prose_facts --smoke        # 2 chunks per doc
    python -m scripts.build_dev_prose_facts --doc nflx-q4-2023-letter

Build log goes to ``data/logs/dev_prose_build.json``.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from agents.llm_client import OpenRouterClient
from ingestion.chunker.section_aware import iter_dev_subset_documents, chunk_document
from ingestion.prose.extract import (
    DocumentMeta,
    ExtractionStats,
    FactIdMinter,
    extract_facts_from_document,
    post_filter_facts,
    save_facts_jsonl,
)
from schemas.records import FactRecord


DEV_SUBSET_ROOT = Path("data/dev_subset")
DEV_PULL_LOG = Path("data/logs/dev_subset_pull.json")
JSONL_PATH = Path("data/fact_store/dev_prose_facts.jsonl")
BUILD_LOG_PATH = Path("data/logs/dev_prose_build.json")

# Per-document metadata. Transcript date isn't in the pull log, so it's
# hardcoded from the transcript header (FQ1 2024 call, Apr 18 2024).
_DOC_META: dict[str, DocumentMeta] = {
    "nflx-10q-2024-q3": DocumentMeta(
        document_id="nflx-10q-2024-q3",
        asserter_default="Netflix",
        assertion_date="2024-10-18",
    ),
    "nflx-q4-2023-letter": DocumentMeta(
        document_id="nflx-q4-2023-letter",
        asserter_default="Netflix",
        assertion_date="2024-01-23",
    ),
    "nflx-q1-2024-transcript": DocumentMeta(
        document_id="nflx-q1-2024-transcript",
        # Transcript chunks contain inline "Speaker Name:" labels — the
        # extractor prompt instructs the LLM to read those. When a chunk
        # straddles a speaker boundary or has no inline label, the LLM is
        # told to emit "unknown_speaker" rather than guess. We mirror that
        # convention as the runner-side fallback.
        asserter_default="unknown_speaker",
        assertion_date="2024-04-18",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Process only the first 2 chunks of each document (quick sanity check).",
    )
    p.add_argument(
        "--doc",
        default=None,
        help="Restrict to a single document_id (e.g. nflx-q4-2023-letter).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override LLM model slug (default: client default = deepseek/deepseek-chat).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    docs = iter_dev_subset_documents(DEV_SUBSET_ROOT)
    if not docs:
        raise SystemExit(f"No documents under {DEV_SUBSET_ROOT}.")
    if args.doc:
        docs = [d for d in docs if d.document_id == args.doc]
        if not docs:
            raise SystemExit(f"No document matched --doc {args.doc!r}.")

    client = OpenRouterClient()
    minter = FactIdMinter()
    stats = ExtractionStats()
    all_facts: list[FactRecord] = []
    per_doc_kept: dict[str, int] = {}
    per_doc_chunks: dict[str, int] = {}

    for doc in docs:
        meta = _DOC_META.get(doc.document_id)
        if meta is None:
            print(f"!! No DocumentMeta entry for {doc.document_id}; skipping.")
            continue
        chunks = chunk_document(doc)
        if args.smoke:
            chunks = chunks[:2]
        per_doc_chunks[doc.document_id] = len(chunks)
        print(f"\n=== {doc.document_id}: {len(chunks)} chunks ===")
        facts = extract_facts_from_document(
            chunks, meta, client=client, minter=minter, stats=stats, model=args.model
        )
        per_doc_kept[doc.document_id] = len(facts)
        all_facts.extend(facts)
        print(f"  -> kept {len(facts)} facts from this document")

    pre_filter_count = len(all_facts)
    all_facts, post_filter_drops = post_filter_facts(all_facts)

    type_counter: Counter[str] = Counter(f.fact_type.value for f in all_facts)
    asserter_counter: Counter[str] = Counter(f.asserter for f in all_facts)
    period_counter: Counter[str] = Counter(f.period or "<none>" for f in all_facts)

    save_facts_jsonl(all_facts, JSONL_PATH)

    print("\n=========== summary ===========")
    print(f"chunks processed:      {stats.n_chunks}")
    print(f"chunks with >=1 fact:  {stats.n_chunks_with_facts}")
    print(f"raw facts (LLM):       {stats.n_raw_facts}")
    print(f"kept (post-validate):  {pre_filter_count}")
    print(f"  anchor ws-recovered: {stats.n_anchor_ws_recovered}")
    print(f"post-filter drops:     {sum(post_filter_drops.values())} {dict(post_filter_drops)}")
    print(f"final kept facts:      {len(all_facts)}")
    print(f"  by type:             {dict(type_counter)}")
    print(f"  by asserter:         {dict(asserter_counter.most_common(8))}")
    print(f"  by period (top 8):   {dict(period_counter.most_common(8))}")
    print(f"dropped — anchor:      {stats.n_dropped_anchor}")
    print(f"dropped — fin metric:  {stats.n_dropped_financial_metric}")
    print(f"dropped — low conf:    {stats.n_dropped_low_confidence}")
    print(f"dropped — schema:      {stats.n_dropped_schema}")
    print(f"dropped — other:       {stats.n_dropped_other}")
    print(f"LLM errors:            {stats.n_llm_errors}")
    print(f"\nFacts -> {JSONL_PATH}")

    BUILD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    BUILD_LOG_PATH.write_text(
        json.dumps(
            {
                "smoke": bool(args.smoke),
                "doc_filter": args.doc,
                "model_override": args.model,
                "stats": stats.to_dict(),
                "post_filter_drops": post_filter_drops,
                "n_pre_filter": pre_filter_count,
                "n_final_kept": len(all_facts),
                "per_doc_chunks": per_doc_chunks,
                "per_doc_kept": per_doc_kept,
                "fact_type_counts": dict(type_counter),
                "asserter_counts": dict(asserter_counter),
                "period_counts": dict(period_counter),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Build log -> {BUILD_LOG_PATH}")


if __name__ == "__main__":
    main()

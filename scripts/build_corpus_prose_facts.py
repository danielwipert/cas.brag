"""Block 6e: prose fact extraction across the full corpus.

Walks ``data/corpus/<document_id>/*.txt`` for every document in the
manifest (filings + transcripts), chunks each one, and feeds chunks
through the OpenRouter DeepSeek prose-fact extractor. Survivors are
written one document at a time to::

    data/fact_store/prose_facts/<document_id>.jsonl

After all documents complete, the per-doc files are concatenated +
post-filtered (GAAP-keyword + corporate-action) into the canonical::

    data/fact_store/prose_facts.jsonl

A build log goes to ``data/logs/corpus_prose_build.json``.

Restart-safety: each per-doc JSONL is written atomically (.tmp + rename).
On restart, any document whose ``<document_id>.jsonl`` already exists is
skipped — pass ``--force`` to re-extract. Fact IDs are doc-scoped
(``F-PROSE-{document_id}-{idx:04d}``) so re-runs do not collide with
the rest of the corpus.

CLI::

    python -m scripts.build_corpus_prose_facts                      # full run
    python -m scripts.build_corpus_prose_facts --only doc1,doc2     # subset
    python -m scripts.build_corpus_prose_facts --limit 2            # first 2 docs
    python -m scripts.build_corpus_prose_facts --smoke              # 2 chunks per doc
    python -m scripts.build_corpus_prose_facts --force              # redo done docs
    python -m scripts.build_corpus_prose_facts --concatenate-only   # rebuild prose_facts.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.llm_client import OpenRouterClient
from ingestion.chunker.section_aware import (
    chunk_document,
    iter_dev_subset_documents,
    _load_document,
)
from ingestion.prose.extract import (
    DocumentMeta,
    ExtractionStats,
    FactIdMinter,
    extract_facts_from_document,
    load_facts_jsonl,
    post_filter_facts,
    save_facts_jsonl,
)
from schemas.records import FactRecord


CORPUS_ROOT = REPO_ROOT / "data" / "corpus"
MANIFEST_PATH = REPO_ROOT / "data" / "raw" / "document_manifest.json"
PER_DOC_DIR = REPO_ROOT / "data" / "fact_store" / "prose_facts"
COMBINED_JSONL = REPO_ROOT / "data" / "fact_store" / "prose_facts.jsonl"
LOG_PATH = REPO_ROOT / "data" / "logs" / "corpus_prose_build.json"


# Netflix releases earnings on the 17th-19th of the month historically.
# Used only as a fallback for transcripts whose corresponding 8-K letter
# is outside the manifest window (only nflx-q1-2016-transcript at present).
_QUARTER_RELEASE_FALLBACK: dict[int, tuple[int, int, int]] = {
    1: (4, 18, 0),   # Q1: April 18 of Y
    2: (7, 18, 0),   # Q2: July 18 of Y
    3: (10, 18, 0),  # Q3: October 18 of Y
    4: (1, 18, 1),   # Q4: January 18 of Y+1
}


class _DocPrefixedMinter(FactIdMinter):
    """Mints ``F-PROSE-{document_id}-{idx:04d}``. Per-doc — instantiate fresh
    for each document and the IDs stay non-colliding across the corpus."""

    def __init__(self, document_id: str) -> None:
        super().__init__(start=1)
        self._doc_id = document_id

    def mint(self) -> str:  # type: ignore[override]
        fid = f"F-PROSE-{self._doc_id}-{self._next:04d}"
        self._next += 1
        return fid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--only",
        default="",
        help="Comma-separated document_ids to limit the run to.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N documents (after --only filtering).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Process only the first 2 chunks of each document (quick sanity check). "
             "Per-doc JSONL is written under a .smoke suffix to avoid polluting "
             "real outputs; --concatenate-only ignores smoke files.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-extract documents whose per-doc JSONL already exists.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override LLM model slug (default: deepseek/deepseek-chat).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent LLM calls per document. 1 = sequential "
             "(legacy). 8 is a reasonable default for the corpus run "
             "(~5x faster wall time, well under OpenRouter rate limits).",
    )
    p.add_argument(
        "--concatenate-only",
        action="store_true",
        help="Skip extraction; only merge existing per-doc JSONLs into the "
             "combined prose_facts.jsonl and apply post-filter.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-doc metadata resolution
# ---------------------------------------------------------------------------


def _quarter_release_date(fiscal_period: str) -> date:
    """Fallback release-date derivation for transcripts outside the manifest
    window. ``fiscal_period`` is e.g. "2016Q1"."""
    m = re.match(r"^(\d{4})Q([1-4])$", fiscal_period)
    if not m:
        raise ValueError(f"Cannot derive release date for fiscal_period={fiscal_period!r}")
    year, qtr = int(m.group(1)), int(m.group(2))
    month, day, year_offset = _QUARTER_RELEASE_FALLBACK[qtr]
    return date(year + year_offset, month, day)


def _build_doc_meta(manifest: dict) -> dict[str, DocumentMeta]:
    """Map document_id -> DocumentMeta for every document in the manifest
    (filings + transcripts), excluding letter_unmapped."""
    out: dict[str, DocumentMeta] = {}

    # Filings: assertion_date = filing_date, asserter = "Netflix".
    letter_by_period: dict[str, date] = {}
    for f in manifest.get("filings", []):
        if f.get("document_kind") == "letter_unmapped":
            continue
        if not f.get("local_path"):
            continue
        did = f["document_id"]
        out[did] = DocumentMeta(
            document_id=did,
            asserter_default="Netflix",
            assertion_date=f["filing_date"],
        )
        if f.get("document_kind") == "letter" and f.get("fiscal_period"):
            letter_by_period[f["fiscal_period"]] = date.fromisoformat(f["filing_date"])

    # Transcripts: assertion_date = matching letter's filing_date (the
    # earnings-release 8-K is filed the same day as the call); fallback to
    # the canonical Q-release date convention if no letter is in window.
    for t in manifest.get("transcripts", []):
        if not t.get("local_path"):
            continue
        did = t["document_id"]
        fiscal_period = t.get("fiscal_period")
        rel_date = letter_by_period.get(fiscal_period) if fiscal_period else None
        if rel_date is None and fiscal_period:
            try:
                rel_date = _quarter_release_date(fiscal_period)
            except ValueError:
                continue
        if rel_date is None:
            continue
        out[did] = DocumentMeta(
            document_id=did,
            # Transcript chunks contain inline "Speaker:" labels; the LLM
            # prompt parses those. Use the dev convention as the fallback.
            asserter_default="unknown_speaker",
            assertion_date=rel_date.isoformat(),
        )
    return out


# ---------------------------------------------------------------------------
# Per-doc atomic write
# ---------------------------------------------------------------------------


def _per_doc_path(document_id: str, smoke: bool) -> Path:
    suffix = ".smoke.jsonl" if smoke else ".jsonl"
    return PER_DOC_DIR / f"{document_id}{suffix}"


def _save_per_doc_atomic(facts: list[FactRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in facts:
            f.write(rec.model_dump_json() + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Concatenation + post-filter
# ---------------------------------------------------------------------------


def _concatenate_and_filter(write_combined: bool = True) -> dict:
    """Read every non-smoke per-doc JSONL, combine, post-filter, write the
    canonical prose_facts.jsonl. Returns stats."""
    if not PER_DOC_DIR.exists():
        raise SystemExit(f"{PER_DOC_DIR} does not exist — nothing to concatenate.")
    files = sorted(p for p in PER_DOC_DIR.glob("*.jsonl") if not p.name.endswith(".smoke.jsonl"))
    if not files:
        raise SystemExit(f"No per-doc JSONLs found in {PER_DOC_DIR}.")

    all_facts: list[FactRecord] = []
    per_doc_kept: dict[str, int] = {}
    for fp in files:
        facts = load_facts_jsonl(fp)
        per_doc_kept[fp.stem] = len(facts)
        all_facts.extend(facts)

    pre = len(all_facts)
    all_facts, drops = post_filter_facts(all_facts)
    if write_combined:
        save_facts_jsonl(all_facts, COMBINED_JSONL)
    return {
        "n_documents_merged": len(files),
        "n_pre_filter": pre,
        "n_after_post_filter": len(all_facts),
        "post_filter_drops": drops,
        "per_document_kept": per_doc_kept,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if args.concatenate_only:
        stats = _concatenate_and_filter()
        print("Concatenate-only mode:")
        print(f"  merged {stats['n_documents_merged']} per-doc JSONLs")
        print(f"  pre-filter facts:  {stats['n_pre_filter']}")
        print(f"  after post-filter: {stats['n_after_post_filter']}")
        print(f"  drops: {stats['post_filter_drops']}")
        print(f"  combined -> {COMBINED_JSONL}")
        return

    if not MANIFEST_PATH.exists():
        raise SystemExit(f"{MANIFEST_PATH} not found.")
    if not CORPUS_ROOT.exists():
        raise SystemExit(
            f"{CORPUS_ROOT} not found — run scripts/build_corpus_sections.py first."
        )

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    doc_meta = _build_doc_meta(manifest)

    # Discover docs in deterministic order: iter_dev_subset_documents returns
    # them sorted by document_id (alphabetic). That's stable across reruns.
    docs = iter_dev_subset_documents(CORPUS_ROOT)
    if not docs:
        raise SystemExit(f"No documents under {CORPUS_ROOT}.")

    only: set[str] = set()
    if args.only:
        only = {s.strip() for s in args.only.split(",") if s.strip()}
        unknown = only - {d.document_id for d in docs}
        if unknown:
            raise SystemExit(f"--only mentions unknown document_ids: {sorted(unknown)}")

    selected = [d for d in docs if not only or d.document_id in only]
    missing_meta = [d.document_id for d in selected if d.document_id not in doc_meta]
    if missing_meta:
        raise SystemExit(
            f"No DocumentMeta for {missing_meta} — check the manifest "
            f"and _build_doc_meta logic."
        )

    if args.limit is not None:
        selected = selected[: args.limit]

    # Skip already-done docs unless --force.
    to_process = []
    for d in selected:
        out_path = _per_doc_path(d.document_id, args.smoke)
        if out_path.exists() and not args.force:
            print(f"skip {d.document_id} (already extracted -> {out_path.relative_to(REPO_ROOT)})")
            continue
        to_process.append(d)

    if not to_process:
        print("All selected documents are already extracted. "
              "Run with --concatenate-only to rebuild the combined JSONL.")
        return

    print(f"\nProcessing {len(to_process)} document(s); model="
          f"{args.model or 'deepseek/deepseek-chat (client default)'}; "
          f"smoke={args.smoke}; workers={args.workers}")
    client = OpenRouterClient()

    aggregate_stats = ExtractionStats()
    per_doc_summaries: list[dict] = []

    for i, doc in enumerate(to_process, start=1):
        meta = doc_meta[doc.document_id]
        chunks = chunk_document(doc)
        if args.smoke:
            chunks = chunks[:2]
        print(f"\n[{i}/{len(to_process)}] {doc.document_id} ({len(chunks)} chunks, "
              f"asserter_default={meta.asserter_default!r}, "
              f"assertion_date={meta.assertion_date})")

        per_doc_stats = ExtractionStats()
        minter = _DocPrefixedMinter(doc.document_id)
        try:
            facts = extract_facts_from_document(
                chunks, meta,
                client=client, minter=minter, stats=per_doc_stats,
                model=args.model,
                progress=True,
                max_workers=args.workers,
            )
        except KeyboardInterrupt:
            print("\nInterrupted — partial in-flight doc not saved. "
                  "Previously completed docs remain on disk.")
            raise

        out_path = _per_doc_path(doc.document_id, args.smoke)
        _save_per_doc_atomic(facts, out_path)
        print(f"  -> {len(facts)} facts written to {out_path.relative_to(REPO_ROOT)}")
        per_doc_summaries.append({
            "document_id": doc.document_id,
            "n_chunks": per_doc_stats.n_chunks,
            "n_kept": per_doc_stats.n_kept,
            "n_dropped_anchor": per_doc_stats.n_dropped_anchor,
            "n_dropped_schema": per_doc_stats.n_dropped_schema,
            "n_dropped_financial_metric": per_doc_stats.n_dropped_financial_metric,
            "n_dropped_low_confidence": per_doc_stats.n_dropped_low_confidence,
            "n_anchor_ws_recovered": per_doc_stats.n_anchor_ws_recovered,
            "n_llm_errors": per_doc_stats.n_llm_errors,
        })
        # Roll up into aggregate.
        for k, v in per_doc_stats.to_dict().items():
            setattr(aggregate_stats, k, getattr(aggregate_stats, k) + v)

    print("\n=== Aggregate (this run) ===")
    print(f"  chunks processed:   {aggregate_stats.n_chunks}")
    print(f"  chunks with facts:  {aggregate_stats.n_chunks_with_facts}")
    print(f"  raw facts:          {aggregate_stats.n_raw_facts}")
    print(f"  kept:               {aggregate_stats.n_kept}")
    print(f"  ws-recovered anchors: {aggregate_stats.n_anchor_ws_recovered}")
    print(f"  dropped anchor:     {aggregate_stats.n_dropped_anchor}")
    print(f"  dropped fin_metric: {aggregate_stats.n_dropped_financial_metric}")
    print(f"  dropped low_conf:   {aggregate_stats.n_dropped_low_confidence}")
    print(f"  dropped schema:     {aggregate_stats.n_dropped_schema}")
    print(f"  LLM errors:         {aggregate_stats.n_llm_errors}")

    # Rebuild the combined JSONL from all per-doc files on disk (smoke files
    # excluded). Skipped in --smoke mode since the smoke outputs aren't part
    # of the canonical corpus state.
    if not args.smoke:
        combined_stats = _concatenate_and_filter()
        print("\n=== Combined corpus ===")
        print(f"  merged {combined_stats['n_documents_merged']} per-doc JSONLs")
        print(f"  pre-filter facts:  {combined_stats['n_pre_filter']}")
        print(f"  after post-filter: {combined_stats['n_after_post_filter']}")
        print(f"  drops: {combined_stats['post_filter_drops']}")
        print(f"  combined -> {COMBINED_JSONL.relative_to(REPO_ROOT)}")
    else:
        combined_stats = None

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(
        json.dumps(
            {
                "smoke": bool(args.smoke),
                "force": bool(args.force),
                "only": sorted(only) if only else None,
                "limit": args.limit,
                "model_override": args.model,
                "aggregate_stats": aggregate_stats.to_dict(),
                "per_document": per_doc_summaries,
                "combined": combined_stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nBuild log -> {LOG_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

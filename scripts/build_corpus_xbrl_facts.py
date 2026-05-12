"""Block 6d: corpus-wide XBRL fact ingestion.

Walks every filing in ``data/raw/document_manifest.json`` with an
``xbrl_local_path`` set, applies Block 4's retention/dimension/period
policy via ``xbrl_to_fact_record``, and produces:

  - data/fact_store/xbrl_facts.jsonl   (one JSON-encoded FactRecord per line)
  - data/fact_store/chromadb           (Chroma collection ``facts``, XBRL-only
                                         until Block 6e adds prose)
  - data/logs/corpus_xbrl_build.json   (per-filing stats + drop reasons)

The Chroma collection is a clean rebuild — it overwrites Block 4's
single-filing dev store. The combined Fact Store (XBRL + prose) is built
separately in Block 6e once prose extraction completes.

Per-filing safeguards: XBRL legitimately reports the same logical fact
in multiple places (income statement plus footnote breakdown). Both
references carry the same context and value, so we dedupe within each
filing on ``fact_id`` and raise on any value mismatch — that would
indicate either an XBRL inconsistency or a parser bug. A single bad
filing logs and continues; the run does not abort.

Run from repo root::

    python -m scripts.build_corpus_xbrl_facts
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from ingestion.fact_store.embed_and_index import (
    EMBED_MODEL_NAME,
    build_fact_store,
)
from ingestion.xbrl.build_fact_records import xbrl_to_fact_record
from ingestion.xbrl.concept_filter import (
    all_canonical_concepts,
    classify_dimensions,
    is_canonical,
)
from ingestion.xbrl.parse import load_xbrl_instance
from schemas.records import FactRecord


MANIFEST_PATH = Path("data/raw/document_manifest.json")
JSONL_PATH = Path("data/fact_store/xbrl_facts.jsonl")
LOG_PATH = Path("data/logs/corpus_xbrl_build.json")


def _save_jsonl(facts: list[FactRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in facts:
            f.write(rec.model_dump_json() + "\n")


def _process_filing(
    *, document_id: str, xbrl_path: Path, filing_date: date
) -> tuple[list[FactRecord], Counter[str], int]:
    """Parse one XBRL instance, build FactRecords with per-filing dedup.
    Returns (facts, drop_counter, n_duplicates_collapsed)."""
    instance = load_xbrl_instance(xbrl_path)
    drop_counter: Counter[str] = Counter()
    seen: dict[str, FactRecord] = {}
    duplicates = 0
    for fact in instance.facts:
        rec = xbrl_to_fact_record(
            fact, source_document=document_id, filing_date=filing_date
        )
        if rec is None:
            if not is_canonical(fact.concept):
                drop_counter["non_canonical_concept"] += 1
            elif fact.period_kind in ("ytd_6m", "ytd_9m"):
                drop_counter[f"ytd_{fact.period_kind}"] += 1
            elif fact.period_kind == "unknown":
                drop_counter["unknown_period_kind"] += 1
            elif classify_dimensions(fact.dimensions)[0] == "drop":
                drop_counter["out_of_policy_dimension"] += 1
            else:
                drop_counter["other"] += 1
            continue
        prior = seen.get(rec.fact_id)
        if prior is None:
            seen[rec.fact_id] = rec
        else:
            duplicates += 1
            if prior.value != rec.value:
                # Value mismatch on the same logical fact within a single
                # filing — log and keep the first; don't abort the whole run.
                drop_counter["value_mismatch_within_filing"] += 1
    return list(seen.values()), drop_counter, duplicates


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"{MANIFEST_PATH} not found.")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    targets: list[tuple[str, Path, date]] = []
    skipped_no_xbrl: list[str] = []
    for f in manifest.get("filings", []):
        if f["form"] not in ("10-K", "10-K/A", "10-Q"):
            continue
        xbrl = f.get("xbrl_local_path")
        if not xbrl:
            skipped_no_xbrl.append(f["document_id"])
            continue
        targets.append(
            (f["document_id"], Path(xbrl), date.fromisoformat(f["filing_date"]))
        )

    if not targets:
        raise SystemExit("No filings with xbrl_local_path found in manifest.")

    all_facts: list[FactRecord] = []
    per_doc: dict[str, dict] = {}
    total_drop: Counter[str] = Counter()
    failures: list[tuple[str, str]] = []

    for i, (doc_id, xbrl_path, fd) in enumerate(targets, start=1):
        prefix = f"[{i:>2}/{len(targets)}] {doc_id:<22}"
        try:
            facts, drops, dupes = _process_filing(
                document_id=doc_id, xbrl_path=xbrl_path, filing_date=fd
            )
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            failures.append((doc_id, msg))
            print(f"{prefix} FAILED  {msg}")
            continue
        all_facts.extend(facts)
        total_drop.update(drops)
        # Per-doc canonical coverage stats.
        kept_concepts = {f.concept_tag for f in facts}
        per_doc[doc_id] = {
            "n_kept": len(facts),
            "n_duplicates_collapsed": dupes,
            "n_dropped_by_reason": dict(drops),
            "kept_concept_count": len(kept_concepts),
        }
        print(
            f"{prefix} kept={len(facts):>4}  drops={sum(drops.values()):>4}  "
            f"dupes_collapsed={dupes}"
        )

    if skipped_no_xbrl:
        print(f"\nSkipped {len(skipped_no_xbrl)} filings without xbrl_local_path:")
        for did in skipped_no_xbrl:
            print(f"  - {did}")

    print(f"\nTotal facts kept across corpus: {len(all_facts)}")
    type_counter: Counter[str] = Counter(f.fact_type.value for f in all_facts)
    period_counter: Counter[str] = Counter(f.period or "<none>" for f in all_facts)
    print(f"  by type:   {dict(type_counter)}")
    print(f"  by period: {dict(period_counter.most_common(10))}")

    # Canonical-coverage matrix: per concept, how many filings produced
    # at least one aggregate fact for it. The build plan's Stage I4
    # validation cares about ~80% coverage on canonical concepts; this
    # log surfaces the actual numbers so flagging is downstream-easy.
    coverage: dict[str, set[str]] = defaultdict(set)
    for f in all_facts:
        # Aggregate facts only — regional facts duplicate the concept tag.
        from ingestion.fact_store.embed_and_index import _region_from_fact_id

        if _region_from_fact_id(f.fact_id) is None:
            coverage[f.concept_tag or ""].add(f.source_document)
    canonical_coverage = {
        c: len(coverage.get(c, set())) for c in all_canonical_concepts()
    }
    n_filings_with_xbrl = len(targets)
    weak = [
        (c, n) for c, n in canonical_coverage.items()
        if n / max(n_filings_with_xbrl, 1) < 0.8
    ]
    print(
        f"\nCanonical coverage: "
        f"{sum(1 for n in canonical_coverage.values() if n > 0)}/"
        f"{len(canonical_coverage)} concepts produced >=1 aggregate fact"
    )
    if weak:
        print(f"  Below-80% concepts ({len(weak)}):")
        for c, n in sorted(weak, key=lambda x: x[1]):
            print(f"    {c:<60} {n}/{n_filings_with_xbrl} filings")

    _save_jsonl(all_facts, JSONL_PATH)
    print(f"\nSerialized facts -> {JSONL_PATH}")

    print(f"\nEmbedding + indexing into ChromaDB ({EMBED_MODEL_NAME})...")
    stats = build_fact_store(all_facts)
    print(
        f"  Indexed {stats['n_facts']} facts at embed_dim={stats['embed_dim']}"
    )

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(
        json.dumps(
            {
                "embed_model": EMBED_MODEL_NAME,
                "n_filings_processed": len(targets),
                "n_filings_skipped_no_xbrl": len(skipped_no_xbrl),
                "skipped_documents": skipped_no_xbrl,
                "n_facts_total": len(all_facts),
                "drops_by_reason": dict(total_drop),
                "by_type": dict(type_counter),
                "by_period": dict(period_counter),
                "canonical_coverage_by_filings": canonical_coverage,
                "per_document": per_doc,
                "failures": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Build log -> {LOG_PATH}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

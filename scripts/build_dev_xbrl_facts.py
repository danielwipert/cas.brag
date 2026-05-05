"""Block 4 runner: ingest XBRL facts from the dev subset.

Reads ``data/raw/nflx-10q-2024-q3.xbrl.xml`` (the on-disk XBRL instance from
Block 2), applies Block 4 retention/dimension/period policy, builds
FactRecord objects, writes them to ``data/fact_store/dev_xbrl_facts.jsonl``,
embeds them with BGE-small, and persists them to the ``facts`` Chroma
collection.

Run from the repo root::

    python -m scripts.build_dev_xbrl_facts
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

from ingestion.fact_store.embed_and_index import (
    EMBED_MODEL_NAME,
    build_fact_store,
)
from ingestion.xbrl.build_fact_records import xbrl_to_fact_record
from ingestion.xbrl.parse import load_xbrl_instance
from schemas.records import FactRecord


DEV_XBRL_PATH = Path("data/raw/nflx-10q-2024-q3.xbrl.xml")
DEV_PULL_LOG = Path("data/logs/dev_subset_pull.json")
JSONL_PATH = Path("data/fact_store/dev_xbrl_facts.jsonl")
BUILD_LOG_PATH = Path("data/logs/dev_xbrl_build.json")
DEV_DOCUMENT_ID = "nflx-10q-2024-q3"


def _read_filing_date() -> date:
    """Read the filing date from Block 2's pull log; fall back to the
    XBRL DocumentPeriodEndDate if the log is missing."""
    if DEV_PULL_LOG.exists():
        meta = json.loads(DEV_PULL_LOG.read_text(encoding="utf-8"))
        for entry in meta.values():
            if isinstance(entry, dict) and entry.get("document_id") == DEV_DOCUMENT_ID:
                fd = entry.get("filing_date")
                if fd:
                    return date.fromisoformat(fd)
    raise SystemExit(
        f"Could not determine filing date for {DEV_DOCUMENT_ID}. "
        f"Expected a record in {DEV_PULL_LOG}."
    )


def _save_jsonl(facts: list[FactRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in facts:
            f.write(rec.model_dump_json() + "\n")


def main() -> None:
    if not DEV_XBRL_PATH.exists():
        raise SystemExit(f"Missing XBRL instance at {DEV_XBRL_PATH}.")

    filing_date = _read_filing_date()
    instance = load_xbrl_instance(DEV_XBRL_PATH)

    print(
        f"Parsed XBRL instance: {len(instance.contexts)} contexts, "
        f"{len(instance.units)} units, {len(instance.facts)} numeric facts"
    )

    drop_counter: Counter[str] = Counter()
    # XBRL legitimately reports the same logical fact in multiple places
    # (e.g. Revenues in the income statement and again in a revenue
    # recognition footnote). Both reference the same context and carry the
    # same value. Dedupe by BRAG fact_id and warn loudly if any two
    # occurrences disagree on the numeric value (would indicate a parse bug
    # or an inconsistent filing).
    seen: dict[str, FactRecord] = {}
    duplicate_count = 0
    for f in instance.facts:
        rec = xbrl_to_fact_record(
            f,
            source_document=DEV_DOCUMENT_ID,
            filing_date=filing_date,
        )
        if rec is None:
            # Diagnose why it was dropped (best-effort, for the build log).
            from ingestion.xbrl.concept_filter import classify_dimensions, is_canonical

            if not is_canonical(f.concept):
                drop_counter["non_canonical_concept"] += 1
            elif f.period_kind in ("ytd_6m", "ytd_9m"):
                drop_counter[f"ytd_{f.period_kind}"] += 1
            elif f.period_kind == "unknown":
                drop_counter["unknown_period_kind"] += 1
            elif classify_dimensions(f.dimensions)[0] == "drop":
                drop_counter["out_of_policy_dimension"] += 1
            else:
                drop_counter["other"] += 1
            continue
        prior = seen.get(rec.fact_id)
        if prior is None:
            seen[rec.fact_id] = rec
        else:
            duplicate_count += 1
            if prior.value != rec.value:
                raise SystemExit(
                    f"VALUE MISMATCH on {rec.fact_id}: "
                    f"first={prior.value!r}, later={rec.value!r}. "
                    "This indicates either an XBRL inconsistency or a parser bug."
                )
    keep: list[FactRecord] = list(seen.values())

    period_counter: Counter[str] = Counter(r.period or "<none>" for r in keep)
    section_counter: Counter[str] = Counter(r.source_section for r in keep)
    region_counter: Counter[str] = Counter()
    for r in keep:
        for tag in ("UCAN", "EMEA", "LATAM", "APAC"):
            if r.fact_id.endswith(f"-{tag}"):
                region_counter[tag] += 1
                break
        else:
            region_counter["aggregate"] += 1

    print(
        f"\nKept {len(keep)} unique fact records "
        f"(filing_date={filing_date}, deduped {duplicate_count} XBRL replicas)"
    )
    print(f"  Drops by reason: {dict(drop_counter)}")
    print(f"  By statement section: {dict(section_counter)}")
    print(f"  By region: {dict(region_counter)}")
    print(f"  By period: {dict(period_counter.most_common(10))}")

    _save_jsonl(keep, JSONL_PATH)
    print(f"\nSerialized facts -> {JSONL_PATH}")

    stats = build_fact_store(keep)
    print(
        f"Indexed {stats['n_facts']} facts at embed_dim={stats['embed_dim']} "
        f"(model={EMBED_MODEL_NAME})"
    )

    BUILD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    BUILD_LOG_PATH.write_text(
        json.dumps(
            {
                "embed_model": EMBED_MODEL_NAME,
                "filing_date": filing_date.isoformat(),
                "n_raw_facts": len(instance.facts),
                "n_kept": len(keep),
                "n_xbrl_duplicates_collapsed": duplicate_count,
                "drops_by_reason": dict(drop_counter),
                "by_region": dict(region_counter),
                "by_section": dict(section_counter),
                "by_period": dict(period_counter),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Build log -> {BUILD_LOG_PATH}")


if __name__ == "__main__":
    main()

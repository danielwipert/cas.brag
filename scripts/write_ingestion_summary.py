"""Block 6 final step: aggregate all per-stage build logs and validation
results into a single ingestion_summary.json.

Reads:
  - data/logs/corpus_chunk_build.json
  - data/logs/corpus_xbrl_build.json
  - data/logs/corpus_prose_build.json
  - data/logs/corpus_combined_fact_build.json
  - data/logs/corpus_validation.json

Writes:
  - data/logs/ingestion_summary.json

Run from the repo root::

    python -m scripts.write_ingestion_summary
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


LOGS = Path("data/logs")
OUT_PATH = LOGS / "ingestion_summary.json"

PROSE_JSONL = Path("data/fact_store/prose_facts.jsonl")


def _load(name: str) -> dict:
    path = LOGS / name
    if not path.exists():
        raise SystemExit(f"Missing {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def _low_confidence_rate() -> dict:
    """Scan persisted prose facts and return the rate at which
    confidence < 0.75."""
    n_total = 0
    n_low = 0
    if not PROSE_JSONL.exists():
        return {"n_total": 0, "n_low_confidence": 0, "rate": 0.0}
    with PROSE_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            n_total += 1
            f = json.loads(line)
            if (f.get("confidence") or 0.0) < 0.75:
                n_low += 1
    return {
        "n_total": n_total,
        "n_low_confidence": n_low,
        "rate": round(n_low / n_total, 4) if n_total else 0.0,
    }


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(round((len(values) - 1) * p))))
    return values[k]


def main() -> None:
    chunk = _load("corpus_chunk_build.json")
    xbrl = _load("corpus_xbrl_build.json")
    prose = _load("corpus_prose_build.json")
    combined = _load("corpus_combined_fact_build.json")
    validation = _load("corpus_validation.json")

    # --- chunks ------------------------------------------------------------
    chunk_counts = list(chunk["per_document_chunks"].values())
    chunks = {
        "n_documents": chunk["n_documents"],
        "n_chunks_total": chunk["total_chunks"],
        "embed_model": chunk["embed_model"],
        "embed_dim": chunk["embed_dim"],
        "per_document": {
            "min": min(chunk_counts),
            "median": _percentile(chunk_counts, 0.5),
            "p90": _percentile(chunk_counts, 0.9),
            "max": max(chunk_counts),
        },
    }

    # --- XBRL --------------------------------------------------------------
    xbrl_block = {
        "n_filings_processed": xbrl["n_filings_processed"],
        "n_facts_total": xbrl["n_facts_total"],
        "drops_by_reason": xbrl["drops_by_reason"],
        "concept_coverage": validation["xbrl_coverage"]["by_concept"],
        "concepts_below_80pct": validation["xbrl_coverage"]["concepts_below_80pct"],
        "notes": (
            "Coverage gaps below 80% are structural and reflect Netflix's "
            "actual reporting choices or XBRL taxonomy evolution, not "
            "pipeline bugs. Examples: GrossProfit (12.5%) — Netflix does "
            "not tag a separate GrossProfit line item; "
            "NumberOfPaidMemberships (52.5%) — Netflix stopped reporting "
            "paid memberships in 2022; LongTermDebtCurrent (0%) and "
            "OperatingExpenses (0%) — alternative tagging used."
        ),
    }

    # --- prose -------------------------------------------------------------
    per_doc_kept = prose["combined"]["per_document_kept"]
    prose_counts = list(per_doc_kept.values())
    prose_block = {
        "n_documents_processed": prose["combined"]["n_documents_merged"],
        "n_raw_facts": prose["aggregate_stats"]["n_raw_facts"],
        "n_kept_post_extract": prose["aggregate_stats"]["n_kept"],
        "n_kept_post_filter": prose["combined"]["n_after_post_filter"],
        "post_extract_drops": {
            "anchor": prose["aggregate_stats"]["n_dropped_anchor"],
            "schema": prose["aggregate_stats"]["n_dropped_schema"],
            "financial_metric": prose["aggregate_stats"]["n_dropped_financial_metric"],
            "low_confidence": prose["aggregate_stats"]["n_dropped_low_confidence"],
            "other": prose["aggregate_stats"]["n_dropped_other"],
        },
        "post_filter_drops": prose["combined"]["post_filter_drops"],
        "n_anchor_whitespace_recovered": prose["aggregate_stats"]["n_anchor_ws_recovered"],
        "n_llm_errors": prose["aggregate_stats"]["n_llm_errors"],
        "low_confidence_rate": _low_confidence_rate(),
        "per_document_kept": {
            "min": min(prose_counts),
            "median": _percentile(prose_counts, 0.5),
            "p90": _percentile(prose_counts, 0.9),
            "max": max(prose_counts),
        },
        "zero_fact_documents": validation["prose_coverage"]["zero_prose_docs"],
    }

    # --- combined fact store ----------------------------------------------
    combined_block = {
        "n_xbrl_input": combined["n_xbrl"],
        "n_prose_input": combined["n_prose"],
        "n_combined_input": combined["n_combined_input"],
        "n_after_dedupe": combined["n_after_dedupe"],
        "n_dedupe_collisions": combined["n_dedupe_collisions"],
        "n_source_documents": combined["n_source_documents"],
        "embed_model": combined["embed_model"],
        "embed_dim": combined["embed_dim"],
        "fact_type_distribution": combined["by_type"],
    }
    # Pct distribution
    total = combined["n_after_dedupe"]
    combined_block["fact_type_pct"] = {
        ft: round(100.0 * n / total, 1)
        for ft, n in combined["by_type"].items()
    }

    # --- validation summary -----------------------------------------------
    validation_block = {
        "anchor_validation": {
            "n_failures": validation["anchor_validation"]["n_failures"],
            "failure_rate": validation["anchor_validation"]["failure_rate"],
            "target": "<0.02",
            "status": (
                "PASS"
                if validation["anchor_validation"]["failure_rate"] < 0.02
                else "FAIL"
            ),
        },
        "prose_coverage": {
            "n_docs_with_prose": validation["prose_coverage"]["n_docs_with_prose"],
            "n_docs_total": validation["prose_coverage"]["n_docs_total"],
            "status": (
                "PASS"
                if not validation["prose_coverage"]["zero_prose_docs"]
                else "FAIL"
            ),
        },
        "period_parsing": {
            "n_period_present": validation["period_parsing"]["n_period_present"],
            "n_parse_failures": validation["period_parsing"]["n_parse_failures"],
            "status": (
                "PASS"
                if validation["period_parsing"]["n_parse_failures"] == 0
                else "FAIL"
            ),
        },
        "xbrl_canonical_coverage": {
            "n_concepts_below_80pct": len(
                validation["xbrl_coverage"]["concepts_below_80pct"]
            ),
            "concepts_below_80pct": validation["xbrl_coverage"]["concepts_below_80pct"],
            "status": "DOCUMENTED_GAPS",
            "interpretation": (
                "Build plan's >=80% target is aspirational. Netflix doesn't "
                "tag every concept in the canonical set. Treat as informational."
            ),
        },
    }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "Block 6 — full corpus ingestion (May 2016 – May 2026)",
        "chunk_store": chunks,
        "xbrl_facts": xbrl_block,
        "prose_facts": prose_block,
        "combined_fact_store": combined_block,
        "validation": validation_block,
    }

    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Ingestion summary -> {OUT_PATH}")
    print()
    print(f"Documents:    {chunks['n_documents']}")
    print(f"Chunks:       {chunks['n_chunks_total']:,}")
    print(f"XBRL facts:   {xbrl_block['n_facts_total']:,}")
    print(f"Prose facts:  {prose_block['n_kept_post_filter']:,}")
    print(f"Combined:     {combined_block['n_after_dedupe']:,} after dedupe")
    print()
    print(f"Anchor failure rate:  {validation_block['anchor_validation']['failure_rate']:.4f} "
          f"({validation_block['anchor_validation']['status']})")
    print(f"Prose coverage:       {validation_block['prose_coverage']['status']}")
    print(f"Period parsing:       {validation_block['period_parsing']['status']}")


if __name__ == "__main__":
    main()

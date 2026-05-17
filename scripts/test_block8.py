"""Block 8 smoke test: drive the Retriever with real Planner slots from
Block 7's test output, check the done-when criteria from the build
plan, and log results to data/logs/block8_test.json for review.

Criteria checked per slot:

* Period-filtered slot's candidates all fall within the constraint
  (this is the load-bearing v3 behavior).
* Vector and BM25 channels return non-identical results (Jaccard
  overlap < 1.0).
* RRF score is monotonically decreasing across the fused candidate
  list.
* pass_origin is set correctly.

Run from the repo root::

    python -m scripts.test_block8
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from agents.retriever.bm25_channel import bm25_search
from agents.retriever.period_filter import period_from_document_id
from agents.retriever.retriever import retrieve, _K_BY_TIER
from agents.retriever.vector_channel import vector_search
from schemas.enums import ComplexityTier, PassOrigin
from schemas.records import EvidenceSlot


BLOCK7_LOG = Path("data/logs/block7_test.json")
OUT_PATH = Path("data/logs/block8_test.json")


def _load_slots() -> list[tuple[str, ComplexityTier, EvidenceSlot]]:
    """Pick 5 representative slots from the Block 7 test output:
    one specific_metric with period_filter, one accounting_policy
    without, one risk_disclosure with period_filter, one
    strategic_position from a complex query, one cross_period_comparison.
    Returns (query_text, tier, slot) tuples."""
    if not BLOCK7_LOG.exists():
        raise SystemExit(
            f"{BLOCK7_LOG} not found. Run scripts.test_block7 first."
        )
    data = json.loads(BLOCK7_LOG.read_text(encoding="utf-8"))

    picks: list[tuple[str, ComplexityTier, EvidenceSlot]] = []
    seen_kinds: set[str] = set()

    # Order matters: we want diversity. Prefer the first instance of each
    # (evidence_type, has_period_filter) combination we encounter.
    for q in data["planner_queries"]:
        plan = q.get("plan")
        if not plan:
            continue
        tier = ComplexityTier(plan["complexity_tier"])
        for slot_raw in plan["slots"]:
            slot = EvidenceSlot.model_validate(slot_raw)
            kind = (
                slot.evidence_type.value,
                slot.period_filter is not None,
            )
            if kind in seen_kinds:
                continue
            seen_kinds.add(kind)
            picks.append((q["query"], tier, slot))
            if len(picks) >= 5:
                return picks
    return picks


def _check_period_filter(
    record_period: str | None,
    candidates: list[dict],
    candidate_periods: dict[str, str | None],
    candidate_source_docs: dict[str, str],
) -> tuple[bool, str]:
    """A candidate satisfies the period filter if its intrinsic period
    matches OR (intrinsic period is None AND its source document's
    derived period matches) — mirrors filter_by_period's policy."""
    if record_period is None:
        return (True, "no filter")
    mismatched: list[str] = []
    for c in candidates:
        cid = c["candidate_id"]
        intrinsic = candidate_periods.get(cid)
        if intrinsic == record_period:
            continue
        if intrinsic is None:
            doc_period = period_from_document_id(candidate_source_docs.get(cid, ""))
            if doc_period == record_period:
                continue
        mismatched.append(cid)
    if mismatched:
        return (False, f"{len(mismatched)} candidate(s) mismatched: {mismatched[:3]}")
    return (True, f"all {len(candidates)} match {record_period}")


def _check_rrf_monotone(candidates: list[dict]) -> tuple[bool, str]:
    if len(candidates) < 2:
        return (True, "too few to check")
    scores = [c["rrf_score"] for c in candidates]
    monotone = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    if not monotone:
        return (False, f"non-monotone: {scores[:5]}")
    return (True, f"{len(scores)} scores descending")


def _check_channel_divergence(
    vec_ids: list[str], bm25_ids: list[str]
) -> tuple[bool, str]:
    set_v, set_b = set(vec_ids), set(bm25_ids)
    if not set_v and not set_b:
        return (False, "both channels empty")
    union = set_v | set_b
    inter = set_v & set_b
    jaccard = len(inter) / len(union) if union else 0.0
    diverged = jaccard < 1.0
    return (diverged, f"jaccard={jaccard:.2f} |v|={len(set_v)} |b25|={len(set_b)}")


def main() -> None:
    picks = _load_slots()
    print(f"Picked {len(picks)} representative slots from Block 7 output.\n")

    results: list[dict] = []
    for query_text, tier, slot in picks:
        print("=" * 78)
        print(f"QUERY: {query_text[:100]}")
        print(f"  tier={tier.value}  slot_id={slot.slot_id}")
        print(f"  evidence_type={slot.evidence_type.value}  "
              f"target_layer={slot.target_layer.value}  "
              f"period_filter={slot.period_filter}")
        print(f"  sub_q: {slot.sub_question[:100]}")
        print(f"  key_terms: {list(slot.key_terms)}")

        # For divergence-check, also run raw channels to measure overlap
        # before fusion / period-filter.
        k = _K_BY_TIER[tier]
        vec_hits_raw = vector_search(slot.sub_question, slot.target_layer, k)
        bm25_hits_raw = bm25_search(list(slot.key_terms), slot.target_layer, k)

        t0 = time.time()
        record = retrieve(slot, complexity_tier=tier)
        elapsed = round(time.time() - t0, 2)
        record_dict = record.model_dump()

        # Build candidate_id -> period / source_document maps for the
        # period-filter check (handles intrinsic and doc-fallback paths).
        period_by_id: dict[str, str | None] = {}
        source_doc_by_id: dict[str, str] = {}
        for c in vec_hits_raw + bm25_hits_raw:
            period_by_id.setdefault(c.candidate_id, c.period)
            source_doc_by_id.setdefault(c.candidate_id, c.source_document)

        ok_period, note_period = _check_period_filter(
            record.period_filter,
            record_dict["candidates"],
            period_by_id,
            source_doc_by_id,
        )
        ok_rrf, note_rrf = _check_rrf_monotone(record_dict["candidates"])
        ok_div, note_div = _check_channel_divergence(
            [c.candidate_id for c in vec_hits_raw],
            [c.candidate_id for c in bm25_hits_raw],
        )
        ok_origin = record.pass_origin == PassOrigin.verifier_loop

        marker = "PASS" if all([ok_period, ok_rrf, ok_div, ok_origin]) else "FAIL"
        print(f"  [{marker}] elapsed={elapsed}s  candidates={len(record_dict['candidates'])}")
        print(f"    period_filter:  {'PASS' if ok_period else 'FAIL'} ({note_period})")
        print(f"    rrf monotone:   {'PASS' if ok_rrf else 'FAIL'} ({note_rrf})")
        print(f"    channel diverge:{'PASS' if ok_div else 'FAIL'} ({note_div})")
        print(f"    pass_origin:    {'PASS' if ok_origin else 'FAIL'} "
              f"({record.pass_origin.value})")
        print("  Top 5 candidates:")
        for c in record_dict["candidates"][:5]:
            print(f"    [{c['rrf_score']:.5f}] {c['source']:5} "
                  f"v={c['vector_score']} b={c['bm25_score']}  {c['candidate_id']}")

        results.append({
            "query": query_text,
            "tier": tier.value,
            "slot": slot.model_dump(),
            "elapsed_seconds": elapsed,
            "record": record_dict,
            "checks": {
                "period_filter": {"ok": ok_period, "note": note_period},
                "rrf_monotone": {"ok": ok_rrf, "note": note_rrf},
                "channel_divergence": {"ok": ok_div, "note": note_div},
                "pass_origin": {"ok": ok_origin, "value": record.pass_origin.value},
            },
            "raw_channel_sizes": {
                "vector": len(vec_hits_raw),
                "bm25": len(bm25_hits_raw),
            },
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nLog -> {OUT_PATH}")


if __name__ == "__main__":
    main()

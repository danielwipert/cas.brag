"""Block 9c: Pipeline orchestrator.

Wires Validate → Plan → per-slot loop(Retrieve → Verify → RETRY with
gap-reformulated sub_question) → Degradation → ExecutionTrace.

The orchestrator is sequential by slot for v1 (the LLM call dominates
wall time; async parallel is a follow-up optimization that doesn't
change the trace shape). Within each slot, the loop:

  1. Retrieves with memory exclusion from the Ledger
  2. Verifies via the seven-checks Verifier
  3. Records coverage / gap on the Ledger
  4. Decides PASS / RETRY / FLAG / EXHAUSTED per spec §3.5

Reformulation: on RETRY, the slot's sub_question is rewritten as
``original_sub_question + " | " + gap_description``. The Retriever
re-embeds the augmented question; key_terms stay the same. This v1
approach is simple, deterministic, and gives the Verifier's gap
report direct influence over the next pass.

Max iterations by tier (spec §3.2):
  Simple   2
  Standard 3
  Complex  4
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from agents.llm_client import LLMError
from agents.planner import plan as run_planner
from agents.retriever.retriever import retrieve
from agents.verifier import verify
from pipeline.degradation import decide_degradation
from pipeline.input_validation import validate
from pipeline.memory_ledger import Ledger
from schemas.enums import (
    ComplexityTier,
    DegradationCause,
    DegradationLevel,
    PassOrigin,
    VerifierVerdict,
)
from schemas.records import (
    CoverageProgressionEntry,
    EvidenceSlot,
    ExecutionTrace,
    FinalSlotState,
    QueryTraceInfo,
    RetrievalRecord,
    VerifierOutput,
)


_MAX_ITER_BY_TIER: dict[ComplexityTier, int] = {
    ComplexityTier.simple: 2,
    ComplexityTier.standard: 3,
    ComplexityTier.complex: 4,
}


@dataclass
class _SlotRun:
    """Accumulator for a single slot's loop."""
    final_verdict: VerifierVerdict
    final_coverage: float
    retrievals: list[RetrievalRecord] = field(default_factory=list)
    verdicts: list[VerifierOutput] = field(default_factory=list)


def _reformulate(slot: EvidenceSlot, gap: str) -> EvidenceSlot:
    """Augment the slot's sub_question with the Verifier's gap
    description for the next retrieval pass. key_terms and
    period_filter are preserved — they're still the right filters."""
    augmented = f"{slot.sub_question} | Gap from prior pass: {gap.strip()}"
    return slot.model_copy(update={"sub_question": augmented})


def _run_slot(
    slot: EvidenceSlot,
    tier: ComplexityTier,
    ledger: Ledger,
    *,
    verbose: bool,
) -> _SlotRun:
    """Run the iterative Verifier loop for one slot."""
    max_iter = _MAX_ITER_BY_TIER[tier]
    current_slot = slot
    last_verdict: VerifierOutput | None = None
    run = _SlotRun(final_verdict=VerifierVerdict.gap, final_coverage=0.0)

    for iteration in range(1, max_iter + 1):
        excluded = ledger.excluded_for_slot(slot.slot_id)
        retrieval = retrieve(
            current_slot,
            complexity_tier=tier,
            iteration=iteration,
            pass_origin=PassOrigin.verifier_loop,
            excluded_ids=excluded,
        )
        ledger.add_retrieval(
            slot.slot_id,
            iteration,
            [c.candidate_id for c in retrieval.candidates],
        )
        run.retrievals.append(retrieval)

        verdict = verify(current_slot, retrieval)
        run.verdicts.append(verdict)
        last_verdict = verdict
        ledger.add_coverage(slot.slot_id, iteration, verdict.coverage_score)
        ledger.add_supported(slot.slot_id, list(verdict.supported_candidates))

        if verbose:
            print(
                f"      iter {iteration}: "
                f"retrieved={len(retrieval.candidates)} "
                f"coverage={verdict.coverage_score:.2f} "
                f"verdict={verdict.verdict.value}"
            )

        if verdict.verdict == VerifierVerdict.covered:
            run.final_verdict = VerifierVerdict.covered
            run.final_coverage = verdict.coverage_score
            return run
        if verdict.verdict == VerifierVerdict.contradiction:
            run.final_verdict = VerifierVerdict.contradiction
            run.final_coverage = verdict.coverage_score
            return run
        # gap (or anything else): consider RETRY
        if iteration == max_iter or ledger.should_exhaust_early(slot.slot_id):
            run.final_verdict = VerifierVerdict.exhausted
            run.final_coverage = verdict.coverage_score
            if verbose:
                why = (
                    "max_iter reached"
                    if iteration == max_iter
                    else "zero-progress (two consecutive stalls)"
                )
                print(f"      → EXHAUSTED ({why})")
            return run
        # RETRY: record the gap and reformulate the slot for the next loop.
        gap = verdict.gap_description or "(no gap description)"
        ledger.add_gap(slot.slot_id, iteration, gap)
        current_slot = _reformulate(current_slot, gap)
        if verbose:
            print(f"      → RETRY with reformulated sub_question")

    # Defensive: should be unreachable.
    if last_verdict:
        run.final_coverage = last_verdict.coverage_score
    return run


def run_pipeline(
    query: str,
    *,
    run_id: str | None = None,
    verbose: bool = False,
) -> ExecutionTrace:
    """Run the full Validate → Plan → Retriever-loop → Degradation
    pipeline for one ``query``. Returns a fully-populated
    ``ExecutionTrace``."""
    rid = run_id or f"run-{uuid.uuid4().hex[:8]}"
    t0 = time.time()
    models_used: set[str] = set()

    # 1. Input Validation
    val = validate(query)
    if verbose:
        print(f"[validate] passed={val.passed} tier={val.complexity_tier.value}"
              f" warnings={list(val.warnings)}")
    if not val.passed:
        # Hard halt: validation rejected the input.
        from schemas.records import DecompositionPlan
        empty_plan = DecompositionPlan(
            query_id=rid,
            original_query=query,
            complexity_tier=val.complexity_tier,
            synthesis_strategy="integrate",  # type: ignore[arg-type]
        )
        return ExecutionTrace(
            run_id=rid,
            query=QueryTraceInfo(
                original_query=query,
                complexity_tier=val.complexity_tier,
                validation_status="rejected",
            ),
            decomposition_plan=empty_plan,
            degradation_level=DegradationLevel.HARD_HALT,
            degradation_cause=DegradationCause.input_failure,
            elapsed_seconds=round(time.time() - t0, 3),
            extra={"rejection_reason": val.rejection_reason},
        )

    # 2. Planner
    try:
        plan_obj, planner_resp = run_planner(val.normalized_query, val.complexity_tier)
        models_used.add(planner_resp.model)
    except LLMError as exc:
        from schemas.records import DecompositionPlan
        empty_plan = DecompositionPlan(
            query_id=rid,
            original_query=query,
            complexity_tier=val.complexity_tier,
            synthesis_strategy="integrate",  # type: ignore[arg-type]
        )
        return ExecutionTrace(
            run_id=rid,
            query=QueryTraceInfo(
                original_query=query,
                complexity_tier=val.complexity_tier,
            ),
            decomposition_plan=empty_plan,
            degradation_level=DegradationLevel.HARD_HALT,
            degradation_cause=DegradationCause.verifier_unavailable,
            elapsed_seconds=round(time.time() - t0, 3),
            extra={"planner_error": str(exc)},
        )

    if verbose:
        print(f"[plan] {len(plan_obj.slots)} slots, "
              f"strategy={plan_obj.synthesis_strategy.value}")

    # 3. Per-slot iterative loop
    ledger = Ledger(rid)
    all_retrievals: list[RetrievalRecord] = []
    all_verdicts: list[VerifierOutput] = []
    final_states: list[FinalSlotState] = []
    total_iterations = 0

    for slot in plan_obj.slots:
        if verbose:
            print(f"  [{slot.slot_id}] {slot.evidence_type.value}"
                  f" / {slot.target_layer.value}"
                  f" / pf={slot.period_filter}")
            print(f"      sub_q: {slot.sub_question[:90]}")
        try:
            slot_run = _run_slot(slot, val.complexity_tier, ledger, verbose=verbose)
        except LLMError as exc:
            # Treat as exhausted for this slot only; keep going for others.
            if verbose:
                print(f"      LLM ERROR: {exc}")
            slot_run = _SlotRun(
                final_verdict=VerifierVerdict.exhausted,
                final_coverage=0.0,
            )
        all_retrievals.extend(slot_run.retrievals)
        all_verdicts.extend(slot_run.verdicts)
        final_states.append(
            FinalSlotState(
                slot_id=slot.slot_id,
                terminal_verdict=slot_run.final_verdict,
                final_coverage=slot_run.final_coverage,
            )
        )
        total_iterations += len(slot_run.verdicts)

    # 4. Degradation
    final_verdicts_for_decision = [
        v for v in all_verdicts
        # take the last verdict per slot for the degradation decision
        if v.slot_id in {fs.slot_id for fs in final_states}
    ]
    # Collapse to last-per-slot.
    last_by_slot: dict[str, VerifierOutput] = {}
    for v in all_verdicts:
        last_by_slot[v.slot_id] = v
    # Override verdicts to match the FinalSlotState (which marks exhaustion).
    final_decision_outputs = []
    for fs in final_states:
        last_v = last_by_slot.get(fs.slot_id)
        if last_v is None:
            continue
        if last_v.verdict != fs.terminal_verdict:
            last_v = last_v.model_copy(update={"verdict": fs.terminal_verdict})
        final_decision_outputs.append(last_v)
    degradation_level, degradation_cause = decide_degradation(final_decision_outputs)

    # 5. Build ExecutionTrace
    coverage_progression = [
        CoverageProgressionEntry(
            slot_id=v.slot_id,
            iteration=i + 1,
            coverage_score=v.coverage_score,
        )
        for v_list in [all_verdicts]
        for i, v in enumerate(v_list)
    ]
    # Track verifier model from the env default. We could attribute
    # per-call but the simpler approach is to record both planner and
    # verifier models when they were exercised.
    if any(rec.candidates for rec in all_retrievals):
        from agents.verifier import VERIFIER_MODEL
        models_used.add(VERIFIER_MODEL)

    return ExecutionTrace(
        run_id=rid,
        query=QueryTraceInfo(
            original_query=query,
            complexity_tier=val.complexity_tier,
        ),
        decomposition_plan=plan_obj,
        retrieval_passes=all_retrievals,
        verifier_verdicts=all_verdicts,
        coverage_progression=coverage_progression,
        final_slot_states=final_states,
        degradation_level=degradation_level,
        degradation_cause=degradation_cause,
        total_iterations=total_iterations,
        elapsed_seconds=round(time.time() - t0, 3),
        models_used=sorted(models_used),
        extra={
            "ledger": ledger.to_record().model_dump(),
            "warnings": list(val.warnings),
        },
    )

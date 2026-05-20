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
from typing import Any

from agents.generator.agent import GENERATOR_MODEL, run_generator
from agents.llm_client import LLMError
from agents.planner import plan as run_planner
from agents.refutation.agent import (
    REFUTATION_FALLBACK_MODEL,
    REFUTATION_MODEL,
    RefutationAgentResult,
    lookup_facts,
    run_refutation,
)
from agents.retriever.retriever import retrieve
from agents.verifier import verify
from pipeline.degradation import decide_degradation
from pipeline.governance import check_governance, format_violations_for_retry
from pipeline.input_validation import validate
from pipeline.memory_ledger import Ledger
from schemas.enums import (
    ComplexityTier,
    DegradationCause,
    DegradationLevel,
    EvidenceType,
    GovernanceSeverity,
    PassOrigin,
    RefutationOverallVerdict,
    RefutationStrategy,
    RefutationVerdict,
    TargetLayer,
    VerifierVerdict,
)
from schemas.records import (
    AnswerSchema,
    CoverageProgressionEntry,
    DisclosedContradiction,
    DisclosedGap,
    DisclosedRefutation,
    EvidenceSlot,
    ExecutionTrace,
    FactRecord,
    FinalSlotState,
    GovernanceViolation,
    QueryTraceInfo,
    RefutationHypothesis,
    RefutationLoopRecord,
    RefutationReport,
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
    pass_origin: PassOrigin = PassOrigin.verifier_loop,
) -> _SlotRun:
    """Run the iterative Verifier loop for one slot. ``pass_origin``
    is stamped on every retrieval; defaults to ``verifier_loop`` for
    Stage 3, and is set to ``refutation_loop`` by Stage 5 re-entry."""
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
            pass_origin=pass_origin,
            excluded_ids=excluded,
        )
        ledger.add_retrieval(
            slot.slot_id,
            iteration,
            [c.candidate_id for c in retrieval.candidates],
        )
        run.retrievals.append(retrieval)

        # Block 21a: when the Verifier LLM errors transiently (provider
        # routing flake, 429, null content), preserve the retrievals and
        # any prior verdicts already recorded in this slot's run, mark
        # the slot exhausted, and return. The previous code let the
        # exception propagate to the outer handler in run_pipeline,
        # which then created a fresh empty _SlotRun and dropped this
        # slot's evidence from the trace entirely — observed ~1-in-3 on
        # complex queries during Block 21 variance runs (Q7).
        try:
            verdict = verify(current_slot, retrieval)
        except LLMError as exc:
            if verbose:
                print(
                    f"      LLM ERROR on verify (iter {iteration}): {exc}\n"
                    "      → EXHAUSTED with partial slot_run preserved"
                )
            run.final_verdict = VerifierVerdict.exhausted
            run.final_coverage = (
                last_verdict.coverage_score if last_verdict else 0.0
            )
            return run
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


_PROBE_LOOP_ETYPE: dict[RefutationStrategy, EvidenceType] = {
    RefutationStrategy.restated_value: EvidenceType.specific_metric,
    RefutationStrategy.revised_value: EvidenceType.specific_metric,
    RefutationStrategy.guidance_vs_actual: EvidenceType.specific_metric,
    RefutationStrategy.later_reversal: EvidenceType.strategic_position,
    RefutationStrategy.alternative_cause: EvidenceType.causal_explanation,
    RefutationStrategy.materialization: EvidenceType.risk_disclosure,
    RefutationStrategy.policy_change: EvidenceType.accounting_policy,
}


@dataclass
class _RefutationStageResult:
    report: RefutationReport | None = None
    retrievals: list[RetrievalRecord] = field(default_factory=list)
    loop_verdicts: list[VerifierOutput] = field(default_factory=list)
    loop_records: list[RefutationLoopRecord] = field(default_factory=list)
    additional_facts: list[FactRecord] = field(default_factory=list)
    fallback_invoked: bool = False
    models_used: list[str] = field(default_factory=list)
    all_resolved: bool = True
    bypassed: bool = False
    bypass_reason: str = ""
    error: str | None = None


def _build_loop_slot(
    h: RefutationHypothesis,
    refuting_facts: list[FactRecord],
    iteration: int,
) -> EvidenceSlot:
    """Synthetic slot used to verify the refuting position the agent
    surfaced. The slot's sub_question is the hypothesis text — already
    phrased as a counter-claim — and key_terms come from the asserter
    + the refuting facts' claims so BM25 can locate the same evidence
    the probe found. period_filter is intentionally empty: the
    refutation is by definition outside the original period."""
    extra_terms: list[str] = []
    for rec in refuting_facts[:3]:
        # Take the first content-bearing tokens from each refuting fact.
        # Cheap and avoids re-implementing key-term extraction.
        for tok in rec.claim.split()[:6]:
            t = tok.strip(".,;:()").lower()
            if len(t) >= 4 and t not in extra_terms:
                extra_terms.append(t)
    return EvidenceSlot(
        slot_id=f"refutation_loop::{h.hypothesis_id}::iter{iteration}",
        sub_question=h.hypothesis_text,
        evidence_type=_PROBE_LOOP_ETYPE[h.strategy],
        target_layer=TargetLayer.both,
        period_filter=None,
        key_terms=extra_terms[:8],
        coverage_threshold=0.50,
    )


def _run_refutation_stage(
    *,
    run_id: str,
    query: str,
    tier: ComplexityTier,
    supported_ids: list[str],
    ledger: Ledger,
    verbose: bool,
    max_loop_iterations: int = 2,
) -> _RefutationStageResult:
    """Stage 5: Refutation Agent + (if needed) refutation loop.

    Resolves ``supported_ids`` to full ``FactRecord`` objects via the
    fact_store JSONL index, calls ``run_refutation``, and — if the
    agent flagged any hypothesis as ``strongly_refuted`` — runs a
    Stage-3-style verifier loop on a synthetic slot for each.

    Returns a ``_RefutationStageResult`` describing what happened. If
    no verified facts survive the supported→FactRecord lookup (e.g.
    every supported candidate was a chunk), the stage bypasses with
    ``bypassed=True`` and the report is ``None``."""
    result = _RefutationStageResult()
    fact_map = lookup_facts(supported_ids)
    verified_facts = list(fact_map.values())
    if not verified_facts:
        result.bypassed = True
        result.bypass_reason = "no fact-resolved supported candidates"
        if verbose:
            print("  [refutation] BYPASS: no verified FactRecords to refute")
        return result

    if verbose:
        print(
            f"  [refutation] {len(verified_facts)} verified facts; "
            f"max_loop_iterations={max_loop_iterations}"
        )

    try:
        agent_result: RefutationAgentResult = run_refutation(
            run_id=run_id,
            query=query,
            complexity_tier=tier,
            verified_facts=verified_facts,
            ledger=ledger,
            iteration=1,
            max_loop_iterations=max_loop_iterations,
        )
    except LLMError as exc:
        # Both Mistral and Llama 3.3 70B were unreachable. Log and
        # bypass — the orchestrator will record refutation_unavailable
        # on the trace.
        if verbose:
            print(f"  [refutation] LLM ERROR (no fallback succeeded): {exc}")
        result.bypassed = True
        result.bypass_reason = f"llm_error: {exc}"
        result.error = str(exc)
        return result

    result.report = agent_result.report
    result.retrievals = list(agent_result.retrieval_records)
    result.fallback_invoked = agent_result.fallback_invoked
    result.models_used = list(agent_result.models_used)

    if verbose:
        rep = agent_result.report
        print(
            f"  [refutation] overall_verdict={rep.overall_verdict.value} "
            f"hypotheses={len(rep.hypotheses)} "
            f"strongly_refuted={len(agent_result.strongly_refuted)}"
        )

    if agent_result.report.overall_verdict == RefutationOverallVerdict.answer_strengthened:
        # PASS or DISCLOSE — nothing more to do.
        return result

    # refutation_to_loop OR refutation_to_partial. In both, the agent
    # has flagged at least one strongly_refuted hypothesis. We attempt
    # a single loop pass per hypothesis (one slot, one verifier loop).
    # If the verifier covers the refuting position, the conflict is
    # resolved (Normal with structured temporal-evolution). If the
    # verifier exhausts, that hypothesis remains unresolved and the
    # orchestrator drops to Partial.
    all_resolved = True
    for h in agent_result.strongly_refuted:
        refuting = agent_result.refuting_facts.get(h.hypothesis_id, [])
        loop_iter = ledger.refutation_loop_count() + 1
        loop_slot = _build_loop_slot(h, refuting, loop_iter)
        if verbose:
            print(
                f"  [refutation_loop {loop_iter}] hypothesis={h.hypothesis_id} "
                f"strategy={h.strategy.value} -> slot {loop_slot.slot_id}"
            )
        try:
            loop_run = _run_slot(
                loop_slot, tier, ledger,
                verbose=verbose,
                pass_origin=PassOrigin.refutation_loop,
            )
        except LLMError as exc:
            if verbose:
                print(f"      LLM ERROR in refutation loop: {exc}")
            loop_run = _SlotRun(
                final_verdict=VerifierVerdict.exhausted,
                final_coverage=0.0,
            )

        result.retrievals.extend(loop_run.retrievals)
        result.loop_verdicts.extend(loop_run.verdicts)

        # Pull the supported FactRecords from the loop slot's verdict.
        new_facts: list[FactRecord] = []
        if loop_run.final_verdict == VerifierVerdict.covered:
            last_v = loop_run.verdicts[-1] if loop_run.verdicts else None
            if last_v is not None:
                resolved_map = lookup_facts(last_v.supported_candidates)
                # De-dupe against the original verified set.
                existing_ids = {f.fact_id for f in verified_facts}
                for fid, rec in resolved_map.items():
                    if fid not in existing_ids:
                        new_facts.append(rec)
                        existing_ids.add(fid)

        loop_record = RefutationLoopRecord(
            iteration=loop_iter,
            triggering_hypothesis_id=h.hypothesis_id,
            targets_claim_id=h.targets_claim_id,
            coverage_after=loop_run.final_coverage,
        )
        ledger.add_refutation_loop(loop_record)
        result.loop_records.append(loop_record)

        if new_facts:
            result.additional_facts.extend(new_facts)
            if verbose:
                print(
                    f"      -> RESOLVED: {len(new_facts)} new verified fact(s) "
                    "added (structured temporal-evolution path)"
                )
        else:
            all_resolved = False
            if verbose:
                print(
                    "      -> UNRESOLVED: refutation loop did not cover the "
                    "counter-position; downgrade to Partial"
                )

    result.all_resolved = all_resolved
    return result


def _build_disclosed_gaps(
    slots: list[EvidenceSlot],
    final_outputs: list[VerifierOutput],
) -> list[DisclosedGap]:
    """For Partial outputs, surface every slot whose terminal verdict
    is gap / exhausted with the last gap_description from the Verifier
    so the answer text can call them out. Slots that came back covered
    don't appear here."""
    out: list[DisclosedGap] = []
    for v in final_outputs:
        if v.verdict in (VerifierVerdict.gap, VerifierVerdict.exhausted):
            description = v.gap_description or (
                f"slot {v.slot_id} did not reach the coverage threshold"
            )
            out.append(DisclosedGap(slot_id=v.slot_id, gap_description=description))
    return out


def _build_disclosed_contradictions(
    final_outputs: list[VerifierOutput],
) -> list[DisclosedContradiction]:
    """Hoist the Verifier's contradiction_details onto the answer."""
    out: list[DisclosedContradiction] = []
    for v in final_outputs:
        for cd in v.contradiction_details:
            out.append(
                DisclosedContradiction(
                    description=cd.description,
                    conflicting_ids=list(cd.conflicting_ids),
                )
            )
    return out


def _inject_missing_disclosures(
    answer: AnswerSchema,
    refutation_report: RefutationReport | None,
) -> AnswerSchema:
    """Last-ditch repair for the post-retry undisclosed_refutation path:
    deterministically rebuild the disclosed_refutations list from the
    report so the structured field is at least correct, even if the
    answer_text remains under-disclosed (which is what drops us to
    Partial)."""
    if refutation_report is None:
        return answer
    rebuilt: list[DisclosedRefutation] = []
    for h in refutation_report.hypotheses:
        if h.refutation_verdict == RefutationVerdict.unrefuted:
            continue
        rebuilt.append(
            DisclosedRefutation(
                targets_claim_id=h.targets_claim_id,
                refuting_evidence_ids=list(h.evidence_ids),
                refutation_verdict=h.refutation_verdict,
                strategy=h.strategy,
            )
        )
    return answer.model_copy(update={"disclosed_refutations": rebuilt})


def _run_generator_and_governance(
    *,
    original_query: str,
    verified_facts: list[FactRecord],
    refutation_report: RefutationReport | None,
    degradation_level: DegradationLevel,
    degradation_cause: DegradationCause,
    adversarially_probed: bool,
    disclosed_gaps: list[DisclosedGap],
    disclosed_contradictions: list[DisclosedContradiction],
    models_used: set[str],
    verbose: bool,
) -> tuple[
    AnswerSchema,
    list[GovernanceViolation],
    DegradationLevel,
    DegradationCause,
]:
    """Stages 7 + 8.

    Bypass paths (CR / Hard Halt / empty verified set) produce a canned
    AnswerSchema via ``run_generator``. The Normal/Partial path calls
    the Generator and then the Governance gate, with one retry on
    ``undisclosed_refutation`` and an immediate Hard Halt on
    ``numerical_mismatch``.
    """
    violations: list[GovernanceViolation] = []

    # CR / Hard Halt → canned answer, no Generator + no Governance.
    if degradation_level in (
        DegradationLevel.CLARIFICATION_REQUEST,
        DegradationLevel.HARD_HALT,
    ):
        answer, _ = run_generator(
            original_query=original_query,
            verified_facts=verified_facts,
            refutation_report=refutation_report,
            degradation_level=degradation_level,
            adversarially_probed=adversarially_probed,
            disclosed_gaps=disclosed_gaps,
            disclosed_contradictions=disclosed_contradictions,
        )
        if verbose:
            print(f"  [generator] BYPASS: degradation={degradation_level.name}")
        return answer, violations, degradation_level, degradation_cause

    # Empty verified set on a Normal/Partial path: nothing to ground an
    # answer with. Reroute to Clarification Request.
    if not verified_facts:
        if verbose:
            print("  [generator] BYPASS: empty verified set -> Clarification Request")
        new_level = DegradationLevel.CLARIFICATION_REQUEST
        new_cause = DegradationCause.slot_exhaustion
        answer, _ = run_generator(
            original_query=original_query,
            verified_facts=verified_facts,
            refutation_report=refutation_report,
            degradation_level=new_level,
            adversarially_probed=adversarially_probed,
            disclosed_gaps=disclosed_gaps,
            disclosed_contradictions=disclosed_contradictions,
        )
        return answer, violations, new_level, new_cause

    # Stage 7: Generator
    try:
        answer, gen_resp = run_generator(
            original_query=original_query,
            verified_facts=verified_facts,
            refutation_report=refutation_report,
            degradation_level=degradation_level,
            adversarially_probed=adversarially_probed,
            disclosed_gaps=disclosed_gaps,
            disclosed_contradictions=disclosed_contradictions,
        )
        if gen_resp is not None:
            models_used.add(gen_resp.model)
        if verbose:
            print(
                f"  [generator] OK — {len(answer.claims)} claim(s), "
                f"answer_len={len(answer.answer_text)} chars"
            )
    except LLMError as exc:
        if verbose:
            print(f"  [generator] LLM ERROR: {exc}")
        new_level = DegradationLevel.HARD_HALT
        new_cause = DegradationCause.generator_unavailable
        stub_answer, _ = run_generator(
            original_query=original_query,
            verified_facts=verified_facts,
            refutation_report=refutation_report,
            degradation_level=new_level,
            adversarially_probed=adversarially_probed,
            disclosed_gaps=disclosed_gaps,
            disclosed_contradictions=disclosed_contradictions,
        )
        return stub_answer, violations, new_level, new_cause

    # Stage 8: Governance
    initial = check_governance(
        answer=answer,
        verified_facts=verified_facts,
        refutation_report=refutation_report,
        expected_adversarially_probed=adversarially_probed,
    )
    violations.extend(initial)
    if verbose and initial:
        for v in initial:
            print(f"  [governance] {v.severity.value}: {v.message}")

    has_numerical = any(
        v.severity == GovernanceSeverity.numerical_mismatch for v in initial
    )
    if has_numerical:
        # Constitutional violation — never release a wrong number.
        return (
            answer,
            violations,
            DegradationLevel.HARD_HALT,
            DegradationCause.constitutional_violation,
        )

    has_undisclosed = any(
        v.severity == GovernanceSeverity.undisclosed_refutation for v in initial
    )
    has_badge = any(
        v.severity == GovernanceSeverity.badge_mismatch for v in initial
    )

    if has_undisclosed:
        feedback = format_violations_for_retry(
            [
                v for v in initial
                if v.severity == GovernanceSeverity.undisclosed_refutation
            ]
        )
        try:
            answer2, gen_resp2 = run_generator(
                original_query=original_query,
                verified_facts=verified_facts,
                refutation_report=refutation_report,
                degradation_level=degradation_level,
                adversarially_probed=adversarially_probed,
                disclosed_gaps=disclosed_gaps,
                disclosed_contradictions=disclosed_contradictions,
                prior_governance_feedback=feedback,
            )
            if gen_resp2 is not None:
                models_used.add(gen_resp2.model)
            second = check_governance(
                answer=answer2,
                verified_facts=verified_facts,
                refutation_report=refutation_report,
                expected_adversarially_probed=adversarially_probed,
            )
            violations.extend(second)
            if any(
                v.severity == GovernanceSeverity.numerical_mismatch for v in second
            ):
                return (
                    answer2,
                    violations,
                    DegradationLevel.HARD_HALT,
                    DegradationCause.constitutional_violation,
                )
            if any(
                v.severity == GovernanceSeverity.undisclosed_refutation
                for v in second
            ):
                repaired = _inject_missing_disclosures(answer2, refutation_report)
                # Fix any badge issue while we're patching.
                if repaired.adversarially_probed != adversarially_probed:
                    repaired = repaired.model_copy(
                        update={"adversarially_probed": adversarially_probed}
                    )
                new_level = (
                    DegradationLevel.PARTIAL
                    if degradation_level == DegradationLevel.NORMAL
                    else degradation_level
                )
                new_cause = (
                    DegradationCause.governance_failure
                    if degradation_level == DegradationLevel.NORMAL
                    else degradation_cause
                )
                return repaired, violations, new_level, new_cause
            answer = answer2
            if any(
                v.severity == GovernanceSeverity.badge_mismatch for v in second
            ):
                answer = answer.model_copy(
                    update={"adversarially_probed": adversarially_probed}
                )
            return answer, violations, degradation_level, degradation_cause
        except LLMError as exc:
            if verbose:
                print(f"  [generator-retry] LLM ERROR: {exc}")
            repaired = _inject_missing_disclosures(answer, refutation_report)
            new_level = (
                DegradationLevel.PARTIAL
                if degradation_level == DegradationLevel.NORMAL
                else degradation_level
            )
            new_cause = (
                DegradationCause.governance_failure
                if degradation_level == DegradationLevel.NORMAL
                else degradation_cause
            )
            return repaired, violations, new_level, new_cause

    if has_badge:
        answer = answer.model_copy(
            update={"adversarially_probed": adversarially_probed}
        )

    return answer, violations, degradation_level, degradation_cause


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

    # 4. Provisional degradation (before refutation). The bypass rule
    # for the Refutation Agent is: skip whenever the run is heading to
    # anything other than Normal — there's no clean answer to refute.
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

    # 4b. Stage 5: Refutation Agent (run only on Normal-bound runs).
    refutation_report: RefutationReport | None = None
    refutation_loop_iterations: list[RefutationLoopRecord] = []
    adversarially_probed = False
    refutation_bypass_reason: str | None = None
    ref_result: _RefutationStageResult | None = None

    # Accumulate the union of supported ids across slots — used both as
    # input to the Refutation Agent and (after refutation) as the
    # verified-fact pool for the Generator.
    all_supported_ids: list[str] = []
    seen_supported: set[str] = set()
    for v in final_decision_outputs:
        for cid in v.supported_candidates:
            if cid not in seen_supported:
                all_supported_ids.append(cid)
                seen_supported.add(cid)

    if degradation_level == DegradationLevel.NORMAL:
        ref_result = _run_refutation_stage(
            run_id=rid,
            query=val.normalized_query,
            tier=val.complexity_tier,
            supported_ids=all_supported_ids,
            ledger=ledger,
            verbose=verbose,
        )

        if ref_result.bypassed:
            refutation_bypass_reason = ref_result.bypass_reason
            adversarially_probed = False
        else:
            refutation_report = ref_result.report
            refutation_loop_iterations.extend(ref_result.loop_records)
            all_retrievals.extend(ref_result.retrievals)
            all_verdicts.extend(ref_result.loop_verdicts)
            total_iterations += len(ref_result.loop_verdicts)
            for m in ref_result.models_used:
                models_used.add(m)
            adversarially_probed = True
            if not ref_result.all_resolved:
                degradation_level = DegradationLevel.PARTIAL
                degradation_cause = DegradationCause.refutation_unresolved
    else:
        # Bypass path: refutation does not run under Partial /
        # Clarification Request / Hard Halt — the spec's "no clean
        # answer to refute" rule.
        refutation_bypass_reason = (
            f"degradation={degradation_level.name} — refutation bypassed"
        )
        if verbose:
            print(f"  [refutation] BYPASS: {refutation_bypass_reason}")

    # 4c. Build the verified fact set the Generator (and Governance)
    # will see. Combines supported_ids resolved to FactRecords with any
    # additional facts the refutation loop pulled in (the structured
    # temporal-evolution path).
    verified_facts = list(lookup_facts(all_supported_ids).values())
    if ref_result is not None and not ref_result.bypassed:
        existing = {f.fact_id for f in verified_facts}
        for f in ref_result.additional_facts:
            if f.fact_id not in existing:
                verified_facts.append(f)
                existing.add(f.fact_id)

    # Build the disclosure inputs from the slot state.
    disclosed_gaps = _build_disclosed_gaps(plan_obj.slots, final_decision_outputs)
    disclosed_contradictions = _build_disclosed_contradictions(final_decision_outputs)

    # 4d. Stage 7 (Generator) + Stage 8 (Governance).
    answer, governance_violations, degradation_level, degradation_cause = (
        _run_generator_and_governance(
            original_query=val.normalized_query,
            verified_facts=verified_facts,
            refutation_report=refutation_report,
            degradation_level=degradation_level,
            degradation_cause=degradation_cause,
            adversarially_probed=adversarially_probed,
            disclosed_gaps=disclosed_gaps,
            disclosed_contradictions=disclosed_contradictions,
            models_used=models_used,
            verbose=verbose,
        )
    )

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

    extra: dict[str, Any] = {
        "ledger": ledger.to_record().model_dump(),
        "warnings": list(val.warnings),
        "adversarially_probed": adversarially_probed,
    }
    if refutation_bypass_reason is not None:
        extra["refutation_bypass_reason"] = refutation_bypass_reason
    if refutation_report is not None and any(
        # The fallback flag the Refutation Agent sets is on the report
        # via model_used (Llama 3.3 70B != Mistral) — surface it as an
        # explicit boolean for downstream Layer-5 monitoring.
        m == REFUTATION_FALLBACK_MODEL for m in models_used
    ) and REFUTATION_MODEL not in models_used:
        extra["refutation_unavailable_fallback_invoked"] = True

    return ExecutionTrace(
        run_id=rid,
        query=QueryTraceInfo(
            original_query=query,
            complexity_tier=val.complexity_tier,
        ),
        decomposition_plan=plan_obj,
        retrieval_passes=all_retrievals,
        verifier_verdicts=all_verdicts,
        refutation_report=refutation_report,
        coverage_progression=coverage_progression,
        refutation_loop_iterations=refutation_loop_iterations,
        final_slot_states=final_states,
        answer=answer,
        governance_violations=governance_violations,
        degradation_level=degradation_level,
        degradation_cause=degradation_cause,
        total_iterations=total_iterations,
        elapsed_seconds=round(time.time() - t0, 3),
        models_used=sorted(models_used),
        extra=extra,
    )

"""Block 14c showcase end-to-end test.

Two passes, mirroring Block 11's structure:

  1. **Canonical** — run the Q5 showcase query through
     ``run_pipeline()``. With Block 14 wiring this now exercises
     Stage 7 (Generator) + Stage 8 (Governance) when the verified set
     resolves. The trace is logged regardless of degradation outcome.

  2. **Asserting** — hand-pick the 2017–2018 no-ads strategic_claim
     fact_ids that we know exist in the corpus, drive them through
     ``_run_refutation_stage`` (Block 11 mechanism) and then through
     ``_run_generator_and_governance`` (Block 14 wiring). This is the
     reliable showcase: the Verifier ID-naming bug noted in Block 11
     is bypassed by skipping the Verifier coverage call entirely, so
     the Generator + Governance can run on a known-good verified set.

Acceptance criteria (build plan v3 §Block 14):

  - Verifier covers the no-ads slot. (Asserting path: pre-seeded.)
  - Refutation surfaces the Q4 2022 ad-tier as ``strongly_refuted``.
  - Loop resolves to Normal-with-temporal-evolution OR drops to
    Partial-with-refutation_unresolved (both are spec-acceptable
    outcomes; Block 11's note about the ID bug applies).
  - Generator produces an answer naming BOTH the 2018 no-ads claim
    AND the Q4 2022 reversal, with both ``assertion_date``s present
    in the answer text.
  - Governance passes (no ``numerical_mismatch`` violations) — any
    ``undisclosed_refutation`` violations should resolve via the
    Generator retry path or the manual-injection fallback.
  - Final ``AnswerSchema.adversarially_probed == True``.
  - Trace populates the new ``answer`` and ``governance_violations``
    fields.

Run from repo root::

    python -m scripts.test_block14

Logs both traces to data/logs/block14_test.json.
"""
from __future__ import annotations

import json
import re
import sys
import time
import uuid
from datetime import date
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from agents.refutation.agent import lookup_facts
from pipeline.memory_ledger import Ledger
from pipeline.orchestrator import (
    _build_disclosed_contradictions,
    _build_disclosed_gaps,
    _run_generator_and_governance,
    _run_refutation_stage,
    run_pipeline,
)
from schemas.enums import (
    ComplexityTier,
    DegradationCause,
    DegradationLevel,
    GovernanceSeverity,
    RefutationStrategy,
    RefutationVerdict,
)
from schemas.records import FactRecord


CANONICAL_QUERY = "Did Netflix ever say it had no plans to add ads?"
ASSERTING_QUERY = "Has Netflix's stance on advertising changed over time?"


# 2017–2018 strategic_claim fact_ids in the corpus that assert
# Netflix's no-advertising position. Reused from test_block11.
_VERIFIED_NO_ADS_IDS = [
    "F-PROSE-nflx-q4-2017-transcript-0070",  # 2018-01-22  no-ads as differentiator
    "F-PROSE-nflx-q1-2018-transcript-0096",  # 2018-04-16  "Netflix does not sell advertising"
    "F-PROSE-nflx-q4-2018-transcript-0066",  # 2019-01-17  no advertising, on-demand
]


def _print_section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _check(label: str, ok: bool, detail: str = "") -> bool:
    marker = "OK  " if ok else "FAIL"
    line = f"  [{marker}] {label}"
    if detail:
        line += f"  — {detail}"
    print(line)
    return ok


# ---------------------------------------------------------------------------
# Date detection in answer_text (mirrors generator_prompt_harness logic)
# ---------------------------------------------------------------------------


def _date_token_variants(d: date) -> list[str]:
    return [
        d.isoformat(),
        f"{d.year}-{d.month:02d}",
        str(d.year),
        d.strftime("%B %d, %Y"),
        d.strftime("%B %Y"),
        f"Q{(d.month - 1) // 3 + 1} {d.year}",
        f"Q{(d.month - 1) // 3 + 1} {str(d.year)[-2:]}",
    ]


def _mentions_date(answer_text: str, d: date) -> bool:
    return any(v in answer_text for v in _date_token_variants(d))


# ---------------------------------------------------------------------------
# Asserting path
# ---------------------------------------------------------------------------


def _run_asserting_path() -> dict:
    rid = f"asserting-b14-{uuid.uuid4().hex[:6]}"
    ledger = Ledger(rid)
    tier = ComplexityTier.standard

    _print_section("ASSERTING: hand-picked verified fact_ids")
    for fid in _VERIFIED_NO_ADS_IDS:
        print(f"  verified: {fid}")

    print(f"\n  -- refutation stage ({len(_VERIFIED_NO_ADS_IDS)} verified ids) --")
    ref_result = _run_refutation_stage(
        run_id=rid,
        query=ASSERTING_QUERY,
        tier=tier,
        supported_ids=list(_VERIFIED_NO_ADS_IDS),
        ledger=ledger,
        verbose=True,
    )

    if ref_result.bypassed or ref_result.report is None:
        return {
            "ref_result": ref_result,
            "answer": None,
            "violations": [],
            "degradation_level": DegradationLevel.PARTIAL,
            "degradation_cause": DegradationCause.refutation_unavailable,
            "verified_facts": [],
            "ledger": ledger,
        }

    # Build the verified-fact set the Generator will see: the original
    # supported facts + any facts the refutation loop pulled in.
    verified_facts: list[FactRecord] = list(
        lookup_facts(_VERIFIED_NO_ADS_IDS).values()
    )
    existing = {f.fact_id for f in verified_facts}
    for f in ref_result.additional_facts:
        if f.fact_id not in existing:
            verified_facts.append(f)
            existing.add(f.fact_id)

    # If the refutation loop did not resolve, we drop to Partial before
    # Generator (mirroring the orchestrator's contract).
    if ref_result.all_resolved:
        degradation_level = DegradationLevel.NORMAL
        degradation_cause = DegradationCause.none
    else:
        degradation_level = DegradationLevel.PARTIAL
        degradation_cause = DegradationCause.refutation_unresolved

    # No real verifier slots ran on the asserting path, so disclosed_gaps
    # / disclosed_contradictions are empty. (The Generator handles
    # Partial without a gap list by relying on the refutation report.)
    models_used: set[str] = set(ref_result.models_used or [])

    print(f"\n  -- generator + governance (degradation={degradation_level.name}) --")
    answer, violations, degradation_level, degradation_cause = (
        _run_generator_and_governance(
            original_query=ASSERTING_QUERY,
            verified_facts=verified_facts,
            refutation_report=ref_result.report,
            degradation_level=degradation_level,
            degradation_cause=degradation_cause,
            adversarially_probed=True,  # the agent ran on this path
            disclosed_gaps=_build_disclosed_gaps([], []),
            disclosed_contradictions=_build_disclosed_contradictions([]),
            models_used=models_used,
            verbose=True,
        )
    )

    return {
        "ref_result": ref_result,
        "answer": answer,
        "violations": violations,
        "degradation_level": degradation_level,
        "degradation_cause": degradation_cause,
        "verified_facts": verified_facts,
        "ledger": ledger,
        "models_used": sorted(models_used),
    }


# ---------------------------------------------------------------------------
# Acceptance checks
# ---------------------------------------------------------------------------


def _run_acceptance_checks(state: dict) -> tuple[int, int]:
    _print_section("BLOCK 14 ACCEPTANCE CHECKS")
    checks: list[bool] = []

    ref_result = state["ref_result"]
    report = ref_result.report if ref_result is not None else None
    answer = state["answer"]
    violations = state["violations"]

    checks.append(_check(
        "Refutation Agent executed (not bypassed)",
        report is not None,
        ref_result.bypass_reason if ref_result and ref_result.bypassed else "",
    ))

    if report is None or answer is None:
        return sum(1 for c in checks if c), len(checks)

    # 2. Strong refutation present (the later_reversal hypothesis).
    has_strong = any(
        h.refutation_verdict == RefutationVerdict.strongly_refuted
        and h.strategy == RefutationStrategy.later_reversal
        for h in report.hypotheses
    )
    checks.append(_check(
        "At least one later_reversal hypothesis was strongly_refuted",
        has_strong,
        "Q4-2022 / 2023 ad-tier facts should clear the strong gates",
    ))

    # 3. AnswerSchema is populated.
    checks.append(_check(
        "ExecutionTrace populated answer (AnswerSchema)",
        answer is not None
        and isinstance(answer.answer_text, str)
        and len(answer.answer_text) > 0,
        f"answer_text length={len(answer.answer_text)}",
    ))

    # 4. adversarially_probed = True
    checks.append(_check(
        "AnswerSchema.adversarially_probed == True",
        answer.adversarially_probed is True,
        f"flag={answer.adversarially_probed}",
    ))

    # 5. Governance found no numerical_mismatch violations.
    num_violations = [
        v for v in violations
        if v.severity == GovernanceSeverity.numerical_mismatch
    ]
    checks.append(_check(
        "Governance: no numerical_mismatch violations",
        len(num_violations) == 0,
        f"{len(num_violations)} numerical_mismatch violation(s)",
    ))

    # 6. disclosed_refutations[] has the strong refutation entry.
    expected_disclosures = sum(
        1 for h in report.hypotheses
        if h.refutation_verdict != RefutationVerdict.unrefuted
    )
    checks.append(_check(
        "disclosed_refutations[] populated for non-unrefuted hypotheses",
        len(answer.disclosed_refutations) >= expected_disclosures,
        f"expected ≥{expected_disclosures}, got "
        f"{len(answer.disclosed_refutations)}",
    ))

    # 7. answer_text mentions BOTH the targeted (no-ads) and refuting
    # (ad-tier) assertion_dates. Spec §3.8 only requires AT LEAST ONE
    # refuting fact's date per strong refutation, not every
    # evidence_ids entry — the narrative names position A and position
    # B, not every fact that supports position B.
    fact_by_id = {f.fact_id: f for f in state["verified_facts"]}
    targeted_dates_ok = True
    refuting_dates_ok = True
    for h in report.hypotheses:
        if h.refutation_verdict != RefutationVerdict.strongly_refuted:
            continue
        targeted = fact_by_id.get(h.targets_claim_id)
        if targeted is not None and not _mentions_date(
            answer.answer_text, targeted.assertion_date
        ):
            targeted_dates_ok = False
        # At-least-one match across this hypothesis's evidence_ids.
        any_refuting_matched = False
        any_refuting_known = False
        for eid in h.evidence_ids:
            refuting = fact_by_id.get(eid)
            if refuting is None:
                rmap = lookup_facts([eid])
                refuting = rmap.get(eid)
            if refuting is None:
                continue
            any_refuting_known = True
            if _mentions_date(answer.answer_text, refuting.assertion_date):
                any_refuting_matched = True
                break
        if any_refuting_known and not any_refuting_matched:
            refuting_dates_ok = False
    checks.append(_check(
        "answer_text mentions the targeted (no-ads) assertion_date",
        targeted_dates_ok,
        "structural temporal-evolution narrative requirement",
    ))
    checks.append(_check(
        "answer_text mentions at least one refuting assertion_date "
        "per strong hypothesis",
        refuting_dates_ok,
        "structural temporal-evolution narrative requirement",
    ))

    # 8. Final state is one of the spec-acceptable outcomes.
    lvl = state["degradation_level"]
    cause = state["degradation_cause"]
    ok_final = (
        lvl == DegradationLevel.NORMAL
        or (lvl == DegradationLevel.PARTIAL
            and cause in (
                DegradationCause.refutation_unresolved,
                DegradationCause.governance_failure,
            ))
    )
    checks.append(_check(
        "Final state Normal-resolved OR Partial-with-known-cause",
        ok_final,
        f"level={lvl.name} cause={cause.value}",
    ))

    return sum(1 for c in checks if c), len(checks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # ---- Canonical run (informational) -------------------------------
    _print_section(f"CANONICAL (observational): {CANONICAL_QUERY}")
    t0 = time.time()
    canon_trace = run_pipeline(CANONICAL_QUERY, verbose=True)
    canon_elapsed = round(time.time() - t0, 2)
    canon_answer_len = (
        len(canon_trace.answer.answer_text) if canon_trace.answer else 0
    )
    print(
        f"  canonical result: degradation={canon_trace.degradation_level.name} "
        f"refutation={'ran' if canon_trace.refutation_report else 'bypassed'} "
        f"answer_len={canon_answer_len} "
        f"governance_violations={len(canon_trace.governance_violations)} "
        f"elapsed={canon_elapsed}s"
    )

    # ---- Asserting run -----------------------------------------------
    t0 = time.time()
    state = _run_asserting_path()
    elapsed = round(time.time() - t0, 2)

    _print_section("ASSERTING RESULT SUMMARY")
    print(f"  degradation_level: {state['degradation_level'].name}")
    print(f"  degradation_cause: {state['degradation_cause'].value}")
    print(f"  elapsed:           {elapsed}s")
    if state.get("models_used"):
        print(f"  models_used:       {state['models_used']}")
    answer = state["answer"]
    if answer is not None:
        text = answer.answer_text
        if len(text) > 600:
            text = text[:600] + "…"
        print()
        print("  answer_text:")
        for line in text.splitlines() or [text]:
            print(f"    {line}")
        print(f"  claims:                {len(answer.claims)}")
        print(f"  disclosed_refutations: {len(answer.disclosed_refutations)}")
        print(f"  adversarially_probed:  {answer.adversarially_probed}")
    violations = state["violations"]
    if violations:
        print(f"\n  governance_violations ({len(violations)}):")
        for v in violations:
            print(f"    [{v.severity.value}] {v.message}")
    else:
        print("\n  governance_violations: NONE")

    # ---- Persist traces ----------------------------------------------
    out_path = Path("data/logs/block14_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    asserting_payload = {
        "elapsed_s": elapsed,
        "degradation_level": state["degradation_level"].name,
        "degradation_cause": state["degradation_cause"].value,
        "answer": answer.model_dump(mode="json") if answer else None,
        "governance_violations": [v.model_dump(mode="json") for v in violations],
        "refutation_report": (
            state["ref_result"].report.model_dump(mode="json")
            if state["ref_result"] and state["ref_result"].report else None
        ),
        "ledger": state["ledger"].to_record().model_dump(mode="json"),
        "models_used": state.get("models_used", []),
    }
    out_path.write_text(
        json.dumps(
            {
                "canonical": canon_trace.model_dump(mode="json"),
                "asserting": asserting_payload,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nTraces -> {out_path}")

    passed, total = _run_acceptance_checks(state)
    print()
    print("=" * 78)
    print(f"OVERALL: {passed}/{total} acceptance checks passed")
    print("=" * 78)
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()

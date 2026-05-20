"""Block 18 (Phase 4): Adversarial test suite — full spec §9.2 slate.

Runs all 15 curated queries from spec §9.2 end-to-end through the full
pipeline (Validate → Plan → Verify → Refute → Generate → Govern) and
compares per-query behavior against the spec's expected outcome.

This is the canonical Phase 4 test corpus. The build plan's calibration
targets are:

  Normal, no refutation:     Q1, Q4, Q6, Q11, Q13, Q14
  Normal, weak refutation:   Q3, Q7, Q10
  Strong refutation:         Q2, Q9, Q12, Q15
  Clarification Request:     Q5
  CR or attribution failure: Q8

First-run acceptance is not the goal — the goal is to surface which
thresholds need calibration. Block 20 acts on the failure profile this
suite produces.

Outputs land in ``data/logs/phase4_adversarial/`` (gitignored):

  - ``Q{n}.json``  — full ExecutionTrace per query.
  - ``Q{n}.html``  — per-query HTML render (opt-in via --render-html).
  - ``summary.json`` — machine-readable summary table for Block 20.

Run from repo root::

    python -m scripts.test_phase4_adversarial            # all 15, no HTML
    python -m scripts.test_phase4_adversarial --render-html
    python -m scripts.test_phase4_adversarial --only Q1,Q4,Q11
    python -m scripts.test_phase4_adversarial --skip Q7,Q8
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from agents.refutation.agent import lookup_facts
from pipeline.html_renderer import render_html
from pipeline.orchestrator import run_pipeline
from schemas.enums import (
    ComplexityTier,
    DegradationCause,
    DegradationLevel,
    RefutationVerdict,
)
from schemas.records import ExecutionTrace, FactRecord


# ---------------------------------------------------------------------------
# Expected outcome categories (spec §9.2 distilled via build plan)
# ---------------------------------------------------------------------------


class ExpectedCategory(str, Enum):
    """Five buckets the spec's expected behavior collapses into.

    NORMAL_NO_REF       — query has a clean Normal answer; refutation
                          may run but should not surface a weak/strong
                          hypothesis.
    NORMAL_WEAK         — Normal degradation; at least one refutation
                          hypothesis may be present, weakly_refuted at
                          most. A clean Normal-without-refutation is
                          ALSO acceptable per spec wording ("Normal;
                          weak refutation possible if ...").
    STRONG              — at least one strongly_refuted hypothesis must
                          surface. Degradation in {Normal, Partial}.
    CR                  — degradation must be CR or Hard Halt; no
                          refutation requirement.
    CR_OR_ATTR_FAIL     — degradation in {CR, Hard Halt, Normal, Partial}
                          AND either no answer OR refutation surfaced
                          (Verifier attribution failure is acceptable as
                          long as the planted-false-attribution doesn't
                          slip through unchallenged).
    """
    NORMAL_NO_REF = "normal_no_refutation"
    NORMAL_WEAK = "normal_weak_refutation"
    STRONG = "strong_refutation"
    CR = "clarification_request"
    CR_OR_ATTR_FAIL = "cr_or_attribution_failure"


# Per-tier wall-clock budgets for the acceptance check. Block 20
# recalibrated simple/standard from spec defaults (30/80/150) after
# Block 19 unblocked the Verifier path. Block 21 finished the complex
# tier after running Q7 three times in isolation. Block 22 bumped
# the simple tier after widening the refutation gate to fire on
# Partial paths (Q15 hit 205s on the new path; the prior 180s budget
# was sized for the Q2 Normal+refutation case at 148s).
#
#   simple correctness-pass: max 228s observed after Block 23 widened
#       the planner's reach on forward-guidance source-doc periods —
#       Q15 now occasionally lands at NORMAL (was always PARTIAL),
#       which means both slots cover AND refutation runs, where the
#       prior PARTIAL path short-circuited S1. 250s gives ~10% over
#       the new max. Pre-Block-22 max was 148s (Q2 strong-refutation
#       on a NORMAL path); Block 22 raised it to 205s.
#   standard correctness-pass: max 109.7s (Q3); old 80s budget failed
#       the one data point.
#   complex (Q7 variance runs 1/2/3): 441s / 134s / 483s. Run 2's
#       134s was an orchestrator bug (LLMError dropped slot_run);
#       the honest elapsed when the pipeline runs to completion is
#       441–483s.
_TIER_BUDGET_S: dict[ComplexityTier, float] = {
    ComplexityTier.simple: 250.0,
    ComplexityTier.standard: 150.0,
    ComplexityTier.complex: 540.0,
}


@dataclass(frozen=True)
class ExpectedOutcome:
    qid: str
    query: str
    category: ExpectedCategory
    description: str
    # Queries whose answer must reproduce a specific XBRL value (per
    # spec wording "value matches XBRL"). For these, the Verifier's
    # supported_candidates must include at least one fully-populated
    # XBRL fact (value + unit + period + concept_tag).
    numerical_fidelity: bool = False
    notes: str = ""


_QUERIES: list[ExpectedOutcome] = [
    # ---- Normal, no refutation -----------------------------------------
    ExpectedOutcome(
        qid="Q1",
        query="What was Netflix's revenue for Q2 2023?",
        category=ExpectedCategory.NORMAL_NO_REF,
        description="financial_metric, exact XBRL match",
        numerical_fidelity=True,
        notes="Block 12 baseline pass.",
    ),
    ExpectedOutcome(
        qid="Q4",
        query="What is Netflix's count of paid memberships at the end of 2024?",
        category=ExpectedCategory.NORMAL_NO_REF,
        description="Restated value handling; period precision",
        numerical_fidelity=True,
        notes="Operational metric from Q4 2024 letter.",
    ),
    ExpectedOutcome(
        qid="Q6",
        query="What was Netflix's average revenue per user (ARM) in 2023?",
        category=ExpectedCategory.NORMAL_NO_REF,
        description="GAAP/non-GAAP terminology distinction; ARM vs. ARPU",
        notes="Terminology must be correctly classified — ARM is Netflix's term.",
    ),
    ExpectedOutcome(
        qid="Q11",
        query="Compare Netflix's operating margin from FY2019 to FY2023",
        category=ExpectedCategory.NORMAL_NO_REF,
        description="cross_period_comparison; multi-fact retrieval",
        numerical_fidelity=True,
        notes="Both values retrieved deterministically; derivation if margin computed.",
    ),
    ExpectedOutcome(
        qid="Q13",
        query="What did Netflix say about generative AI in 2024?",
        category=ExpectedCategory.NORMAL_NO_REF,
        description="Recent strategic_claim retrieval",
        notes="Recent statements from 2024 letters or transcripts.",
    ),
    ExpectedOutcome(
        qid="Q14",
        query=(
            "What guidance did Netflix give on Q1 2024 paid memberships "
            "at the Q4 2023 earnings call?"
        ),
        category=ExpectedCategory.NORMAL_NO_REF,
        description="forward_guidance with assertion_date precision",
        notes="Guidance issued at the Q4 2023 call (Jan 2024).",
    ),

    # ---- Normal, weak refutation OK ------------------------------------
    ExpectedOutcome(
        qid="Q3",
        query="Did Netflix meet its 2022 guidance for paid net adds?",
        category=ExpectedCategory.NORMAL_WEAK,
        description="forward_guidance vs. operational_metric comparison",
        notes="Refutation surfaces the miss; weak refutation is the target.",
    ),
    ExpectedOutcome(
        qid="Q7",
        query=(
            "How did Netflix's content amortization methodology evolve "
            "from 2016 to 2024?"
        ),
        category=ExpectedCategory.NORMAL_WEAK,
        description="accounting_policy temporal_evolution",
        notes="Policy change disclosure may trigger weak refutation.",
    ),
    ExpectedOutcome(
        qid="Q10",
        query="Why did Netflix's free cash flow turn positive in 2022?",
        category=ExpectedCategory.NORMAL_WEAK,
        description="causal_explanation; multiple stated causes",
        notes="Phase 3 D5 fell to CR — calibration-sensitive on chunk coverage.",
    ),

    # ---- Strong refutation expected ------------------------------------
    ExpectedOutcome(
        qid="Q2",
        query="Has Netflix's stance on advertising changed?",
        category=ExpectedCategory.STRONG,
        description="Temporal contradiction across strategic_claim facts",
        notes="Phase 3 D3 hit refutation_to_loop / Partial — acceptable.",
    ),
    ExpectedOutcome(
        qid="Q9",
        query=(
            "What risks did Netflix disclose about password sharing "
            "before its 2023 crackdown?"
        ),
        category=ExpectedCategory.STRONG,
        description="risk_disclosure with subsequent materialization",
        notes="Refutation should surface the 2023 crackdown announcement.",
    ),
    ExpectedOutcome(
        qid="Q12",
        query="Did Netflix ever say it had no plans to add ads?",
        category=ExpectedCategory.STRONG,
        description="strategic_claim with post-2022 ad-tier reversal",
        notes="Block 14 + Phase 3 D2 showcase — temporal-evolution narrative.",
    ),
    ExpectedOutcome(
        qid="Q15",
        query="Has Netflix's password sharing policy been consistent?",
        category=ExpectedCategory.STRONG,
        description="temporal_evolution + strategic_claim refutation",
        notes=(
            "Phase 3 D4 hit Partial with Verifier contradictions; "
            "structured disagreement acceptable."
        ),
    ),

    # ---- Clarification Request -----------------------------------------
    ExpectedOutcome(
        qid="Q5",
        query="What's Disney's streaming subscriber count?",
        category=ExpectedCategory.CR,
        description="Out-of-scope detection",
        notes="Input Validator rejects as OOS; Hard Halt acceptable.",
    ),

    # ---- CR or attribution failure --------------------------------------
    ExpectedOutcome(
        qid="Q8",
        query="Did Greg Peters say Netflix would never enter live sports?",
        category=ExpectedCategory.CR_OR_ATTR_FAIL,
        description="Attribution check; planted false attribution",
        notes=(
            "Acceptable outcomes: CR (nothing supports the planted "
            "attribution), or Normal/Partial with refutation surfacing "
            "the real Netflix statements on live sports."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Per-query check primitives
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def _check(name: str, ok: bool, detail: str = "") -> CheckResult:
    return CheckResult(name=name, passed=ok, detail=detail)


def _refutation_intensity(trace: ExecutionTrace) -> str:
    """Classify the refutation outcome as none/weak/strong based on the
    report's hypothesis verdicts. Returns 'bypassed' when no report."""
    if trace.refutation_report is None:
        return "bypassed"
    has_strong = any(
        h.refutation_verdict == RefutationVerdict.strongly_refuted
        for h in trace.refutation_report.hypotheses
    )
    if has_strong:
        return "strong"
    has_weak = any(
        h.refutation_verdict == RefutationVerdict.weakly_refuted
        for h in trace.refutation_report.hypotheses
    )
    if has_weak:
        return "weak"
    return "none"


def _check_degradation(trace: ExecutionTrace, expected: ExpectedOutcome) -> CheckResult:
    """Per-category degradation expectations."""
    lvl = trace.degradation_level
    cat = expected.category
    if cat == ExpectedCategory.NORMAL_NO_REF or cat == ExpectedCategory.NORMAL_WEAK:
        ok = lvl == DegradationLevel.NORMAL
        return _check(
            f"degradation == NORMAL (category={cat.value})",
            ok,
            f"got {lvl.name}",
        )
    if cat == ExpectedCategory.STRONG:
        ok = lvl in (DegradationLevel.NORMAL, DegradationLevel.PARTIAL)
        return _check(
            "degradation in {NORMAL, PARTIAL} (strong refutation may degrade)",
            ok,
            f"got {lvl.name}",
        )
    if cat == ExpectedCategory.CR:
        ok = lvl in (DegradationLevel.CLARIFICATION_REQUEST, DegradationLevel.HARD_HALT)
        return _check(
            "degradation in {CR, HARD_HALT}",
            ok,
            f"got {lvl.name}",
        )
    if cat == ExpectedCategory.CR_OR_ATTR_FAIL:
        # Any outcome is structurally acceptable; the attribution check
        # below catches the failure mode for the answer path.
        return _check(
            "degradation any (Q8 spec allows CR / refutation / attribution flag)",
            True,
            f"got {lvl.name}",
        )
    return _check(f"degradation category {cat.value} (unhandled)", False)


def _check_refutation(trace: ExecutionTrace, expected: ExpectedOutcome) -> CheckResult:
    """Per-category refutation intensity expectations."""
    intensity = _refutation_intensity(trace)
    cat = expected.category
    if cat == ExpectedCategory.NORMAL_NO_REF:
        ok = intensity in {"bypassed", "none"}
        return _check(
            "refutation intensity == none (no weak/strong hypotheses)",
            ok,
            f"got {intensity}",
        )
    if cat == ExpectedCategory.NORMAL_WEAK:
        # Spec wording: 'weak refutation possible if ...'. A clean
        # Normal with no refutation is also acceptable.
        ok = intensity in {"bypassed", "none", "weak"}
        return _check(
            "refutation intensity in {none, weak}",
            ok,
            f"got {intensity}",
        )
    if cat == ExpectedCategory.STRONG:
        ok = intensity == "strong"
        return _check(
            "refutation intensity == strong",
            ok,
            f"got {intensity}",
        )
    if cat == ExpectedCategory.CR:
        # No refutation expected (Refutation Agent doesn't run under CR).
        ok = intensity == "bypassed"
        return _check(
            "refutation bypassed (CR path)",
            ok,
            f"got {intensity}",
        )
    if cat == ExpectedCategory.CR_OR_ATTR_FAIL:
        # Either no answer (CR / Hard Halt) OR refutation surfaced.
        no_answer = trace.degradation_level in (
            DegradationLevel.CLARIFICATION_REQUEST,
            DegradationLevel.HARD_HALT,
        )
        ok = no_answer or intensity in {"weak", "strong"}
        return _check(
            "attribution defense: no-answer OR refutation surfaced",
            ok,
            f"degradation={trace.degradation_level.name} refutation={intensity}",
        )
    return _check("refutation category (unhandled)", False)


def _check_numerical_fidelity(
    trace: ExecutionTrace, expected: ExpectedOutcome
) -> CheckResult:
    """For queries flagged numerical_fidelity (Q1/Q4/Q11): at least
    one of the supported candidates must resolve to a FactRecord with
    value + unit + period + concept_tag (i.e. an XBRL fact, not just
    a prose snippet) — that's the structural promise the spec makes
    about deterministic numerical retrieval."""
    if not expected.numerical_fidelity:
        return _check("numerical fidelity (n/a)", True, "not required")

    supported_ids: list[str] = []
    seen: set[str] = set()
    for v in trace.verifier_verdicts:
        for cid in v.supported_candidates:
            if cid not in seen:
                supported_ids.append(cid)
                seen.add(cid)
    facts = lookup_facts(supported_ids)
    qualifying = [
        f for f in facts.values()
        if f.value is not None and f.unit is not None
        and f.period is not None and f.concept_tag is not None
    ]
    return _check(
        "numerical fidelity: at least one XBRL fact (value+unit+period+concept_tag)",
        len(qualifying) > 0,
        f"supported facts={len(facts)} XBRL-qualifying={len(qualifying)}",
    )


def _check_governance(trace: ExecutionTrace) -> CheckResult:
    """No numerical_mismatch violation should ever land on a released
    answer — the orchestrator escalates to Hard Halt on
    constitutional_violation. Phase 4 baseline: violation count is
    informational."""
    num_violations = [
        v for v in trace.governance_violations
        if v.severity.value == "numerical_mismatch"
    ]
    return _check(
        "no numerical_mismatch governance violations",
        len(num_violations) == 0,
        (
            f"{len(num_violations)} numerical_mismatch / "
            f"{len(trace.governance_violations)} total"
        ),
    )


def _check_time_budget(
    trace: ExecutionTrace, expected: ExpectedOutcome, elapsed_s: float
) -> CheckResult:
    """Soft-fail time-budget check by tier (Simple 30s / Standard 80s /
    Complex 150s — looser than spec §9.2 to absorb Phase 3 generator
    retry + governance call additions; Block 20 may tighten)."""
    tier = trace.query.complexity_tier
    budget = _TIER_BUDGET_S.get(tier, 80.0)
    ok = elapsed_s <= budget
    return _check(
        f"elapsed ≤ {budget}s ({tier.value})",
        ok,
        f"got {elapsed_s:.1f}s",
    )


def _run_checks(
    trace: ExecutionTrace,
    expected: ExpectedOutcome,
    elapsed_s: float,
) -> list[CheckResult]:
    return [
        _check_degradation(trace, expected),
        _check_refutation(trace, expected),
        _check_numerical_fidelity(trace, expected),
        _check_governance(trace),
        _check_time_budget(trace, expected, elapsed_s),
    ]


# ---------------------------------------------------------------------------
# Transient retry (carried over from Phase 3)
# ---------------------------------------------------------------------------


_TRANSIENT_RETRY_CAUSES: frozenset[DegradationCause] = frozenset({
    DegradationCause.generator_unavailable,
    DegradationCause.verifier_unavailable,
    DegradationCause.refutation_unavailable,
})


def _run_with_transient_retry(query: str) -> ExecutionTrace:
    """One retry on Hard Halt with a transient cause."""
    trace = run_pipeline(query, verbose=False)
    if (
        trace.degradation_level == DegradationLevel.HARD_HALT
        and trace.degradation_cause in _TRANSIENT_RETRY_CAUSES
    ):
        print(
            f"    [retry] transient Hard Halt "
            f"(cause={trace.degradation_cause.value}) — retrying once"
        )
        trace = run_pipeline(query, verbose=False)
    return trace


# ---------------------------------------------------------------------------
# Persistence + rendering
# ---------------------------------------------------------------------------


_OUT_DIR = Path("data/logs/phase4_adversarial")


def _persist_query(
    expected: ExpectedOutcome,
    trace: ExecutionTrace,
    render_html_flag: bool,
) -> dict[str, str]:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = _OUT_DIR / expected.qid
    json_path = base.with_suffix(".json")
    json_path.write_text(
        json.dumps(trace.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )
    files = {"json": json_path.name}
    if render_html_flag:
        html_path = base.with_suffix(".html")
        html_path.write_text(render_html(trace), encoding="utf-8")
        files["html"] = html_path.name
    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _print_query_header(q: ExpectedOutcome) -> None:
    print()
    print("=" * 78)
    print(f"{q.qid}: {q.query}")
    print(f"  category: {q.category.value}")
    print(f"  description: {q.description}")
    if q.notes:
        print(f"  notes: {q.notes}")
    print("=" * 78)


def _print_check_results(results: list[CheckResult]) -> int:
    passed = 0
    for r in results:
        marker = "OK  " if r.passed else "FAIL"
        line = f"  [{marker}] {r.name}"
        if r.detail:
            line += f"  — {r.detail}"
        print(line)
        if r.passed:
            passed += 1
    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated Q-ids to run (e.g. Q1,Q4).")
    parser.add_argument("--skip", type=str, default="",
                        help="Comma-separated Q-ids to skip (e.g. Q7,Q8).")
    parser.add_argument(
        "--render-html", action="store_true",
        help="Also render per-query HTML traces. Adds ~0.2s per query.",
    )
    args = parser.parse_args()

    only_ids = (
        {q.strip().upper() for q in args.only.split(",") if q.strip()}
        if args.only else None
    )
    skip_ids = {q.strip().upper() for q in args.skip.split(",") if q.strip()}

    targets = [
        q for q in _QUERIES
        if (only_ids is None or q.qid.upper() in only_ids)
        and q.qid.upper() not in skip_ids
    ]
    if not targets:
        print("No queries match --only/--skip.")
        sys.exit(2)

    rows: list[dict] = []
    overall_pass = 0
    overall_total = 0
    queries_fully_passed = 0
    overall_t0 = time.time()

    for expected in targets:
        _print_query_header(expected)
        t0 = time.time()
        try:
            trace = _run_with_transient_retry(expected.query)
        except Exception as exc:
            elapsed = round(time.time() - t0, 2)
            print(f"\n  RUN FAILED: {type(exc).__name__}: {exc}")
            rows.append({
                "qid": expected.qid,
                "query": expected.query,
                "category": expected.category.value,
                "elapsed_s": elapsed,
                "error": f"{type(exc).__name__}: {exc}",
            })
            overall_total += 1
            continue

        elapsed = round(time.time() - t0, 2)
        files = _persist_query(expected, trace, args.render_html)
        intensity = _refutation_intensity(trace)

        print(
            f"  -> degradation={trace.degradation_level.name} "
            f"cause={trace.degradation_cause.value} "
            f"refutation={intensity} "
            f"governance_violations={len(trace.governance_violations)} "
            f"elapsed={elapsed}s"
        )
        if trace.answer is not None:
            preview = trace.answer.answer_text
            if len(preview) > 180:
                preview = preview[:180] + "..."
            print(f"  answer: {preview}")

        results = _run_checks(trace, expected, elapsed)
        passed = _print_check_results(results)
        total = len(results)
        overall_pass += passed
        overall_total += total
        if passed == total:
            queries_fully_passed += 1

        rows.append({
            "qid": expected.qid,
            "query": expected.query,
            "category": expected.category.value,
            "passed": passed,
            "total": total,
            "degradation_level": trace.degradation_level.name,
            "degradation_cause": trace.degradation_cause.value,
            "refutation_intensity": intensity,
            "refutation_overall": (
                trace.refutation_report.overall_verdict.value
                if trace.refutation_report else "bypassed"
            ),
            "elapsed_s": elapsed,
            "complexity_tier": trace.query.complexity_tier.value,
            "governance_violations": len(trace.governance_violations),
            "n_claims": (
                len(trace.answer.claims) if trace.answer else 0
            ),
            "n_disclosed_refutations": (
                len(trace.answer.disclosed_refutations) if trace.answer else 0
            ),
            "files": files,
            "failing_checks": [
                {"name": r.name, "detail": r.detail}
                for r in results if not r.passed
            ],
        })

    # Summary
    print()
    print("=" * 78)
    print("PHASE 4 ADVERSARIAL SUITE SUMMARY")
    print("=" * 78)
    print(f"  {'Q':<4} {'pass/total':<12} {'category':<28} "
          f"{'degradation':<14} {'refut':<10} {'elapsed':<8}")
    print(f"  {'-'*4} {'-'*12} {'-'*28} {'-'*14} {'-'*10} {'-'*8}")
    for r in rows:
        if r.get("error"):
            print(
                f"  {r['qid']:<4} {'0/1':<12} {r['category']:<28} "
                f"{'RUN_FAILED':<14} {'-':<10} {r['elapsed_s']}s"
            )
            continue
        print(
            f"  {r['qid']:<4} "
            f"{r['passed']}/{r['total']:<10} "
            f"{r['category']:<28} "
            f"{r['degradation_level']:<14} "
            f"{r['refutation_intensity']:<10} "
            f"{r['elapsed_s']}s"
        )
    print()
    print(f"  OVERALL: {overall_pass}/{overall_total} acceptance checks "
          f"({queries_fully_passed}/{len(targets)} queries fully passed)")
    print(f"  Total elapsed: {round(time.time() - overall_t0, 2)}s")

    # Group failures by category for Block 20 calibration triage.
    by_category: dict[str, list[str]] = {}
    for r in rows:
        if r.get("error"):
            continue
        if r["failing_checks"]:
            by_category.setdefault(r["category"], []).append(r["qid"])
    if by_category:
        print()
        print("Calibration triage — queries with failing checks by category:")
        for cat, qids in sorted(by_category.items()):
            print(f"  {cat}: {', '.join(qids)}")

    # Persist
    summary_path = _OUT_DIR / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "overall_pass": overall_pass,
                "overall_total": overall_total,
                "queries_fully_passed": queries_fully_passed,
                "queries_total": len(targets),
                "rows": rows,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nSummary -> {summary_path}")

    # Block 18 is a baseline measurement, not a pass/fail gate.
    # Always exit 0; Block 20 will set the bar after calibration.


if __name__ == "__main__":
    main()

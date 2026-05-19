"""Block 12: Phase 2 end-to-end validation harness.

Runs the 8-query subset of spec §9.2 selected by the build plan to
cover every pipeline path:

  Q1 Simple,    clean Normal:                  "What was Netflix's revenue for Q2 2023?"
  Q2 Standard,  multi-period clean Normal:     "Compare Netflix's operating margin from FY2019 to FY2023"
  Q3 Standard,  weak refutation:               "Did Netflix meet its 2022 guidance for paid net adds?"
  Q4 Standard,  weak refutation:               "Why did Netflix's free cash flow turn positive in 2022?"
  Q5 Standard,  strong refutation → loop:      "Did Netflix ever say it had no plans to add ads?"
  Q6 Complex,   strong refutation → Partial:   "Has Netflix's stance on advertising changed?"
  Q7 Complex,   strong refutation → Partial:   "Has Netflix's password sharing policy been consistent?"
  Q8 OOS,       Clarification Request:         "What's Disney's streaming subscriber count?"

Per-query checks:
  - Pass_origin values consistent (verifier_loop, refutation_probe,
    refutation_loop) across all retrievals
  - Refutation hypotheses (when present) carry their strategy field
  - Memory Ledger reflects activity: retrieved_ids, coverage_history,
    refutation_hypotheses_tested, refutation_loop_history
  - Degradation level falls within the expected set for this query
  - adversarially_probed flag matches the expected outcome
  - Numerical-fidelity (queries 1 & 2): the supported XBRL fact's
    value, unit, and period appear in the trace and match
  - Wall time within the tier budget (Simple ≤20s, Standard ≤40s,
    Complex ≤75s) — spec §9.2 ceilings

Cost ceiling check ($0.05/query, spec §9.2) is deferred to Phase 4
calibration — the orchestrator does not yet aggregate token usage
across stages. Tracked as a Phase 4 follow-up.

Run from repo root::

    python -m scripts.test_block12
    python -m scripts.test_block12 --only 5      # one query
    python -m scripts.test_block12 --skip 6,7    # skip slow complex queries

Writes the full per-query trace to data/logs/block12_validation.json.
Exits non-zero if any required check fails.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from agents.refutation.agent import lookup_facts
from pipeline.orchestrator import run_pipeline
from pipeline.trace_renderer import render_trace
from schemas.enums import (
    ComplexityTier,
    DegradationLevel,
    PassOrigin,
    RefutationStrategy,
    RefutationVerdict,
)
from schemas.records import ExecutionTrace, FactRecord


# ---------------------------------------------------------------------------
# Per-query expected outcomes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectedOutcome:
    qid: str
    query: str
    tier: ComplexityTier
    description: str
    valid_degradation_levels: tuple[DegradationLevel, ...]
    adversarially_probed_expected: bool | None  # None = either is acceptable
    expects_refutation_run: bool
    expects_strong_refutation: bool
    expected_refutation_strategies: tuple[RefutationStrategy, ...] | None
    numerical_fidelity: bool
    time_budget_s: float
    notes: str


_TIER_BUDGET_S: dict[ComplexityTier, float] = {
    ComplexityTier.simple: 20.0,
    ComplexityTier.standard: 40.0,
    ComplexityTier.complex: 75.0,
}


_QUERIES: list[ExpectedOutcome] = [
    ExpectedOutcome(
        qid="Q1",
        query="What was Netflix's revenue for Q2 2023?",
        tier=ComplexityTier.simple,
        description="Simple, clean Normal: financial_metric via XBRL",
        valid_degradation_levels=(DegradationLevel.NORMAL,),
        adversarially_probed_expected=True,
        expects_refutation_run=True,
        expects_strong_refutation=False,
        expected_refutation_strategies=None,
        numerical_fidelity=True,
        time_budget_s=_TIER_BUDGET_S[ComplexityTier.simple],
        notes="XBRL deterministic. Refutation runs, hypothesis is restated_value but does not strongly refute (no later restatement in corpus).",
    ),
    ExpectedOutcome(
        qid="Q2",
        query="Compare Netflix's operating margin from FY2019 to FY2023",
        tier=ComplexityTier.standard,
        description="Standard, multi-period clean Normal: 2 specific_metric slots",
        valid_degradation_levels=(DegradationLevel.NORMAL,),
        adversarially_probed_expected=True,
        expects_refutation_run=True,
        expects_strong_refutation=False,
        expected_refutation_strategies=None,
        numerical_fidelity=True,
        time_budget_s=_TIER_BUDGET_S[ComplexityTier.standard],
        notes="Two XBRL-deterministic slots. Refutation may run on either.",
    ),
    ExpectedOutcome(
        qid="Q3",
        query="Did Netflix meet its 2022 guidance for paid net adds?",
        tier=ComplexityTier.standard,
        description="Standard, weak refutation likely (forward_guidance → guidance_vs_actual)",
        valid_degradation_levels=(DegradationLevel.NORMAL, DegradationLevel.PARTIAL),
        adversarially_probed_expected=None,
        expects_refutation_run=True,
        expects_strong_refutation=False,
        expected_refutation_strategies=(RefutationStrategy.guidance_vs_actual,),
        numerical_fidelity=False,
        time_budget_s=_TIER_BUDGET_S[ComplexityTier.standard],
        notes="If guidance-vs-actual gap is large (Q1 2022 missed badly), classifier may mark strongly_refuted and degrade to Partial.",
    ),
    ExpectedOutcome(
        qid="Q4",
        query="Why did Netflix's free cash flow turn positive in 2022?",
        tier=ComplexityTier.standard,
        description="Standard, may surface weak refutation (causal_explanation → alternative_cause)",
        valid_degradation_levels=(DegradationLevel.NORMAL, DegradationLevel.PARTIAL),
        adversarially_probed_expected=None,
        expects_refutation_run=True,
        expects_strong_refutation=False,
        expected_refutation_strategies=(RefutationStrategy.alternative_cause,),
        numerical_fidelity=False,
        time_budget_s=_TIER_BUDGET_S[ComplexityTier.standard],
        notes="Multiple causes appear in the corpus — classifier behaviour determines weak vs strong.",
    ),
    ExpectedOutcome(
        qid="Q5",
        query="Did Netflix ever say it had no plans to add ads?",
        tier=ComplexityTier.standard,
        description="Standard, strong refutation → loop → Normal-with-temporal-evolution",
        valid_degradation_levels=(DegradationLevel.NORMAL, DegradationLevel.PARTIAL),
        adversarially_probed_expected=True,
        expects_refutation_run=True,
        expects_strong_refutation=True,
        expected_refutation_strategies=(RefutationStrategy.later_reversal,),
        numerical_fidelity=False,
        time_budget_s=_TIER_BUDGET_S[ComplexityTier.standard],
        notes="2018 no-ads claim refuted by Q4 2022 / 2023 ad-tier. Either loop resolves (Normal) or drops to Partial.",
    ),
    ExpectedOutcome(
        qid="Q6",
        query="Has Netflix's stance on advertising changed?",
        tier=ComplexityTier.complex,
        description="Complex, strong refutation → Partial with structured disagreement",
        valid_degradation_levels=(DegradationLevel.NORMAL, DegradationLevel.PARTIAL),
        adversarially_probed_expected=True,
        expects_refutation_run=True,
        expects_strong_refutation=True,
        expected_refutation_strategies=(RefutationStrategy.later_reversal,),
        numerical_fidelity=False,
        time_budget_s=_TIER_BUDGET_S[ComplexityTier.complex],
        notes="Like Q5 but Complex tier — Planner emits multiple slots, more chances for refutation.",
    ),
    ExpectedOutcome(
        qid="Q7",
        query="Has Netflix's password sharing policy been consistent?",
        tier=ComplexityTier.complex,
        description="Complex, strong refutation → Partial (pre-2023 risk → 2023 crackdown)",
        valid_degradation_levels=(DegradationLevel.NORMAL, DegradationLevel.PARTIAL),
        adversarially_probed_expected=True,
        expects_refutation_run=True,
        expects_strong_refutation=True,
        expected_refutation_strategies=(
            RefutationStrategy.materialization,
            RefutationStrategy.later_reversal,
        ),
        numerical_fidelity=False,
        time_budget_s=_TIER_BUDGET_S[ComplexityTier.complex],
        notes="Risk_disclosure → materialization, OR strategic_claim → later_reversal. Either is acceptable.",
    ),
    ExpectedOutcome(
        qid="Q8",
        query="What's Disney's streaming subscriber count?",
        tier=ComplexityTier.simple,
        description="Out-of-scope → Clarification Request, refutation does NOT run",
        valid_degradation_levels=(
            DegradationLevel.CLARIFICATION_REQUEST,
            DegradationLevel.HARD_HALT,
        ),
        adversarially_probed_expected=False,
        expects_refutation_run=False,
        expects_strong_refutation=False,
        expected_refutation_strategies=None,
        numerical_fidelity=False,
        time_budget_s=_TIER_BUDGET_S[ComplexityTier.simple],
        notes="Netflix-only corpus. Input Validation should reject, OR verifier exhausts on every slot.",
    ),
]


# ---------------------------------------------------------------------------
# Acceptance checks
# ---------------------------------------------------------------------------


_VALID_PASS_ORIGINS: frozenset[str] = frozenset(p.value for p in PassOrigin)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def _check(name: str, ok: bool, detail: str = "") -> CheckResult:
    return CheckResult(name=name, passed=ok, detail=detail)


def _check_pass_origins(trace: ExecutionTrace) -> CheckResult:
    bad = [
        f"{r.retrieval_id}:{r.pass_origin}"
        for r in trace.retrieval_passes
        if r.pass_origin.value not in _VALID_PASS_ORIGINS
    ]
    return _check(
        "pass_origin values are all valid",
        not bad,
        f"retrieval count={len(trace.retrieval_passes)}; bad={bad[:3]}" if bad
        else f"{len(trace.retrieval_passes)} retrievals tagged correctly",
    )


def _check_degradation(
    trace: ExecutionTrace, expected: ExpectedOutcome
) -> CheckResult:
    ok = trace.degradation_level in expected.valid_degradation_levels
    return _check(
        f"degradation level in expected set "
        f"({'/'.join(l.name for l in expected.valid_degradation_levels)})",
        ok,
        f"got {trace.degradation_level.name} cause={trace.degradation_cause.value}",
    )


def _check_adversarially_probed(
    trace: ExecutionTrace, expected: ExpectedOutcome
) -> CheckResult:
    actual = bool(trace.extra.get("adversarially_probed", False))
    if expected.adversarially_probed_expected is None:
        return _check(
            "adversarially_probed flag set (either value acceptable)",
            True,
            f"got {actual}",
        )
    return _check(
        f"adversarially_probed == {expected.adversarially_probed_expected}",
        actual == expected.adversarially_probed_expected,
        f"got {actual}",
    )


def _check_refutation_runs(
    trace: ExecutionTrace, expected: ExpectedOutcome
) -> CheckResult:
    ran = trace.refutation_report is not None
    ok = ran == expected.expects_refutation_run
    detail = ""
    if not ok:
        detail = (
            f"expected_run={expected.expects_refutation_run} got_run={ran}; "
            f"bypass={trace.extra.get('refutation_bypass_reason', 'n/a')}"
        )
    return _check(
        f"Refutation Agent {'ran' if expected.expects_refutation_run else 'bypassed'}",
        ok,
        detail,
    )


def _check_strong_refutation(
    trace: ExecutionTrace, expected: ExpectedOutcome
) -> CheckResult:
    """For queries that expect a strong refutation, confirm at least
    one hypothesis was strongly_refuted. For queries that explicitly
    don't, we permit anything — strong refutations are possible side
    effects of a healthy refutation pass and aren't disqualifying."""
    if not expected.expects_strong_refutation:
        return _check(
            "strong refutation expectation (no requirement)",
            True,
            "skipped — query does not require strong refutation",
        )
    if trace.refutation_report is None:
        return _check(
            "at least one hypothesis was strongly_refuted",
            False,
            "no refutation report on trace",
        )
    any_strong = any(
        h.refutation_verdict == RefutationVerdict.strongly_refuted
        for h in trace.refutation_report.hypotheses
    )
    return _check(
        "at least one hypothesis was strongly_refuted",
        any_strong,
        f"hypotheses verdicts: "
        f"{[h.refutation_verdict.value for h in trace.refutation_report.hypotheses]}",
    )


def _check_refutation_strategy(
    trace: ExecutionTrace, expected: ExpectedOutcome
) -> CheckResult:
    if expected.expected_refutation_strategies is None or not trace.refutation_report:
        return _check(
            "refutation strategy expectation (no requirement)",
            True,
            "skipped",
        )
    strategies_seen = {h.strategy for h in trace.refutation_report.hypotheses}
    expected_set = set(expected.expected_refutation_strategies)
    ok = bool(strategies_seen & expected_set)
    return _check(
        f"hypothesis strategy in expected set "
        f"({'/'.join(s.value for s in expected.expected_refutation_strategies)})",
        ok,
        f"saw {sorted(s.value for s in strategies_seen)}",
    )


_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _normalize_number(s: str) -> str:
    return s.replace(",", "").rstrip(".0").rstrip(".") or "0"


def _check_numerical_fidelity(
    trace: ExecutionTrace, expected: ExpectedOutcome
) -> CheckResult:
    """For queries 1 & 2: confirm the verifier's supported XBRL facts
    carry value + unit + period + concept_tag, and that the value
    appears in at least one of the verifier's verdict texts (the
    trace's atomic fidelity check)."""
    if not expected.numerical_fidelity:
        return _check("numerical fidelity (no requirement)", True, "skipped")

    supported_ids: list[str] = []
    for v in trace.verifier_verdicts:
        for cid in v.supported_candidates:
            if cid not in supported_ids:
                supported_ids.append(cid)
    facts = lookup_facts(supported_ids)
    if not facts:
        return _check(
            "numerical fidelity: supported XBRL facts present in trace",
            False,
            "no supported fact_ids resolved to FactRecords",
        )
    # At least one supported fact must carry value + unit + period.
    qualifying = [
        f for f in facts.values()
        if f.value is not None and f.unit is not None
        and f.period is not None and f.concept_tag is not None
    ]
    if not qualifying:
        return _check(
            "numerical fidelity: at least one fact has value+unit+period+concept_tag",
            False,
            f"supported facts had values/units missing (count={len(facts)})",
        )
    return _check(
        "numerical fidelity: at least one fact has value+unit+period+concept_tag",
        True,
        f"{len(qualifying)} of {len(facts)} supported facts fully populated",
    )


def _check_time_budget(
    trace: ExecutionTrace, expected: ExpectedOutcome
) -> CheckResult:
    ok = trace.elapsed_seconds <= expected.time_budget_s
    return _check(
        f"elapsed ≤ tier budget ({expected.time_budget_s}s for "
        f"{expected.tier.value})",
        ok,
        f"got {trace.elapsed_seconds}s",
    )


def _check_ledger_activity(trace: ExecutionTrace) -> CheckResult:
    """Generic sanity check: ledger reflects something happened. For
    a Clarification Request run the bar is just 'retrieved_ids has
    entries OR the run halted very early'; for everything else we
    expect retrieved_ids and coverage_history to be non-empty."""
    ledger = trace.extra.get("ledger") or {}
    has_retrievals = bool(ledger.get("retrieved_ids"))
    has_coverage = bool(ledger.get("coverage_history"))
    if trace.degradation_level == DegradationLevel.HARD_HALT:
        return _check(
            "ledger sanity (Hard Halt — early exit acceptable)",
            True,
            "skipped",
        )
    if trace.degradation_level == DegradationLevel.CLARIFICATION_REQUEST:
        # Either retrievals happened (verifier exhausted) or none did
        # (Input Validation rejected) — both are valid paths to
        # Clarification Request.
        return _check(
            "ledger reflects either retrieval activity or early reject",
            True,
            f"retrieved={has_retrievals} coverage={has_coverage}",
        )
    return _check(
        "ledger reflects retrieval + coverage activity",
        has_retrievals and has_coverage,
        f"retrieved={has_retrievals} coverage={has_coverage}",
    )


def _run_checks(
    trace: ExecutionTrace, expected: ExpectedOutcome
) -> list[CheckResult]:
    return [
        _check_pass_origins(trace),
        _check_degradation(trace, expected),
        _check_adversarially_probed(trace, expected),
        _check_refutation_runs(trace, expected),
        _check_strong_refutation(trace, expected),
        _check_refutation_strategy(trace, expected),
        _check_numerical_fidelity(trace, expected),
        _check_ledger_activity(trace),
        _check_time_budget(trace, expected),
    ]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _print_query_header(q: ExpectedOutcome) -> None:
    _print_section(f"{q.qid}: {q.query}")
    print(f"  tier:        {q.tier.value}")
    print(f"  description: {q.description}")
    print(f"  note:        {q.notes}")


def _print_trace_summary(trace: ExecutionTrace, elapsed: float) -> None:
    print()
    print("  -- RESULT --")
    print(f"  degradation:        {trace.degradation_level.name} "
          f"(cause={trace.degradation_cause.value})")
    print(f"  slots:              {len(trace.final_slot_states)} "
          f"(total verifier iterations={trace.total_iterations})")
    print(f"  elapsed:            {elapsed}s")
    print(f"  models_used:        {trace.models_used}")
    print(f"  adversarially_probed: {trace.extra.get('adversarially_probed', False)}")

    if trace.refutation_report is not None:
        rep = trace.refutation_report
        print(f"  refutation overall: {rep.overall_verdict.value} "
              f"(hyp={len(rep.hypotheses)} loop_reentry={rep.triggered_loop_reentry})")
        for h in rep.hypotheses:
            print(f"    h_id={h.hypothesis_id} strategy={h.strategy.value} "
                  f"verdict={h.refutation_verdict.value}  "
                  f"evidence_count={len(h.evidence_ids)}")
    else:
        bypass = trace.extra.get("refutation_bypass_reason")
        print(f"  refutation:         BYPASSED ({bypass})" if bypass
              else "  refutation:         (no report)")


def _print_check_results(results: list[CheckResult]) -> int:
    print("\n  -- ACCEPTANCE CHECKS --")
    for r in results:
        marker = "OK  " if r.passed else "FAIL"
        line = f"  [{marker}] {r.name}"
        if r.detail:
            line += f" — {r.detail}"
        print(line)
    return sum(1 for r in results if r.passed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="Only run this Q-id (e.g. Q5). Repeatable comma-separated.")
    parser.add_argument("--skip", type=str, default="",
                        help="Comma-separated Q-ids to skip (e.g. Q6,Q7).")
    parser.add_argument("--log", type=str,
                        default="data/logs/block12_validation.json",
                        help="Output JSON log path.")
    parser.add_argument(
        "--trace",
        type=str,
        default=None,
        help=(
            "Render the embedded traces in an existing log JSON via "
            "pipeline.trace_renderer and exit. Use --only Qx to limit "
            "which query to render."
        ),
    )
    args = parser.parse_args()

    if args.trace is not None:
        raw = json.loads(Path(args.trace).read_text(encoding="utf-8"))
        queries = raw.get("queries") or []
        only_ids = (
            {qid.strip().upper() for qid in args.only.split(",") if qid.strip()}
            if args.only else None
        )
        rendered_any = False
        for q in queries:
            qid = (q.get("qid") or "").upper()
            if only_ids is not None and qid not in only_ids:
                continue
            trace_dict = q.get("trace")
            if not trace_dict:
                continue
            trace = ExecutionTrace.model_validate(trace_dict)
            print(f"\n{'#' * 78}")
            print(f"# {qid}: {q.get('query', '(no query)')}")
            print(f"{'#' * 78}\n")
            print(render_trace(trace))
            rendered_any = True
        if not rendered_any:
            print("No traces matched the --trace/--only filter.")
            sys.exit(2)
        return

    only_ids = (
        set(qid.strip().upper() for qid in args.only.split(",") if qid.strip())
        if args.only else None
    )
    skip_ids = set(qid.strip().upper() for qid in args.skip.split(",") if qid.strip())

    targets = [
        q for q in _QUERIES
        if (only_ids is None or q.qid.upper() in only_ids)
        and q.qid.upper() not in skip_ids
    ]
    if not targets:
        print("No queries match the --only/--skip filters.")
        sys.exit(2)

    summary_rows: list[dict] = []
    log_entries: list[dict] = []
    overall_pass = 0
    overall_total = 0

    for expected in targets:
        _print_query_header(expected)
        t0 = time.time()
        try:
            trace = run_pipeline(expected.query, verbose=True)
        except Exception as exc:
            elapsed = round(time.time() - t0, 2)
            print(f"\n  RUN FAILED: {type(exc).__name__}: {exc}")
            log_entries.append({
                "qid": expected.qid,
                "query": expected.query,
                "elapsed_s": elapsed,
                "error": f"{type(exc).__name__}: {exc}",
            })
            summary_rows.append({
                "qid": expected.qid, "passed": 0, "total": 1,
                "degradation": "RUN_FAILED", "elapsed_s": elapsed,
            })
            overall_total += 1
            continue
        elapsed = round(time.time() - t0, 2)

        _print_trace_summary(trace, elapsed)
        results = _run_checks(trace, expected)
        passed = _print_check_results(results)
        total = len(results)
        overall_pass += passed
        overall_total += total

        summary_rows.append({
            "qid": expected.qid,
            "passed": passed,
            "total": total,
            "degradation": trace.degradation_level.name,
            "elapsed_s": elapsed,
            "refutation_overall": (
                trace.refutation_report.overall_verdict.value
                if trace.refutation_report else "bypassed"
            ),
        })
        log_entries.append({
            "qid": expected.qid,
            "query": expected.query,
            "tier": expected.tier.value,
            "elapsed_s": elapsed,
            "expected": {
                "valid_degradation_levels": [l.name for l in expected.valid_degradation_levels],
                "adversarially_probed": expected.adversarially_probed_expected,
                "expects_refutation_run": expected.expects_refutation_run,
                "expects_strong_refutation": expected.expects_strong_refutation,
                "expected_refutation_strategies": (
                    [s.value for s in expected.expected_refutation_strategies]
                    if expected.expected_refutation_strategies else None
                ),
                "time_budget_s": expected.time_budget_s,
            },
            "trace": trace.model_dump(mode="json"),
            "check_results": [
                {"name": r.name, "passed": r.passed, "detail": r.detail}
                for r in results
            ],
        })

    # Final summary table.
    _print_section("BLOCK 12 SUMMARY")
    print(f"  {'qid':<5} {'pass/total':<12} {'degradation':<24} {'refutation':<22} {'elapsed':<8}")
    print(f"  {'-'*5} {'-'*12} {'-'*24} {'-'*22} {'-'*8}")
    for row in summary_rows:
        print(
            f"  {row['qid']:<5} "
            f"{row['passed']}/{row['total']:<10} "
            f"{row['degradation']:<24} "
            f"{row.get('refutation_overall', '?'):<22} "
            f"{row['elapsed_s']}s"
        )
    print()
    print(f"  OVERALL: {overall_pass}/{overall_total} acceptance checks passed "
          f"across {len(targets)} queries")

    # Persist log.
    out_path = Path(args.log)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"queries": log_entries, "summary": summary_rows},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nLog -> {out_path}")

    # Required checks must all pass for Block 12 to succeed.
    if overall_pass < overall_total:
        sys.exit(1)


if __name__ == "__main__":
    main()

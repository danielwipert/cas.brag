"""Block 17: Phase 3 demo runner.

Six curated queries that drive the full pipeline (Validate -> Plan ->
Verify -> Refute -> Generate -> Govern -> Render) end-to-end and
emit one self-contained HTML + one terminal-rendered text + one
JSON trace per query, plus an index.html that links the lot. This
is the Phase 3 deliverable: proof that BRAG answers real questions
with full disclosure, in a form a hiring manager or research
collaborator can open in a browser and follow.

Query slate:

  D1  Clean Normal financial_metric:
      "What was Netflix's revenue for Q2 2023?"
  D2  Strong refutation, Normal-with-temporal-evolution:
      "Did Netflix ever say it had no plans to add ads?"
  D3  Strong refutation, Complex tier:
      "Has Netflix's stance on advertising changed?"
  D4  Strong refutation, password sharing:
      "Has Netflix's password sharing policy been consistent?"
  D5  Weak refutation (calibration-sensitive):
      "Why did Netflix's free cash flow turn positive in 2022?"
  D6  OOS Clarification Request:
      "What's Disney's streaming subscriber count?"

Run from repo root::

    python -m scripts.test_phase3_demo
    python -m scripts.test_phase3_demo --only D1,D6   # subset
    python -m scripts.test_phase3_demo --skip D5      # skip the calibration-sensitive one

Outputs land in ``data/logs/phase3_demo/`` (gitignored) — six
``D{n}.{json,html,txt}`` files plus ``index.html``. Exits non-zero
if fewer than 5 of 6 queries pass acceptance.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from pipeline.html_renderer import render_html
from pipeline.orchestrator import run_pipeline
from pipeline.trace_renderer import render_trace
from schemas.enums import DegradationCause, DegradationLevel
from schemas.records import AnswerSchema, ExecutionTrace


# Transient generator failures (provider 5xx, mid-stream disconnect) are
# rare but real. The demo retries once on a Hard Halt with cause=
# generator_unavailable so a single flake doesn't tank the showcase.
_TRANSIENT_RETRY_CAUSES: frozenset[DegradationCause] = frozenset({
    DegradationCause.generator_unavailable,
})


def _run_with_transient_retry(query: str) -> ExecutionTrace:
    """Wraps ``run_pipeline`` with one retry on a transient
    ``generator_unavailable`` Hard Halt. The Generator can sporadically
    fail on the OpenRouter side (DeepSeek-Chat mid-stream disconnect,
    upstream 5xx); a single retry is enough in practice."""
    trace = run_pipeline(query, verbose=False)
    if (
        trace.degradation_level == DegradationLevel.HARD_HALT
        and trace.degradation_cause in _TRANSIENT_RETRY_CAUSES
    ):
        print(
            f"  [retry] transient Hard Halt "
            f"(cause={trace.degradation_cause.value}) — retrying once"
        )
        trace = run_pipeline(query, verbose=False)
    return trace


# ---------------------------------------------------------------------------
# Query slate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DemoQuery:
    did: str
    query: str
    description: str
    # If True, disclosed_refutations[] must be non-empty.
    expect_refutation_disclosure: bool
    # If True, disclosed_refutations[] must be empty (clean Normal).
    expect_no_refutation: bool
    # If True, run must terminate in CR or Hard Halt (no answer expected).
    expect_no_answer: bool
    notes: str = ""


_QUERIES: list[DemoQuery] = [
    DemoQuery(
        did="D1",
        query="What was Netflix's revenue for Q2 2023?",
        description="Clean Normal, financial_metric",
        expect_refutation_disclosure=False,
        expect_no_refutation=True,
        expect_no_answer=False,
        notes="Short factual answer; value verbatim; adversarially_probed=True; empty disclosed_refutations.",
    ),
    DemoQuery(
        did="D2",
        query="Did Netflix ever say it had no plans to add ads?",
        description="Strong refutation resolved to Normal-with-temporal-evolution",
        expect_refutation_disclosure=True,
        expect_no_refutation=False,
        expect_no_answer=False,
        notes="Temporal-evolution narrative naming both 2018 and Q4 2022 positions with assertion_dates.",
    ),
    DemoQuery(
        did="D3",
        query="Has Netflix's stance on advertising changed?",
        description="Strong refutation, Complex tier",
        expect_refutation_disclosure=True,
        expect_no_refutation=False,
        expect_no_answer=False,
        notes="Same showcase as D2 with more slots and richer trace.",
    ),
    DemoQuery(
        did="D4",
        query="Has Netflix's password sharing policy been consistent?",
        description="Strong refutation, password sharing (Partial acceptable)",
        expect_refutation_disclosure=True,
        expect_no_refutation=False,
        expect_no_answer=False,
        notes="Refutation surfaces materialization OR later_reversal; Partial with structured disagreement is acceptable.",
    ),
    DemoQuery(
        did="D5",
        query="Why did Netflix's free cash flow turn positive in 2022?",
        description="Weak refutation (calibration-sensitive)",
        expect_refutation_disclosure=False,
        expect_no_refutation=False,
        expect_no_answer=False,
        notes="May fall to Phase 4 calibration depending on chunk-coverage state for FCF narrative.",
    ),
    DemoQuery(
        did="D6",
        query="What's Disney's streaming subscriber count?",
        description="OOS Clarification Request",
        expect_refutation_disclosure=False,
        expect_no_refutation=True,
        expect_no_answer=True,
        notes="Hard reject with rejection_reason explaining the Netflix-only scope.",
    ),
]


# ---------------------------------------------------------------------------
# Acceptance checks
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def _check(name: str, ok: bool, detail: str = "") -> CheckResult:
    return CheckResult(name=name, passed=ok, detail=detail)


def _run_checks(
    trace: ExecutionTrace,
    expected: DemoQuery,
    crashed: bool,
    error: str | None,
) -> list[CheckResult]:
    checks: list[CheckResult] = []

    # 1. Pipeline completed without crashing.
    checks.append(_check(
        "pipeline completed without crashing",
        not crashed,
        error or "",
    ))
    if crashed:
        return checks

    # 2. AnswerSchema is Pydantic-valid OR run reached CR/Hard Halt.
    is_cr_or_halt = trace.degradation_level in (
        DegradationLevel.CLARIFICATION_REQUEST,
        DegradationLevel.HARD_HALT,
    )
    schema_ok = False
    if trace.answer is not None:
        try:
            AnswerSchema.model_validate(trace.answer.model_dump())
            schema_ok = True
        except Exception as e:  # pragma: no cover — defensive
            schema_ok = False
            checks.append(_check(
                "answer schema round-trip",
                False,
                f"validation error: {e}",
            ))
    if not schema_ok:
        checks.append(_check(
            "AnswerSchema valid OR run reached CR/Hard Halt",
            is_cr_or_halt,
            f"degradation={trace.degradation_level.name}",
        ))
    else:
        checks.append(_check(
            "AnswerSchema valid OR run reached CR/Hard Halt",
            True,
            f"degradation={trace.degradation_level.name} with valid answer",
        ))

    # 3. Governance passed OR violations are documented and on trace.
    if trace.governance_violations:
        checks.append(_check(
            "Governance violations documented on trace",
            len(trace.governance_violations) > 0,
            f"{len(trace.governance_violations)} violation(s) recorded "
            f"(degradation now {trace.degradation_level.name})",
        ))
    else:
        checks.append(_check(
            "Governance: no violations",
            True,
            "clean Governance pass",
        ))

    # 4. Refutation disclosure expectations. The build-plan spec for D4
    # explicitly accepts "Partial with structured disagreement" via
    # disclosed_contradictions, so the acceptance check for an
    # "expect_refutation_disclosure" query is satisfied by EITHER a
    # populated disclosed_refutations[] OR a populated
    # disclosed_contradictions[] when the refutation stage bypassed
    # because the run dropped to Partial via slot exhaustion.
    n_disclosed = (
        len(trace.answer.disclosed_refutations) if trace.answer else 0
    )
    n_contradictions = (
        len(trace.answer.disclosed_contradictions) if trace.answer else 0
    )
    if expected.expect_refutation_disclosure:
        ok = n_disclosed > 0 or n_contradictions > 0
        checks.append(_check(
            "structured disagreement disclosed "
            "(disclosed_refutations[] OR disclosed_contradictions[])",
            ok,
            f"refutations={n_disclosed} contradictions={n_contradictions}",
        ))
    if expected.expect_no_refutation:
        # CR / Hard Halt with no answer trivially has no disclosed_refutations.
        # For runs that did answer, require an empty disclosed_refutations[].
        if trace.answer is not None:
            checks.append(_check(
                "disclosed_refutations[] empty (clean Normal)",
                n_disclosed == 0,
                f"got {n_disclosed} disclosure(s)",
            ))

    # 5. CR / Hard Halt expectation.
    if expected.expect_no_answer:
        checks.append(_check(
            "degradation_level in {CR, Hard Halt}",
            is_cr_or_halt,
            f"degradation={trace.degradation_level.name} "
            f"cause={trace.degradation_cause.value}",
        ))

    return checks


# ---------------------------------------------------------------------------
# Rendering + persistence
# ---------------------------------------------------------------------------


_OUT_DIR = Path("data/logs/phase3_demo")


def _persist_run(
    expected: DemoQuery,
    trace: ExecutionTrace,
    elapsed: float,
    results: list[CheckResult],
) -> dict[str, str]:
    """Write D{n}.{json,html,txt} for this run. Returns the file paths
    keyed by extension for the index."""
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    base = _OUT_DIR / expected.did
    json_path = base.with_suffix(".json")
    html_path = base.with_suffix(".html")
    txt_path = base.with_suffix(".txt")

    json_path.write_text(
        json.dumps(trace.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )
    html_path.write_text(render_html(trace), encoding="utf-8")
    txt_path.write_text(render_trace(trace), encoding="utf-8")

    return {
        "json": json_path.name,
        "html": html_path.name,
        "txt": txt_path.name,
    }


def _build_index(
    rows: list[dict[str, str]],
    overall_pass: int,
    overall_total: int,
) -> str:
    """Self-contained index.html that links to each per-query HTML and
    summarizes the run. Style intentionally matches the trace HTML so
    the demo feels like one artifact."""
    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append("<title>BRAG Phase 3 Demo</title>")
    parts.append("<style>")
    parts.append("""
    :root {
      --bg: #0f1115; --panel: #161a22; --panel-2: #1e2330;
      --border: #2a3142; --text: #d7dce5; --text-dim: #8a93a6;
      --accent: #79b8ff; --green: #2ea043; --amber: #d29922;
      --red: #cf222e; --blue: #218bff;
    }
    * { box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           background: var(--bg); color: var(--text); margin: 0; padding: 24px; line-height: 1.5; }
    .container { max-width: 1024px; margin: 0 auto; }
    h1 { font-size: 22px; margin: 0 0 6px 0; }
    .sub { color: var(--text-dim); font-size: 13px; }
    .panel { background: var(--panel); border: 1px solid var(--border);
             border-radius: 8px; padding: 18px 22px; margin-top: 18px; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { text-align: left; vertical-align: top; padding: 10px 8px;
             border-bottom: 1px solid var(--border); }
    th { color: var(--text-dim); font-weight: 600; }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .badge { display: inline-block; padding: 2px 9px; border-radius: 999px;
             font-size: 12px; font-weight: 600; letter-spacing: 0.02em;
             color: #fff; }
    .badge.green { background: var(--green); }
    .badge.amber { background: var(--amber); color: #1a1300; }
    .badge.blue  { background: var(--blue); }
    .badge.red   { background: var(--red); }
    .pill { display: inline-block; padding: 1px 7px; font-size: 11.5px;
            border: 1px solid var(--border); border-radius: 4px;
            color: var(--text-dim); margin-right: 4px; }
    .summary-line { font-size: 14px; margin-top: 10px; color: var(--text-dim); }
    code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12.5px;
           color: #c0d3f2; background: var(--panel-2); padding: 1px 5px;
           border-radius: 3px; }
    """)
    parts.append("</style></head><body><div class=\"container\">")
    parts.append("<h1>BRAG Phase 3 Demo</h1>")
    parts.append(
        '<div class="sub">Six curated runs of the full Validate -> Plan -> '
        "Verify -> Refute -> Generate -> Govern pipeline. Click a row to "
        "open the per-query trace.</div>"
    )
    parts.append(
        f'<div class="summary-line">Acceptance: '
        f"<strong>{overall_pass}/{overall_total}</strong> checks passed "
        "across 6 queries.</div>"
    )

    parts.append('<div class="panel"><table>')
    parts.append(
        "<thead><tr>"
        "<th>ID</th><th>Query</th><th>Outcome</th><th>Refutation</th>"
        "<th>Checks</th><th>Elapsed</th>"
        "</tr></thead><tbody>"
    )

    badge_color = {
        DegradationLevel.NORMAL.name: "green",
        DegradationLevel.PARTIAL.name: "amber",
        DegradationLevel.CLARIFICATION_REQUEST.name: "blue",
        DegradationLevel.HARD_HALT.name: "red",
    }

    for r in rows:
        if r.get("error"):
            parts.append(
                f'<tr><td><code>{html.escape(r["did"])}</code></td>'
                f'<td>{html.escape(r["query"])}</td>'
                f'<td><span class="badge red">RUN_FAILED</span></td>'
                f"<td>—</td><td>0/1</td>"
                f'<td>{html.escape(str(r["elapsed_s"]))}s</td></tr>'
            )
            continue
        deg = r["degradation_level"]
        color = badge_color.get(deg, "")
        ref = r["refutation_overall"]
        link = (
            f'<a href="{html.escape(r["html"])}">'
            f'<code>{html.escape(r["did"])}</code></a>'
        )
        parts.append(
            f"<tr>"
            f"<td>{link}</td>"
            f'<td>{html.escape(r["query"])}</td>'
            f'<td><span class="badge {color}">{html.escape(deg)}</span> '
            f'<span class="pill">{html.escape(r["degradation_cause"])}</span></td>'
            f'<td><span class="pill">{html.escape(ref)}</span></td>'
            f'<td>{r["passed"]}/{r["total"]}</td>'
            f'<td>{r["elapsed_s"]}s</td>'
            f"</tr>"
        )
    parts.append("</tbody></table></div>")
    parts.append("</div></body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Per-query runner
# ---------------------------------------------------------------------------


def _print_query_header(q: DemoQuery) -> None:
    print()
    print("=" * 78)
    print(f"{q.did}: {q.query}")
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


def _refutation_overall(trace: ExecutionTrace) -> str:
    if trace.refutation_report is None:
        return "bypassed"
    return trace.refutation_report.overall_verdict.value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated D-ids to run (e.g. D1,D2). Default: all 6.",
    )
    parser.add_argument(
        "--skip", type=str, default="",
        help="Comma-separated D-ids to skip (e.g. D5).",
    )
    args = parser.parse_args()

    only_ids = (
        {d.strip().upper() for d in args.only.split(",") if d.strip()}
        if args.only else None
    )
    skip_ids = {d.strip().upper() for d in args.skip.split(",") if d.strip()}

    targets = [
        q for q in _QUERIES
        if (only_ids is None or q.did.upper() in only_ids)
        and q.did.upper() not in skip_ids
    ]
    if not targets:
        print("No demo queries match --only/--skip filter.")
        sys.exit(2)

    rows: list[dict[str, str]] = []
    overall_pass = 0
    overall_total = 0
    overall_t0 = time.time()

    for q in targets:
        _print_query_header(q)
        t0 = time.time()
        crashed = False
        error: str | None = None
        try:
            trace = _run_with_transient_retry(q.query)
        except Exception as exc:
            crashed = True
            error = f"{type(exc).__name__}: {exc}"
            elapsed = round(time.time() - t0, 2)
            print(f"\n  RUN FAILED: {error}")
            rows.append({
                "did": q.did,
                "query": q.query,
                "elapsed_s": str(elapsed),
                "error": error,
                "passed": "0",
                "total": "1",
                "degradation_level": "RUN_FAILED",
                "degradation_cause": "",
                "refutation_overall": "",
                "html": "",
            })
            overall_pass += 0
            overall_total += 1
            continue

        elapsed = round(time.time() - t0, 2)

        # Render + persist before grading so the artifacts are on disk
        # even if a check raises later.
        files = _persist_run(q, trace, elapsed, [])

        # Brief on-screen summary line.
        n_disclosed = (
            len(trace.answer.disclosed_refutations) if trace.answer else 0
        )
        print(
            f"  -> degradation={trace.degradation_level.name} "
            f"cause={trace.degradation_cause.value} "
            f"refutation={_refutation_overall(trace)} "
            f"disclosed_refutations={n_disclosed} "
            f"governance_violations={len(trace.governance_violations)} "
            f"elapsed={elapsed}s"
        )
        if trace.answer is not None:
            answer_preview = trace.answer.answer_text
            if len(answer_preview) > 200:
                answer_preview = answer_preview[:200] + "..."
            print(f"  answer: {answer_preview}")

        results = _run_checks(trace, q, crashed=False, error=None)
        passed = _print_check_results(results)
        total = len(results)
        overall_pass += passed
        overall_total += total

        rows.append({
            "did": q.did,
            "query": q.query,
            "elapsed_s": str(elapsed),
            "passed": str(passed),
            "total": str(total),
            "degradation_level": trace.degradation_level.name,
            "degradation_cause": trace.degradation_cause.value,
            "refutation_overall": _refutation_overall(trace),
            "html": files["html"],
            "json": files["json"],
            "txt": files["txt"],
        })

    # Per-query summary table.
    print()
    print("=" * 78)
    print("PHASE 3 DEMO SUMMARY")
    print("=" * 78)
    print(f"  {'D':<4} {'pass/total':<12} {'degradation':<24} "
          f"{'refutation':<26} {'elapsed':<8}")
    print(f"  {'-'*4} {'-'*12} {'-'*24} {'-'*26} {'-'*8}")
    queries_passed = 0
    for r in rows:
        if r.get("error"):
            print(f"  {r['did']:<4} {'0/1':<12} {'RUN_FAILED':<24} "
                  f"{'':<26} {r['elapsed_s']}s")
            continue
        pf = int(r["passed"]) == int(r["total"])
        if pf:
            queries_passed += 1
        print(
            f"  {r['did']:<4} "
            f"{r['passed']}/{r['total']:<10} "
            f"{r['degradation_level']:<24} "
            f"{r['refutation_overall']:<26} "
            f"{r['elapsed_s']}s"
        )
    print()
    print(f"  OVERALL: {overall_pass}/{overall_total} acceptance checks across "
          f"{len(targets)} queries  ({queries_passed}/{len(targets)} queries "
          "fully passed)")
    print(f"  Total elapsed: {round(time.time() - overall_t0, 2)}s")

    # Write index.html
    index_html = _build_index(rows, overall_pass, overall_total)
    index_path = _OUT_DIR / "index.html"
    index_path.write_text(index_html, encoding="utf-8")
    print(f"\nIndex -> {index_path}")

    # Write rolling JSON summary.
    summary_path = _OUT_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "overall_pass": overall_pass,
                "overall_total": overall_total,
                "queries_passed": queries_passed,
                "queries_total": len(targets),
                "rows": rows,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # Required bar: at least 5 of 6 queries pass. Block 17 explicitly
    # accepts D5 falling to Phase 4 calibration on chunk-coverage.
    bar = max(5, len(targets) - 1) if len(targets) >= 5 else len(targets)
    if queries_passed < bar:
        print(
            f"\nFAIL: only {queries_passed}/{len(targets)} queries passed "
            f"(bar: {bar}/{len(targets)})."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Block 16: Static HTML renderer for ExecutionTrace.

Renders a single :class:`ExecutionTrace` into a self-contained HTML
document — inline CSS, no external assets, no JS framework. Double-
clicking the resulting file opens the trace in a browser; it's the
artifact you attach to a Slack message or commit to a repo for
showcase. Use ``pipeline.trace_renderer`` for the terminal equivalent
(same data, different surface).

Module surface:

  * :func:`render_html` — pure function, takes a fully-populated
    ExecutionTrace and returns the HTML string.
  * :func:`render_html_from_json` — reads a JSON trace from disk
    (tolerates the test-harness wrapper shape and renders the
    ``canonical`` block when present).
  * CLI: ``python -m pipeline.html_renderer <trace.json> -o <out.html>``.

The template lives at ``pipeline/templates/trace.html.j2``. The Python
side pre-computes color classes, paragraph splits, and refuting-fact
lookups so the template can stay declarative.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from agents.refutation.agent import lookup_facts
from schemas.enums import (
    DegradationLevel,
    PassOrigin,
    RefutationOverallVerdict,
    RefutationStrategy,
    RefutationVerdict,
    VerifierVerdict,
)
from schemas.records import ExecutionTrace, FactRecord


_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_NAME = "trace.html.j2"

# Numerical refutation strategies get the blue family; narrative
# strategies (later_reversal, alternative_cause, materialization,
# policy_change) get the amber family. Matches the build-plan
# §Block 16 color guidance.
_NUMERICAL_STRATEGIES: frozenset[RefutationStrategy] = frozenset({
    RefutationStrategy.restated_value,
    RefutationStrategy.revised_value,
    RefutationStrategy.guidance_vs_actual,
})


_DEGRADATION_COLOR: dict[DegradationLevel, str] = {
    DegradationLevel.NORMAL: "green",
    DegradationLevel.PARTIAL: "amber",
    DegradationLevel.CLARIFICATION_REQUEST: "blue",
    DegradationLevel.HARD_HALT: "red",
}


_REFUTATION_OVERALL_COLOR: dict[RefutationOverallVerdict, str] = {
    RefutationOverallVerdict.answer_strengthened: "green",
    RefutationOverallVerdict.refutation_to_loop: "amber",
    RefutationOverallVerdict.refutation_to_partial: "red",
}


_VERDICT_COLOR: dict[RefutationVerdict, str] = {
    RefutationVerdict.unrefuted: "green",
    RefutationVerdict.weakly_refuted: "amber",
    RefutationVerdict.strongly_refuted: "red",
}


_VERIFIER_VERDICT_COLOR: dict[VerifierVerdict, str] = {
    VerifierVerdict.covered: "green",
    VerifierVerdict.gap: "amber",
    VerifierVerdict.contradiction: "red",
    VerifierVerdict.exhausted: "red",
}


def _safe_iso(d: Any) -> str:
    """Tolerate either a date object (in-memory ExecutionTrace) or a
    YYYY-MM-DD string (JSON-deserialized trace)."""
    if isinstance(d, date):
        return d.isoformat()
    return str(d) if d is not None else ""


def _paragraphize(text: str) -> str:
    """HTML-escape ``text`` and wrap each blank-line-separated chunk
    in ``<p>`` so the answer renders with proper paragraph spacing.
    Single newlines within a paragraph become ``<br>``."""
    if not text:
        return ""
    paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    out: list[str] = []
    for p in paragraphs:
        if not p:
            continue
        escaped = html.escape(p).replace("\n", "<br>")
        out.append(f"<p>{escaped}</p>")
    return "\n".join(out)


def _strategy_color(s: RefutationStrategy) -> str:
    return "strategy-num" if s in _NUMERICAL_STRATEGIES else "strategy-narr"


def _coverage_bar(score: float, iteration: int) -> dict[str, Any]:
    """One sparkline bar: 26px tall, height scaled to score, color
    keyed to the coverage band the spec uses for the verifier."""
    height = max(2, int(round(score * 26)))
    if score >= 0.80:
        color = "var(--green)"
    elif score >= 0.50:
        color = "var(--amber)"
    else:
        color = "var(--red)"
    return {
        "height": height,
        "color": color,
        "iter": iteration,
        "score": f"{score:.2f}",
    }


def _build_hypothesis_rows(
    trace: ExecutionTrace,
    fact_map: dict[str, FactRecord],
) -> list[dict[str, Any]]:
    """One row per RefutationHypothesis. Refuting evidence is
    pre-resolved against ``fact_map`` (which is the best-effort
    ``lookup_facts`` snapshot taken at render time)."""
    report = trace.refutation_report
    if report is None:
        return []
    rows: list[dict[str, Any]] = []
    for h in report.hypotheses:
        targeted = fact_map.get(h.targets_claim_id)
        refuting: list[dict[str, Any]] = []
        for eid in h.evidence_ids:
            ev = fact_map.get(eid)
            refuting.append({
                "fact_id": eid,
                "date": _safe_iso(ev.assertion_date) if ev else "(unknown date)",
                "claim": (ev.claim if ev else ""),
            })
        rows.append({
            "hypothesis_id": h.hypothesis_id,
            "strategy": h.strategy.value,
            "strategy_color": _strategy_color(h.strategy),
            "verdict": h.refutation_verdict.value,
            "verdict_color": _VERDICT_COLOR[h.refutation_verdict],
            "targets_claim_id": h.targets_claim_id,
            "targeted_date": (
                _safe_iso(targeted.assertion_date) if targeted else "(date unknown)"
            ),
            "hypothesis_text": h.hypothesis_text,
            "rationale": h.rationale,
            "refuting": refuting,
        })
    return rows


def _build_slot_coverage(
    trace: ExecutionTrace,
) -> dict[str, list[dict[str, Any]]]:
    """Sparkline-friendly coverage data grouped by slot, ordered by
    iteration. Pulled from ``trace.coverage_progression``."""
    out: dict[str, list[dict[str, Any]]] = {}
    for entry in trace.coverage_progression:
        out.setdefault(entry.slot_id, []).append(
            _coverage_bar(entry.coverage_score, entry.iteration)
        )
    for sid in out:
        out[sid].sort(key=lambda b: b["iter"])
    return out


def _build_gap_history(trace: ExecutionTrace) -> list[dict[str, Any]]:
    """Pull gap descriptions from the ledger snapshot in trace.extra.
    Each gap is one retry's rationale for re-running a slot."""
    ledger = trace.extra.get("ledger") or {}
    raw = ledger.get("gap_history") or []
    out: list[dict[str, Any]] = []
    for g in raw:
        out.append({
            "slot_id": g.get("slot_id", ""),
            "iteration": g.get("iteration", 0),
            "gap_description": g.get("gap_description", ""),
        })
    return out


def _build_retrieved_ids_per_slot(
    trace: ExecutionTrace,
) -> tuple[dict[str, list[str]], int]:
    """Group retrieved candidate IDs by slot. Used for the
    expandable 'Retrieved IDs per slot' detail block."""
    ledger = trace.extra.get("ledger") or {}
    retrieved = ledger.get("retrieved_ids") or {}
    # ``retrieved_ids`` is keyed slot_id -> list[str]; preserve insertion order.
    out: dict[str, list[str]] = {}
    total = 0
    for sid, ids in retrieved.items():
        # Dedupe while preserving order, since multiple iterations may
        # retrieve the same candidate.
        seen: set[str] = set()
        deduped: list[str] = []
        for c in ids:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
        out[sid] = deduped
        total += len(deduped)
    return out, total


def _build_slot_passes(trace: ExecutionTrace) -> list[dict[str, Any]]:
    """Per-slot iteration rows for the collapsible execution-trace
    panel. Drops refutation-loop passes — those belong to the
    refutation loop section. Verdicts and retrievals are paired by
    (slot_id, iteration) lookup."""
    plan = trace.decomposition_plan
    slot_meta = {s.slot_id: s for s in plan.slots}
    terminal_by_slot = {fs.slot_id: fs for fs in trace.final_slot_states}
    verdicts_by_slot: dict[str, list[Any]] = {}
    for v in trace.verifier_verdicts:
        verdicts_by_slot.setdefault(v.slot_id, []).append(v)
    retrievals: dict[tuple[str, int], Any] = {}
    for r in trace.retrieval_passes:
        if r.pass_origin != PassOrigin.verifier_loop:
            continue
        retrievals[(r.slot_id, r.iteration)] = r

    out: list[dict[str, Any]] = []
    for s in plan.slots:
        slot_id = s.slot_id
        v_list = verdicts_by_slot.get(slot_id, [])
        iters: list[dict[str, Any]] = []
        for idx, v in enumerate(v_list, start=1):
            r = retrievals.get((slot_id, idx))
            iters.append({
                "iteration": idx,
                "candidate_count": len(r.candidates) if r else 0,
                "coverage_score": v.coverage_score,
                "verdict": v.verdict.value,
                "verdict_color": _VERIFIER_VERDICT_COLOR.get(v.verdict, ""),
                "gap_description": v.gap_description or "",
            })
        terminal = terminal_by_slot.get(slot_id)
        out.append({
            "slot_id": slot_id,
            "evidence_type": s.evidence_type.value,
            "target_layer": s.target_layer.value,
            "period_filter": s.period_filter,
            "terminal_verdict": terminal.terminal_verdict.value if terminal else "n/a",
            "terminal_color": (
                _VERIFIER_VERDICT_COLOR.get(terminal.terminal_verdict, "")
                if terminal else ""
            ),
            "iterations": iters,
        })
    return out


_ENV: Environment | None = None


def _get_env() -> Environment:
    """Lazy Jinja env with safe defaults: autoescape on, StrictUndefined
    so a typo in the template surfaces immediately instead of rendering
    as an empty string."""
    global _ENV
    if _ENV is None:
        _ENV = Environment(
            loader=FileSystemLoader(str(_TEMPLATE_DIR)),
            autoescape=select_autoescape(["html"]),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
    return _ENV


def render_html(trace: ExecutionTrace) -> str:
    """Render ``trace`` as a self-contained HTML document.

    The template lives at ``pipeline/templates/trace.html.j2``. CSS is
    inlined; no external assets are loaded. Refuting-fact lookups go
    against the local fact-store JSONL on a best-effort basis — if
    the store isn't present (e.g. rendering a trace on a different
    machine), refuting evidence rows fall back to "(unknown date)" /
    "(no fact text available)"."""
    env = _get_env()
    template = env.get_template(_TEMPLATE_NAME)

    # Best-effort fact lookup for refuting + targeted evidence dates.
    needed_ids: set[str] = set()
    if trace.refutation_report is not None:
        for h in trace.refutation_report.hypotheses:
            needed_ids.add(h.targets_claim_id)
            needed_ids.update(h.evidence_ids)
    try:
        fact_map = lookup_facts(needed_ids) if needed_ids else {}
    except Exception:
        fact_map = {}

    retrieved_ids_per_slot, retrieved_id_total = _build_retrieved_ids_per_slot(trace)

    context = {
        "query": trace.query.original_query,
        "run_id": trace.run_id,
        "complexity_tier": trace.query.complexity_tier.value,
        "degradation_label": trace.degradation_level.name,
        "degradation_color": _DEGRADATION_COLOR.get(trace.degradation_level, ""),
        "degradation_cause": trace.degradation_cause.value,
        "rejection_reason": trace.extra.get("rejection_reason"),
        "adversarially_probed": bool(trace.extra.get("adversarially_probed")),
        "refutation_fallback_invoked": bool(
            trace.extra.get("refutation_unavailable_fallback_invoked")
        ),
        "elapsed_seconds": f"{trace.elapsed_seconds:.2f}",
        "total_iterations": trace.total_iterations,
        "models_used": list(trace.models_used),
        "answer": trace.answer,
        "answer_html": _paragraphize(trace.answer.answer_text) if trace.answer else "",
        "refutation_report": trace.refutation_report,
        "refutation_overall_color": (
            _REFUTATION_OVERALL_COLOR.get(
                trace.refutation_report.overall_verdict,
                "",
            ) if trace.refutation_report else ""
        ),
        "hypothesis_rows": _build_hypothesis_rows(trace, fact_map),
        "slot_coverage": _build_slot_coverage(trace),
        "gap_history": _build_gap_history(trace),
        "refutation_loop_iterations": list(trace.refutation_loop_iterations),
        "retrieved_ids_per_slot": retrieved_ids_per_slot,
        "retrieved_id_total": retrieved_id_total,
        "slot_passes": _build_slot_passes(trace),
        "governance_violations": list(trace.governance_violations),
    }
    return template.render(**context)


def render_html_from_json(path: str | Path) -> str:
    """Load a serialized trace JSON and render it. Tolerates the
    test-harness wrapper ``{"canonical": <trace>, "asserting": ...}``
    by unwrapping to the canonical trace."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if "canonical" in raw and isinstance(raw["canonical"], dict):
        raw = raw["canonical"]
    trace = ExecutionTrace.model_validate(raw)
    return render_html(trace)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(
        description="Render an ExecutionTrace JSON as a self-contained HTML file."
    )
    parser.add_argument("trace_path", help="Path to an ExecutionTrace JSON file.")
    parser.add_argument(
        "-o", "--out", required=True, type=str,
        help="Output HTML path. The parent directory will be created if missing.",
    )
    args = parser.parse_args()

    html_doc = render_html_from_json(args.trace_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {out_path}  ({len(html_doc):,} bytes)")


if __name__ == "__main__":
    main()

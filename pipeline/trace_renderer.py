"""Block 15: Live Execution Trace renderer (text).

Renders an :class:`ExecutionTrace` as terminal-friendly markdown so a
reviewer can follow which slots ran, what the refutation pass tested,
why the answer disclosed the contradiction, and what governance
verified. The spec calls this "Live"; v1 renders post-hoc — same
artifact shape, different timing. The Block 16 HTML renderer fulfills
the same contract for browsers.

The renderer is pure (no I/O, no LLM calls). Refuting-evidence
assertion_dates are looked up from the fact-store JSONL on a
best-effort basis via :func:`agents.refutation.agent.lookup_facts` —
missing facts (e.g. when rendering a trace on a machine without the
fact store) degrade to the raw fact_id.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from agents.refutation.agent import lookup_facts
from schemas.enums import (
    DegradationLevel,
    PassOrigin,
    RefutationVerdict,
    VerifierVerdict,
)
from schemas.records import (
    ExecutionTrace,
    RetrievalRecord,
    VerifierOutput,
)


# A divider that survives both terminal pagers and Markdown viewers.
_DIVIDER = "─" * 76


def _hdr(text: str) -> str:
    """One-line section header underlined with the divider."""
    return f"## {text}\n{_DIVIDER}"


def _safe_iso(d: Any) -> str:
    """Tolerate either a ``date`` object (in-memory ExecutionTrace) or
    a YYYY-MM-DD string (deserialized JSON trace)."""
    if isinstance(d, date):
        return d.isoformat()
    return str(d)


def _badge(text: str) -> str:
    """Bracket-wrapped badge token, terminal-readable."""
    return f"[{text}]"


def _render_header(trace: ExecutionTrace) -> str:
    lines = ["# BRAG Execution Trace"]
    lines.append("")
    lines.append(f"**Query:** {trace.query.original_query}")
    lines.append(f"**Run ID:** `{trace.run_id}`")
    lines.append(f"**Complexity tier:** {trace.query.complexity_tier.value}")
    lines.append(
        f"**Final degradation level:** {trace.degradation_level.name} "
        f"(cause: {trace.degradation_cause.value})"
    )
    badges: list[str] = []
    if trace.extra.get("adversarially_probed"):
        badges.append(_badge("ADVERSARIALLY PROBED"))
    if trace.extra.get("refutation_unavailable_fallback_invoked"):
        badges.append(_badge("REFUTATION FALLBACK"))
    if badges:
        lines.append("**Badges:** " + " ".join(badges))
    return "\n".join(lines)


def _render_validation(trace: ExecutionTrace) -> str:
    lines = [_hdr("Input Validation")]
    status = trace.query.validation_status
    lines.append(f"- Status: `{status}`")
    rejection = trace.extra.get("rejection_reason")
    if rejection:
        lines.append(f"- Rejection reason: {rejection}")
    warnings = trace.extra.get("warnings") or []
    if warnings:
        lines.append(f"- Warnings: {warnings}")
    else:
        lines.append("- Warnings: none")
    return "\n".join(lines)


def _render_plan(trace: ExecutionTrace) -> str:
    plan = trace.decomposition_plan
    lines = [_hdr("Decomposition Plan")]
    lines.append(f"- Synthesis strategy: `{plan.synthesis_strategy.value}`")
    lines.append(f"- Slots ({len(plan.slots)}):")
    for s in plan.slots:
        period = f"pf={s.period_filter}" if s.period_filter else "pf=none"
        lines.append(
            f"  - **{s.slot_id}** — `{s.evidence_type.value}` / "
            f"`{s.target_layer.value}` / {period} / threshold "
            f"{s.coverage_threshold:.2f}"
        )
        lines.append(f"    sub_q: {s.sub_question}")
        if s.key_terms:
            lines.append(f"    key_terms: {s.key_terms}")
    return "\n".join(lines)


def _group_by_slot(
    verdicts: list[VerifierOutput],
    retrievals: list[RetrievalRecord],
    pass_origin: PassOrigin,
) -> dict[str, list[tuple[int, RetrievalRecord | None, VerifierOutput | None]]]:
    """Pair verdicts and retrievals by (slot_id, iteration), filtered
    to ``pass_origin``. The retrieval drives iteration ordering; the
    verdict gets attached by index. Returns a dict slot_id ->
    [(iteration, retrieval, verdict)]."""
    # Index retrievals
    out: dict[str, list[tuple[int, RetrievalRecord | None, VerifierOutput | None]]] = {}
    by_slot_iter: dict[tuple[str, int], RetrievalRecord] = {}
    for r in retrievals:
        if r.pass_origin != pass_origin:
            continue
        by_slot_iter[(r.slot_id, r.iteration)] = r

    # Verdicts are per-iteration but lack iteration numbers — they're
    # appended in order per slot. We zip them onto the retrievals.
    verdicts_by_slot: dict[str, list[VerifierOutput]] = {}
    for v in verdicts:
        verdicts_by_slot.setdefault(v.slot_id, []).append(v)

    # Group iterations by slot in order.
    iters_by_slot: dict[str, list[int]] = {}
    for (sid, it) in by_slot_iter.keys():
        iters_by_slot.setdefault(sid, []).append(it)
    for sid in iters_by_slot:
        iters_by_slot[sid].sort()

    for sid, iters in iters_by_slot.items():
        v_list = verdicts_by_slot.get(sid, [])
        rows: list[tuple[int, RetrievalRecord | None, VerifierOutput | None]] = []
        for idx, it in enumerate(iters):
            r = by_slot_iter.get((sid, it))
            v = v_list[idx] if idx < len(v_list) else None
            rows.append((it, r, v))
        out[sid] = rows
    return out


def _render_verifier_loop(trace: ExecutionTrace) -> str:
    lines = [_hdr("Verifier Loop")]
    groups = _group_by_slot(
        trace.verifier_verdicts,
        trace.retrieval_passes,
        PassOrigin.verifier_loop,
    )
    if not groups:
        lines.append("- (no verifier-loop activity)")
        return "\n".join(lines)
    for sid, rows in groups.items():
        terminal = next(
            (fs for fs in trace.final_slot_states if fs.slot_id == sid),
            None,
        )
        term_str = (
            f" — terminal verdict: **{terminal.terminal_verdict.value}** "
            f"(coverage {terminal.final_coverage:.2f})"
            if terminal else ""
        )
        lines.append(f"- **{sid}**{term_str}")
        for it, r, v in rows:
            n_cands = len(r.candidates) if r else 0
            cov = f"{v.coverage_score:.2f}" if v else "—"
            verdict = v.verdict.value if v else "—"
            lines.append(
                f"  - iter {it}: retrieved={n_cands} "
                f"coverage={cov} verdict={verdict}"
            )
            if v is not None and v.verdict in (
                VerifierVerdict.gap, VerifierVerdict.exhausted,
            ) and v.gap_description:
                lines.append(f"    gap: {v.gap_description}")
            if v is not None and v.contradiction_details:
                for cd in v.contradiction_details:
                    lines.append(
                        f"    contradiction: {cd.description} "
                        f"(conflicting={cd.conflicting_ids})"
                    )
    return "\n".join(lines)


def _render_refutation_stage(trace: ExecutionTrace) -> str:
    report = trace.refutation_report
    lines = [_hdr("Refutation Stage")]
    if report is None:
        reason = trace.extra.get("refutation_bypass_reason") or "(not run)"
        lines.append(f"- Bypassed: {reason}")
        return "\n".join(lines)
    lines.append(f"- Model used: `{report.model_used}`")
    lines.append(f"- Overall verdict: **{report.overall_verdict.value}**")
    lines.append(
        f"- Loop re-entry triggered: {report.triggered_loop_reentry}"
        + (
            f" (iteration {report.loop_reentry_iteration})"
            if report.triggered_loop_reentry
            and report.loop_reentry_iteration is not None
            else ""
        )
    )
    lines.append(f"- Hypotheses ({len(report.hypotheses)}):")
    # Best-effort fact lookup so we can show refuting-evidence
    # assertion_dates per spec.
    needed_ids: set[str] = set()
    for h in report.hypotheses:
        needed_ids.add(h.targets_claim_id)
        needed_ids.update(h.evidence_ids)
    try:
        fact_map = lookup_facts(needed_ids)
    except Exception:
        fact_map = {}
    for h in report.hypotheses:
        targeted = fact_map.get(h.targets_claim_id)
        target_date = (
            f" (targeted_date={_safe_iso(targeted.assertion_date)})"
            if targeted else ""
        )
        verdict_badge = _badge(h.refutation_verdict.value.upper())
        lines.append(
            f"  - **{h.hypothesis_id}** {verdict_badge} "
            f"strategy=`{h.strategy.value}` "
            f"targets=`{h.targets_claim_id}`{target_date}"
        )
        lines.append(f"    hypothesis: {h.hypothesis_text}")
        if h.refutation_verdict != RefutationVerdict.unrefuted and h.evidence_ids:
            for eid in h.evidence_ids:
                ev = fact_map.get(eid)
                date_str = (
                    f" (assertion_date={_safe_iso(ev.assertion_date)})"
                    if ev else ""
                )
                lines.append(f"    refuting: `{eid}`{date_str}")
    return "\n".join(lines)


def _render_refutation_loop(trace: ExecutionTrace) -> str:
    if not trace.refutation_loop_iterations:
        return ""
    lines = [_hdr("Refutation Loop")]
    groups = _group_by_slot(
        trace.verifier_verdicts,
        trace.retrieval_passes,
        PassOrigin.refutation_loop,
    )
    for rec in trace.refutation_loop_iterations:
        resolved = rec.coverage_after >= 0.80
        status = (
            _badge("RESOLVED") if resolved
            else _badge("UNRESOLVED")
        )
        lines.append(
            f"- iter {rec.iteration} {status} "
            f"triggering=`{rec.triggering_hypothesis_id}` "
            f"targets=`{rec.targets_claim_id}` "
            f"coverage_after={rec.coverage_after:.2f}"
        )
    if groups:
        lines.append("- Per-slot retrieval/verifier passes:")
        for sid, rows in groups.items():
            lines.append(f"  - **{sid}**")
            for it, r, v in rows:
                n_cands = len(r.candidates) if r else 0
                cov = f"{v.coverage_score:.2f}" if v else "—"
                verdict = v.verdict.value if v else "—"
                lines.append(
                    f"    - iter {it}: retrieved={n_cands} "
                    f"coverage={cov} verdict={verdict}"
                )
    return "\n".join(lines)


def _render_generator(trace: ExecutionTrace) -> str:
    answer = trace.answer
    lines = [_hdr("Generator Output")]
    if answer is None:
        lines.append("- (no answer produced)")
        return "\n".join(lines)
    lines.append(f"- Claims: {len(answer.claims)}")
    lines.append(f"- Disclosed gaps: {len(answer.disclosed_gaps)}")
    lines.append(f"- Disclosed contradictions: {len(answer.disclosed_contradictions)}")
    lines.append(f"- Disclosed refutations: {len(answer.disclosed_refutations)}")
    lines.append(f"- Adversarially probed: {answer.adversarially_probed}")
    lines.append("")
    lines.append("**Answer text:**")
    lines.append("")
    lines.append(answer.answer_text)
    if answer.claims:
        lines.append("")
        lines.append("**Claims with citations:**")
        for i, c in enumerate(answer.claims, start=1):
            ids = ", ".join(c.source_ids) if c.source_ids else "(no sources)"
            lines.append(
                f"  {i}. [{c.claim_type.value}] {c.claim_text}"
            )
            lines.append(f"     sources: {ids}")
    if answer.disclosed_refutations:
        lines.append("")
        lines.append("**Disclosed refutations:**")
        for d in answer.disclosed_refutations:
            ev = ", ".join(d.refuting_evidence_ids) or "(no refuting ids)"
            lines.append(
                f"  - targets=`{d.targets_claim_id}` "
                f"verdict={d.refutation_verdict.value} "
                f"strategy={d.strategy.value} refuting=[{ev}]"
            )
    if answer.disclosed_gaps:
        lines.append("")
        lines.append("**Disclosed gaps:**")
        for g in answer.disclosed_gaps:
            lines.append(f"  - slot `{g.slot_id}`: {g.gap_description}")
    if answer.disclosed_contradictions:
        lines.append("")
        lines.append("**Disclosed contradictions:**")
        for c in answer.disclosed_contradictions:
            lines.append(
                f"  - {c.description}  conflicting=[{', '.join(c.conflicting_ids)}]"
            )
    return "\n".join(lines)


def _render_governance(trace: ExecutionTrace) -> str:
    lines = [_hdr("Output Governance")]
    violations = trace.governance_violations
    if not violations:
        lines.append(f"- {_badge('PASS')} no governance violations")
        return "\n".join(lines)
    lines.append(
        f"- {_badge('FAIL')} {len(violations)} violation(s):"
    )
    for v in violations:
        extras = []
        if v.claim_index is not None:
            extras.append(f"claim_index={v.claim_index}")
        if v.hypothesis_id is not None:
            extras.append(f"hypothesis_id={v.hypothesis_id}")
        if v.expected is not None:
            extras.append(f"expected={v.expected!r}")
        if v.actual is not None:
            extras.append(f"actual={v.actual!r}")
        extra_str = f"  ({', '.join(extras)})" if extras else ""
        lines.append(f"  - [{v.severity.value}] {v.message}{extra_str}")
    return "\n".join(lines)


def _render_footer(trace: ExecutionTrace) -> str:
    lines = [_hdr("Run Summary")]
    lines.append(f"- Elapsed: {trace.elapsed_seconds:.2f}s")
    lines.append(f"- Total iterations (verifier + refutation loop): {trace.total_iterations}")
    lines.append(f"- Models used: {trace.models_used}")
    if trace.total_tokens_consumed:
        lines.append(f"- Total tokens: {trace.total_tokens_consumed}")
    return "\n".join(lines)


def render_trace(trace: ExecutionTrace) -> str:
    """Render ``trace`` as terminal-friendly markdown.

    Section order (per build plan §Block 15):
      1. Header           — query / run_id / tier / degradation / badges
      2. Input Validation — status + warnings
      3. Decomposition    — synthesis strategy + per-slot summary
      4. Verifier Loop    — per-slot iterations + verdicts
      5. Refutation Stage — hypotheses + verdicts (conditional)
      6. Refutation Loop  — per-iteration coverage + verdicts (conditional)
      7. Generator Output — answer_text + claims + disclosures
      8. Output Governance — pass/fail + violations
      9. Run Summary       — elapsed + models + iterations

    Conditional sections (Refutation Stage / Loop) are emitted only
    when the trace recorded activity. The renderer never reaches out
    to the network; refuting-evidence assertion_dates are best-effort
    via the local fact-store JSONL index."""
    parts: list[str] = [_render_header(trace)]
    parts.append(_render_validation(trace))
    parts.append(_render_plan(trace))
    parts.append(_render_verifier_loop(trace))
    parts.append(_render_refutation_stage(trace))
    loop_section = _render_refutation_loop(trace)
    if loop_section:
        parts.append(loop_section)
    parts.append(_render_generator(trace))
    parts.append(_render_governance(trace))
    parts.append(_render_footer(trace))
    return "\n\n".join(parts) + "\n"


def render_trace_from_json(path: str) -> str:
    """Load a serialized ExecutionTrace JSON and render it. Used by
    the ``--trace`` CLI flag on the Block 11/12/14 test scripts so an
    existing log can be re-rendered without re-running."""
    import json
    from pathlib import Path

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    # Tolerate the test-harness wrappers ({canonical: ..., asserting: ...}).
    if "canonical" in raw and isinstance(raw["canonical"], dict):
        raw = raw["canonical"]
    trace = ExecutionTrace.model_validate(raw)
    return render_trace(trace)


def main() -> None:
    """CLI entry point: ``python -m pipeline.trace_renderer <trace.json>``."""
    import argparse
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(
        description="Render an ExecutionTrace JSON as markdown."
    )
    parser.add_argument("trace_path", help="Path to a trace JSON file.")
    args = parser.parse_args()
    print(render_trace_from_json(args.trace_path))


if __name__ == "__main__":
    main()

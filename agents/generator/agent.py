"""Block 14: Generator Agent wrapper (Stage 7).

Thin orchestrator-facing shell around ``generate_answer`` from
``prompt.py``. Responsibilities:

  - Bypass on Clarification Request / Hard Halt — emit a canned
    AnswerSchema without an LLM call.
  - For Normal / Partial runs, hand the verified set + refutation
    report + degradation level + adversarially_probed flag to the
    Generator prompt and return the AnswerSchema it produced.
  - Optionally accept a ``prior_governance_feedback`` string the
    orchestrator can use to retry after a Governance failure.

The wrapper does not implement Governance — that's Stage 8 in
``pipeline/governance.py``. It does not decide degradation either —
the orchestrator does. This module just gates the LLM call.
"""
from __future__ import annotations

from agents.generator.prompt import (
    GENERATOR_MODEL,
    GENERATOR_MODEL_FALLBACK,
    generate_answer,
)
from agents.llm_client import LLMResponse, OpenRouterClient
from schemas.enums import DegradationLevel, RefutationVerdict
from schemas.records import (
    AnswerSchema,
    DisclosedContradiction,
    DisclosedGap,
    DisclosedRefutation,
    FactRecord,
    RefutationReport,
)


CLARIFICATION_REQUEST_TEXT = (
    "I can't answer this question from the Netflix corpus available "
    "to me — no verified evidence was found across the planned "
    "retrieval slots. Narrow the question to a specific metric, "
    "period, or filing so I can ground the answer in primary sources."
)

HARD_HALT_TEXT = (
    "The run halted before an answer could be generated. The cause "
    "is recorded on the trace's degradation_cause field."
)


def _build_disclosed_refutations_from_report(
    report: RefutationReport | None,
) -> list[DisclosedRefutation]:
    """Mirror the prompt-module helper so the bypass paths (CR / Hard
    Halt) still produce the correct deterministic disclosure list
    without invoking the LLM."""
    if report is None:
        return []
    out: list[DisclosedRefutation] = []
    for h in report.hypotheses:
        if h.refutation_verdict == RefutationVerdict.unrefuted:
            continue
        out.append(
            DisclosedRefutation(
                targets_claim_id=h.targets_claim_id,
                refuting_evidence_ids=list(h.evidence_ids),
                refutation_verdict=h.refutation_verdict,
                strategy=h.strategy,
            )
        )
    return out


def _stub_answer(
    *,
    text: str,
    degradation_level: DegradationLevel,
    adversarially_probed: bool,
    disclosed_gaps: list[DisclosedGap],
    disclosed_contradictions: list[DisclosedContradiction],
    refutation_report: RefutationReport | None,
) -> AnswerSchema:
    return AnswerSchema(
        answer_text=text,
        claims=[],
        disclosed_gaps=disclosed_gaps,
        disclosed_contradictions=disclosed_contradictions,
        disclosed_refutations=_build_disclosed_refutations_from_report(refutation_report),
        adversarially_probed=adversarially_probed,
        degradation_level=degradation_level,
    )


def run_generator(
    *,
    original_query: str,
    verified_facts: list[FactRecord],
    refutation_report: RefutationReport | None,
    degradation_level: DegradationLevel,
    adversarially_probed: bool,
    disclosed_gaps: list[DisclosedGap] | None = None,
    disclosed_contradictions: list[DisclosedContradiction] | None = None,
    client: OpenRouterClient | None = None,
    prior_governance_feedback: str | None = None,
) -> tuple[AnswerSchema, LLMResponse | None]:
    """Stage 7 entry point used by the orchestrator.

    Returns ``(answer, llm_response)``. ``llm_response`` is None on
    bypass paths (CR / Hard Halt) where no LLM call was made.

    Raises ``LLMError`` if the Normal/Partial path's two-attempt
    Generator call fails. The orchestrator catches this and escalates
    to Hard Halt with cause=generator_unavailable."""
    gaps = list(disclosed_gaps or [])
    contradictions = list(disclosed_contradictions or [])

    if degradation_level == DegradationLevel.HARD_HALT:
        return (
            _stub_answer(
                text=HARD_HALT_TEXT,
                degradation_level=degradation_level,
                adversarially_probed=adversarially_probed,
                disclosed_gaps=gaps,
                disclosed_contradictions=contradictions,
                refutation_report=refutation_report,
            ),
            None,
        )

    if degradation_level == DegradationLevel.CLARIFICATION_REQUEST:
        return (
            _stub_answer(
                text=CLARIFICATION_REQUEST_TEXT,
                degradation_level=degradation_level,
                adversarially_probed=adversarially_probed,
                disclosed_gaps=gaps,
                disclosed_contradictions=contradictions,
                refutation_report=refutation_report,
            ),
            None,
        )

    # Normal / Partial: real LLM call.
    answer, resp = generate_answer(
        original_query=original_query,
        verified_facts=verified_facts,
        refutation_report=refutation_report,
        degradation_level=degradation_level,
        adversarially_probed=adversarially_probed,
        disclosed_gaps=gaps,
        disclosed_contradictions=contradictions,
        client=client,
        prior_governance_feedback=prior_governance_feedback,
    )
    return answer, resp


__all__ = [
    "GENERATOR_MODEL",
    "GENERATOR_MODEL_FALLBACK",
    "CLARIFICATION_REQUEST_TEXT",
    "HARD_HALT_TEXT",
    "run_generator",
]

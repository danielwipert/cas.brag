"""Pydantic record schemas for BRAG (spec v3 §2.5, §3.3–§3.8, §7.2)."""
from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from schemas.enums import (
    CandidateSource,
    ClaimType,
    ComplexityTier,
    DegradationCause,
    DegradationLevel,
    EvidenceType,
    FactType,
    GovernanceSeverity,
    PassOrigin,
    RefutationOverallVerdict,
    RefutationStrategy,
    RefutationVerdict,
    SynthesisStrategy,
    TargetLayer,
    VerifierVerdict,
)
from schemas.period import is_valid_period


_FROZEN = ConfigDict(extra="forbid", frozen=False)


def _validate_period(v: str | None) -> str | None:
    if v is None:
        return None
    if not is_valid_period(v):
        raise ValueError(
            f"period {v!r} does not match canonical format "
            "(YYYYQN, FYYYYY, FYYYYY-guidance, or YYYY-MM-DD)"
        )
    return v


# ---------------------------------------------------------------------------
# Layer A / Layer B records (spec §2.5)
# ---------------------------------------------------------------------------


class FactRecord(BaseModel):
    """One atomic claim. XBRL-sourced facts have confidence=1.00 and a
    concept_tag; LLM-extracted prose facts cover the other six fact types."""

    model_config = _FROZEN

    fact_id: str
    claim: str
    asserter: str
    source_document: str
    source_section: str
    verbatim_anchor: str
    fact_type: FactType
    period: str | None = None
    value: float | None = None
    unit: str | None = None
    concept_tag: str | None = None
    assertion_date: date
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("period")
    @classmethod
    def _check_period(cls, v: str | None) -> str | None:
        return _validate_period(v)


class ChunkRecord(BaseModel):
    model_config = _FROZEN

    chunk_id: str
    text: str
    source_document: str
    section: str
    position_index: int = Field(ge=0)
    word_count: int = Field(ge=0)


# ---------------------------------------------------------------------------
# Decomposition Plan (spec §3.3)
# ---------------------------------------------------------------------------


class EvidenceSlot(BaseModel):
    model_config = _FROZEN

    slot_id: str
    sub_question: str
    evidence_type: EvidenceType
    target_layer: TargetLayer
    period_filter: str | None = None
    key_terms: list[str] = Field(default_factory=list)
    coverage_threshold: float = Field(default=0.80, ge=0.0, le=1.0)

    @field_validator("period_filter")
    @classmethod
    def _check_period_filter(cls, v: str | None) -> str | None:
        return _validate_period(v)


class DecompositionPlan(BaseModel):
    model_config = _FROZEN

    query_id: str
    original_query: str
    complexity_tier: ComplexityTier
    synthesis_strategy: SynthesisStrategy
    slots: list[EvidenceSlot] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Retrieval (spec §3.4)
# ---------------------------------------------------------------------------


class RetrievalCandidate(BaseModel):
    model_config = _FROZEN

    candidate_id: str
    source: CandidateSource
    rrf_score: float
    vector_score: float | None = None
    bm25_score: float | None = None


class RetrievalRecord(BaseModel):
    model_config = _FROZEN

    retrieval_id: str
    slot_id: str
    iteration: int = Field(ge=1)
    pass_origin: PassOrigin
    vector_query: str
    bm25_terms: list[str] = Field(default_factory=list)
    period_filter: str | None = None
    candidates: list[RetrievalCandidate] = Field(default_factory=list)
    memory_exclusions: list[str] = Field(default_factory=list)

    @field_validator("period_filter")
    @classmethod
    def _check_period_filter(cls, v: str | None) -> str | None:
        return _validate_period(v)


# ---------------------------------------------------------------------------
# Verifier (spec §3.5)
# ---------------------------------------------------------------------------


class ContradictionDetail(BaseModel):
    model_config = _FROZEN

    description: str
    conflicting_ids: list[str] = Field(default_factory=list)


class VerifierOutput(BaseModel):
    model_config = _FROZEN

    slot_id: str
    coverage_score: float = Field(ge=0.0, le=1.0)
    verdict: VerifierVerdict
    gap_description: str | None = None
    contradiction_details: list[ContradictionDetail] = Field(default_factory=list)
    supported_candidates: list[str] = Field(default_factory=list)
    rejected_candidates: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Refutation (spec §3.6)
# ---------------------------------------------------------------------------


class RefutationHypothesis(BaseModel):
    model_config = _FROZEN

    hypothesis_id: str
    targets_claim_id: str
    hypothesis_text: str
    rationale: str
    strategy: RefutationStrategy
    retrieval_record_id: str | None = None
    refutation_verdict: RefutationVerdict = RefutationVerdict.unrefuted
    evidence_ids: list[str] = Field(default_factory=list)


class RefutationReport(BaseModel):
    model_config = _FROZEN

    run_id: str
    model_used: str
    hypotheses: list[RefutationHypothesis] = Field(default_factory=list)
    overall_verdict: RefutationOverallVerdict
    triggered_loop_reentry: bool = False
    loop_reentry_iteration: int | None = None


# ---------------------------------------------------------------------------
# Memory Ledger (spec §3.7)
# ---------------------------------------------------------------------------


class CoverageHistoryEntry(BaseModel):
    model_config = _FROZEN

    slot_id: str
    iteration: int
    coverage_score: float = Field(ge=0.0, le=1.0)


class GapHistoryEntry(BaseModel):
    model_config = _FROZEN

    slot_id: str
    iteration: int
    gap_description: str


class RefutationLoopRecord(BaseModel):
    model_config = _FROZEN

    iteration: int
    triggering_hypothesis_id: str
    targets_claim_id: str
    coverage_after: float = Field(ge=0.0, le=1.0)


class MemoryLedger(BaseModel):
    model_config = _FROZEN

    run_id: str
    retrieved_ids: dict[str, list[str]] = Field(default_factory=dict)
    exhausted_queries: list[str] = Field(default_factory=list)
    coverage_history: list[CoverageHistoryEntry] = Field(default_factory=list)
    gap_history: list[GapHistoryEntry] = Field(default_factory=list)
    refutation_hypotheses_tested: list[RefutationHypothesis] = Field(default_factory=list)
    refutation_loop_history: list[RefutationLoopRecord] = Field(default_factory=list)
    supported_candidates: dict[str, list[str]] = Field(default_factory=dict)
    session_scope: str = "session"


# ---------------------------------------------------------------------------
# Answer (spec §3.8)
# ---------------------------------------------------------------------------


class AnswerClaim(BaseModel):
    model_config = _FROZEN

    claim_text: str
    source_ids: list[str] = Field(default_factory=list)
    claim_type: ClaimType


class DisclosedGap(BaseModel):
    model_config = _FROZEN

    slot_id: str
    gap_description: str


class DisclosedContradiction(BaseModel):
    model_config = _FROZEN

    description: str
    conflicting_ids: list[str] = Field(default_factory=list)


class DisclosedRefutation(BaseModel):
    model_config = _FROZEN

    targets_claim_id: str
    refuting_evidence_ids: list[str] = Field(default_factory=list)
    refutation_verdict: RefutationVerdict
    strategy: RefutationStrategy


class AnswerSchema(BaseModel):
    model_config = _FROZEN

    answer_text: str
    claims: list[AnswerClaim] = Field(default_factory=list)
    disclosed_gaps: list[DisclosedGap] = Field(default_factory=list)
    disclosed_contradictions: list[DisclosedContradiction] = Field(default_factory=list)
    disclosed_refutations: list[DisclosedRefutation] = Field(default_factory=list)
    adversarially_probed: bool = False
    degradation_level: DegradationLevel = DegradationLevel.NORMAL


# ---------------------------------------------------------------------------
# Output Governance (spec §3.9)
# ---------------------------------------------------------------------------


class GovernanceViolation(BaseModel):
    """One issue raised by the Output Governance gate.

    ``numerical_mismatch`` is a constitutional violation — the answer
    cited a number not present in any of its source facts. The
    orchestrator escalates this to Hard Halt.

    ``undisclosed_refutation`` means the AnswerSchema's
    disclosed_refutations[] is missing a hypothesis the report
    flagged as non-unrefuted. Recoverable with one Generator retry.

    ``badge_mismatch`` means ``adversarially_probed`` on the answer
    does not match the orchestrator's flag. Recoverable by overwriting
    the badge.
    """

    model_config = _FROZEN

    severity: GovernanceSeverity
    message: str
    claim_index: int | None = None
    hypothesis_id: str | None = None
    expected: str | None = None
    actual: str | None = None


# ---------------------------------------------------------------------------
# Execution Trace (spec §7.2)
# ---------------------------------------------------------------------------


class CoverageProgressionEntry(BaseModel):
    model_config = _FROZEN

    slot_id: str
    iteration: int
    coverage_score: float = Field(ge=0.0, le=1.0)


class FinalSlotState(BaseModel):
    model_config = _FROZEN

    slot_id: str
    terminal_verdict: VerifierVerdict
    final_coverage: float = Field(ge=0.0, le=1.0)


class QueryTraceInfo(BaseModel):
    model_config = _FROZEN

    original_query: str
    complexity_tier: ComplexityTier
    validation_status: str = "passed"


class ExecutionTrace(BaseModel):
    model_config = _FROZEN

    run_id: str
    query: QueryTraceInfo
    decomposition_plan: DecompositionPlan
    retrieval_passes: list[RetrievalRecord] = Field(default_factory=list)
    verifier_verdicts: list[VerifierOutput] = Field(default_factory=list)
    refutation_report: RefutationReport | None = None
    coverage_progression: list[CoverageProgressionEntry] = Field(default_factory=list)
    refutation_loop_iterations: list[RefutationLoopRecord] = Field(default_factory=list)
    final_slot_states: list[FinalSlotState] = Field(default_factory=list)
    answer: AnswerSchema | None = None
    governance_violations: list[GovernanceViolation] = Field(default_factory=list)
    degradation_level: DegradationLevel = DegradationLevel.NORMAL
    degradation_cause: DegradationCause = DegradationCause.none
    total_tokens_consumed: int = Field(default=0, ge=0)
    total_iterations: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    models_used: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

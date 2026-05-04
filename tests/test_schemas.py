"""Block 1 done-when test: every schema instantiates with valid data and
round-trips through JSON; every enum has the expected number of members."""
from __future__ import annotations

import json
from datetime import date

import pytest
from pydantic import BaseModel

from schemas import (
    AnswerSchema,
    ChunkRecord,
    DecompositionPlan,
    DegradationLevel,
    EvidenceSlot,
    EvidenceType,
    ExecutionTrace,
    FactRecord,
    FactType,
    MemoryLedger,
    PassOrigin,
    RefutationHypothesis,
    RefutationReport,
    RefutationStrategy,
    RetrievalCandidate,
    RetrievalRecord,
    VerifierOutput,
)
from schemas.enums import (
    CandidateSource,
    ClaimType,
    ComplexityTier,
    RefutationOverallVerdict,
    RefutationVerdict,
    SynthesisStrategy,
    TargetLayer,
    VerifierVerdict,
)
from schemas.records import (
    AnswerClaim,
    DisclosedGap,
    DisclosedRefutation,
    QueryTraceInfo,
)
from schemas.period import format_period, is_valid_period, parse_period


# ---------------------------------------------------------------------------
# Enum cardinality
# ---------------------------------------------------------------------------


def test_fact_type_has_seven_members():
    assert len(list(FactType)) == 7


def test_evidence_type_has_ten_members():
    assert len(list(EvidenceType)) == 10


def test_refutation_strategy_has_seven_members():
    assert len(list(RefutationStrategy)) == 7


def test_pass_origin_has_three_members():
    assert len(list(PassOrigin)) == 3


def test_degradation_level_int_values():
    assert DegradationLevel.NORMAL == 0
    assert DegradationLevel.PARTIAL == 1
    assert DegradationLevel.CLARIFICATION_REQUEST == 2
    assert DegradationLevel.HARD_HALT == 3


# ---------------------------------------------------------------------------
# Period parser
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "s",
    ["2024Q3", "2016Q1", "FY2023", "FY2025-guidance", "2024-09-30"],
)
def test_period_round_trip(s: str):
    p = parse_period(s)
    assert format_period(p) == s
    assert str(p) == s
    assert is_valid_period(s)


@pytest.mark.parametrize(
    "s", ["2024Q5", "FY24", "2024/09/30", "Q3 2024", "", "garbage"]
)
def test_period_invalid_strings_rejected(s: str):
    assert not is_valid_period(s)
    with pytest.raises(ValueError):
        parse_period(s)


# ---------------------------------------------------------------------------
# Round-trip helper
# ---------------------------------------------------------------------------


def _roundtrip(model: BaseModel) -> None:
    payload = model.model_dump_json()
    json.loads(payload)  # well-formed JSON
    cls = type(model)
    rebuilt = cls.model_validate_json(payload)
    assert rebuilt == model


# ---------------------------------------------------------------------------
# Fixtures: minimal valid instances
# ---------------------------------------------------------------------------


@pytest.fixture
def xbrl_fact() -> FactRecord:
    return FactRecord(
        fact_id="F-XBRL-0001065280-24-000093-OperatingIncomeLoss-2024Q3",
        claim="Netflix's operating income for Q3 2024 was $2,907,847 thousand.",
        asserter="Netflix",
        source_document="nflx-10q-2024-q3",
        source_section="Condensed Consolidated Statements of Operations",
        verbatim_anchor="Operating income — Three months ended September 30, 2024 — $2,907,847",
        fact_type=FactType.financial_metric,
        period="2024Q3",
        value=2_907_847_000.0,
        unit="USD",
        concept_tag="us-gaap:OperatingIncomeLoss",
        assertion_date=date(2024, 10, 17),
        confidence=1.00,
    )


@pytest.fixture
def prose_fact() -> FactRecord:
    return FactRecord(
        fact_id="F-PROSE-008912",
        claim="Netflix has no current plans to introduce advertising on its service.",
        asserter="Reed Hastings",
        source_document="nflx-q3-2018-letter",
        source_section="Q&A — Advertising and pricing",
        verbatim_anchor="We don't have any plans to add ads to Netflix",
        fact_type=FactType.strategic_claim,
        period="2018Q3",
        assertion_date=date(2018, 10, 16),
        confidence=0.95,
    )


@pytest.fixture
def chunk() -> ChunkRecord:
    return ChunkRecord(
        chunk_id="nflx-10q-2024-q3__item7_mda__chunk_3",
        text="Revenue increased due to growth in paid memberships and ARM...",
        source_document="nflx-10q-2024-q3",
        section="Item 7 — MD&A",
        position_index=3,
        word_count=487,
    )


@pytest.fixture
def slot() -> EvidenceSlot:
    return EvidenceSlot(
        slot_id="S1",
        sub_question="What was Netflix's operating income for Q3 2024?",
        evidence_type=EvidenceType.specific_metric,
        target_layer=TargetLayer.fact_store,
        period_filter="2024Q3",
        key_terms=["operating income", "Q3 2024"],
        coverage_threshold=0.80,
    )


@pytest.fixture
def plan(slot: EvidenceSlot) -> DecompositionPlan:
    return DecompositionPlan(
        query_id="run-1",
        original_query="What was Netflix's operating income for Q3 2024?",
        complexity_tier=ComplexityTier.simple,
        synthesis_strategy=SynthesisStrategy.integrate,
        slots=[slot],
    )


@pytest.fixture
def retrieval() -> RetrievalRecord:
    return RetrievalRecord(
        retrieval_id="R1_S1_iter1",
        slot_id="S1",
        iteration=1,
        pass_origin=PassOrigin.verifier_loop,
        vector_query="Netflix operating income Q3 2024",
        bm25_terms=["operating income", "Q3 2024"],
        period_filter="2024Q3",
        candidates=[
            RetrievalCandidate(
                candidate_id="F-XBRL-0001065280-24-000093-OperatingIncomeLoss-2024Q3",
                source=CandidateSource.fact,
                rrf_score=0.0312,
                vector_score=0.91,
                bm25_score=12.4,
            )
        ],
        memory_exclusions=[],
    )


@pytest.fixture
def verifier_output() -> VerifierOutput:
    return VerifierOutput(
        slot_id="S1",
        coverage_score=0.95,
        verdict=VerifierVerdict.covered,
        supported_candidates=["F-XBRL-0001065280-24-000093-OperatingIncomeLoss-2024Q3"],
    )


@pytest.fixture
def hypothesis() -> RefutationHypothesis:
    return RefutationHypothesis(
        hypothesis_id="H1",
        targets_claim_id="F-PROSE-008912",
        hypothesis_text=(
            "Netflix later reversed its no-advertising stance and launched an ad-supported tier."
        ),
        rationale=(
            "If a later strategic_claim from Netflix announces an ad-supported tier, "
            "the 2018 no-plans-for-ads claim has been reversed."
        ),
        strategy=RefutationStrategy.later_reversal,
        retrieval_record_id="R1_H1",
        refutation_verdict=RefutationVerdict.strongly_refuted,
        evidence_ids=["F-PROSE-014501"],
    )


@pytest.fixture
def refutation_report(hypothesis: RefutationHypothesis) -> RefutationReport:
    return RefutationReport(
        run_id="run-1",
        model_used="mistralai/Mixtral-8x22B-Instruct-v0.1",
        hypotheses=[hypothesis],
        overall_verdict=RefutationOverallVerdict.refutation_to_loop,
        triggered_loop_reentry=True,
        loop_reentry_iteration=2,
    )


@pytest.fixture
def memory_ledger(hypothesis: RefutationHypothesis) -> MemoryLedger:
    return MemoryLedger(
        run_id="run-1",
        retrieved_ids={"S1": ["F-XBRL-0001065280-24-000093-OperatingIncomeLoss-2024Q3"]},
        exhausted_queries=["Netflix operating income Q3 2024"],
        refutation_hypotheses_tested=[hypothesis],
    )


@pytest.fixture
def answer() -> AnswerSchema:
    return AnswerSchema(
        answer_text="Netflix's operating income for Q3 2024 was $2,907,847 thousand.",
        claims=[
            AnswerClaim(
                claim_text="Netflix's operating income for Q3 2024 was $2,907,847 thousand.",
                source_ids=["F-XBRL-0001065280-24-000093-OperatingIncomeLoss-2024Q3"],
                claim_type=ClaimType.grounded,
            )
        ],
        disclosed_gaps=[],
        disclosed_refutations=[
            DisclosedRefutation(
                targets_claim_id="F-PROSE-008912",
                refuting_evidence_ids=["F-PROSE-014501"],
                refutation_verdict=RefutationVerdict.strongly_refuted,
                strategy=RefutationStrategy.later_reversal,
            )
        ],
        adversarially_probed=True,
        degradation_level=DegradationLevel.NORMAL,
    )


@pytest.fixture
def trace(
    plan: DecompositionPlan,
    retrieval: RetrievalRecord,
    verifier_output: VerifierOutput,
    refutation_report: RefutationReport,
) -> ExecutionTrace:
    return ExecutionTrace(
        run_id="run-1",
        query=QueryTraceInfo(
            original_query=plan.original_query,
            complexity_tier=ComplexityTier.simple,
        ),
        decomposition_plan=plan,
        retrieval_passes=[retrieval],
        verifier_verdicts=[verifier_output],
        refutation_report=refutation_report,
        degradation_level=DegradationLevel.NORMAL,
        total_tokens_consumed=1234,
        total_iterations=2,
        elapsed_seconds=14.7,
        models_used=[
            "meta-llama/Llama-3.3-70B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
        ],
    )


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


def test_xbrl_fact_roundtrip(xbrl_fact: FactRecord):
    _roundtrip(xbrl_fact)


def test_prose_fact_roundtrip(prose_fact: FactRecord):
    _roundtrip(prose_fact)


def test_chunk_roundtrip(chunk: ChunkRecord):
    _roundtrip(chunk)


def test_slot_roundtrip(slot: EvidenceSlot):
    _roundtrip(slot)


def test_plan_roundtrip(plan: DecompositionPlan):
    _roundtrip(plan)


def test_retrieval_roundtrip(retrieval: RetrievalRecord):
    _roundtrip(retrieval)


def test_verifier_output_roundtrip(verifier_output: VerifierOutput):
    _roundtrip(verifier_output)


def test_refutation_report_roundtrip(refutation_report: RefutationReport):
    _roundtrip(refutation_report)


def test_memory_ledger_roundtrip(memory_ledger: MemoryLedger):
    _roundtrip(memory_ledger)


def test_answer_roundtrip(answer: AnswerSchema):
    _roundtrip(answer)


def test_execution_trace_roundtrip(trace: ExecutionTrace):
    _roundtrip(trace)


# ---------------------------------------------------------------------------
# Validator behavior
# ---------------------------------------------------------------------------


def test_invalid_period_on_fact_record_rejected():
    with pytest.raises(Exception):
        FactRecord(
            fact_id="F-PROSE-000001",
            claim="x",
            asserter="Netflix",
            source_document="nflx-q3-2018-letter",
            source_section="Q&A",
            verbatim_anchor="x",
            fact_type=FactType.strategic_claim,
            period="not-a-period",
            assertion_date=date(2018, 10, 16),
            confidence=0.9,
        )


def test_confidence_out_of_range_rejected():
    with pytest.raises(Exception):
        FactRecord(
            fact_id="F-PROSE-000002",
            claim="x",
            asserter="Netflix",
            source_document="nflx-q3-2018-letter",
            source_section="Q&A",
            verbatim_anchor="x",
            fact_type=FactType.strategic_claim,
            assertion_date=date(2018, 10, 16),
            confidence=1.5,
        )


def test_extra_fields_rejected_on_chunk():
    with pytest.raises(Exception):
        ChunkRecord(
            chunk_id="c1",
            text="x",
            source_document="d1",
            section="s1",
            position_index=0,
            word_count=10,
            unexpected_field=True,  # type: ignore[call-arg]
        )


def test_disclosed_gap_construction():
    # smoke: type imported and constructible
    g = DisclosedGap(slot_id="S1", gap_description="missing post-2024 data")
    assert g.slot_id == "S1"

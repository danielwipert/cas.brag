from enum import Enum, IntEnum


class FactType(str, Enum):
    financial_metric = "financial_metric"
    operational_metric = "operational_metric"
    forward_guidance = "forward_guidance"
    strategic_claim = "strategic_claim"
    causal_explanation = "causal_explanation"
    risk_disclosure = "risk_disclosure"
    accounting_policy = "accounting_policy"


class EvidenceType(str, Enum):
    specific_metric = "specific_metric"
    definition = "definition"
    forward_looking_statement = "forward_looking_statement"
    strategic_position = "strategic_position"
    cross_period_comparison = "cross_period_comparison"
    causal_explanation = "causal_explanation"
    temporal_evolution = "temporal_evolution"
    risk_disclosure = "risk_disclosure"
    accounting_policy = "accounting_policy"
    contradiction_detection = "contradiction_detection"


class RefutationStrategy(str, Enum):
    restated_value = "restated_value"
    revised_value = "revised_value"
    guidance_vs_actual = "guidance_vs_actual"
    later_reversal = "later_reversal"
    alternative_cause = "alternative_cause"
    materialization = "materialization"
    policy_change = "policy_change"


class PassOrigin(str, Enum):
    verifier_loop = "verifier_loop"
    refutation_loop = "refutation_loop"
    refutation_probe = "refutation_probe"


class DegradationLevel(IntEnum):
    NORMAL = 0
    PARTIAL = 1
    CLARIFICATION_REQUEST = 2
    HARD_HALT = 3


class ComplexityTier(str, Enum):
    simple = "simple"
    standard = "standard"
    complex = "complex"


class TargetLayer(str, Enum):
    fact_store = "fact_store"
    chunk_store = "chunk_store"
    both = "both"


class SynthesisStrategy(str, Enum):
    compare = "compare"
    contrast = "contrast"
    sequence = "sequence"
    integrate = "integrate"


class VerifierVerdict(str, Enum):
    covered = "covered"
    gap = "gap"
    contradiction = "contradiction"
    exhausted = "exhausted"


class RefutationVerdict(str, Enum):
    unrefuted = "unrefuted"
    weakly_refuted = "weakly_refuted"
    strongly_refuted = "strongly_refuted"


class RefutationOverallVerdict(str, Enum):
    answer_strengthened = "answer_strengthened"
    refutation_to_loop = "refutation_to_loop"
    refutation_to_partial = "refutation_to_partial"


class ClaimType(str, Enum):
    grounded = "grounded"
    derived = "derived"
    interpretive = "interpretive"


class DegradationCause(str, Enum):
    none = "none"
    slot_exhaustion = "slot_exhaustion"
    refutation_unresolved = "refutation_unresolved"
    input_failure = "input_failure"
    constitutional_violation = "constitutional_violation"
    verifier_unavailable = "verifier_unavailable"
    refutation_unavailable = "refutation_unavailable"
    generator_unavailable = "generator_unavailable"
    governance_failure = "governance_failure"


class GovernanceSeverity(str, Enum):
    numerical_mismatch = "numerical_mismatch"
    undisclosed_refutation = "undisclosed_refutation"
    badge_mismatch = "badge_mismatch"


class CandidateSource(str, Enum):
    fact = "fact"
    chunk = "chunk"

"""Unit tests for pipeline.degradation.decide_degradation.

The Block 22 change widened the substantive-evidence count to include
``contradiction`` verdicts that carry supported_candidates — without
that, strong-refutation queries dropped to CR whenever the Verifier
landed on ``contradiction`` instead of ``covered`` (observed
intermittently for Q12 on the adversarial suite).
"""
from __future__ import annotations

from pipeline.degradation import decide_degradation
from schemas.enums import DegradationCause, DegradationLevel, VerifierVerdict
from schemas.records import VerifierOutput


def _v(
    slot_id: str,
    verdict: VerifierVerdict,
    coverage: float,
    supported: list[str] | None = None,
) -> VerifierOutput:
    return VerifierOutput(
        slot_id=slot_id,
        coverage_score=coverage,
        verdict=verdict,
        gap_description=None,
        contradiction_details=[],
        supported_candidates=list(supported or []),
        rejected_candidates=[],
    )


def test_all_covered_no_contradiction_is_normal() -> None:
    level, cause = decide_degradation([
        _v("S1", VerifierVerdict.covered, 0.85, supported=["F-A"]),
        _v("S2", VerifierVerdict.covered, 0.90, supported=["F-B"]),
    ])
    assert level == DegradationLevel.NORMAL
    assert cause == DegradationCause.none


def test_some_covered_some_exhausted_is_partial() -> None:
    level, cause = decide_degradation([
        _v("S1", VerifierVerdict.covered, 0.85, supported=["F-A"]),
        _v("S2", VerifierVerdict.exhausted, 0.4),
    ])
    assert level == DegradationLevel.PARTIAL
    assert cause == DegradationCause.slot_exhaustion


def test_nothing_substantive_is_cr() -> None:
    level, cause = decide_degradation([
        _v("S1", VerifierVerdict.exhausted, 0.0),
        _v("S2", VerifierVerdict.gap, 0.2),
    ])
    assert level == DegradationLevel.CLARIFICATION_REQUEST
    assert cause == DegradationCause.slot_exhaustion


def test_empty_outputs_is_cr() -> None:
    level, cause = decide_degradation([])
    assert level == DegradationLevel.CLARIFICATION_REQUEST
    assert cause == DegradationCause.slot_exhaustion


def test_lone_contradiction_with_supported_is_partial() -> None:
    # Block 22 change: previously this returned CR because n_covered=0.
    # The Verifier returning contradiction with supported_candidates is
    # the strong-refutation signal — degrade to Partial so refutation
    # can run, not CR (which bypasses it).
    level, cause = decide_degradation([
        _v("S1", VerifierVerdict.contradiction, 0.80, supported=["F-A", "F-B"]),
    ])
    assert level == DegradationLevel.PARTIAL
    assert cause == DegradationCause.none


def test_lone_contradiction_without_supported_is_cr() -> None:
    # No supported facts means truly nothing substantive; the spec's
    # "ask the user to narrow" branch still applies.
    level, cause = decide_degradation([
        _v("S1", VerifierVerdict.contradiction, 0.30, supported=[]),
    ])
    assert level == DegradationLevel.CLARIFICATION_REQUEST
    assert cause == DegradationCause.slot_exhaustion


def test_covered_plus_contradiction_with_supported_is_partial() -> None:
    level, cause = decide_degradation([
        _v("S1", VerifierVerdict.covered, 0.85, supported=["F-A"]),
        _v("S2", VerifierVerdict.contradiction, 0.75, supported=["F-B", "F-C"]),
    ])
    assert level == DegradationLevel.PARTIAL
    # No exhausted slot — cause is "none" (contradiction is the
    # informative finding; slot_exhaustion would be misleading).
    assert cause == DegradationCause.none

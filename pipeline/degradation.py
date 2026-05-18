"""Block 9c: Degradation level decision.

Maps the per-slot Verifier verdicts to one of four degradation levels
per spec §3.7. Level 3 (Hard Halt) is set by callers upstream — this
module only decides between Normal / Partial / Clarification Request
given a set of final slot verdicts.

  Normal              — all slots covered, no contradictions
  Partial             — at least one slot covered AND at least one
                        exhausted OR contradicted
  Clarification Req.  — zero slots covered (system can't answer with
                        current evidence; ask the user to narrow)
  Hard Halt           — input validation failed, LLM unavailable, etc.
                        (set by orchestrator before degradation runs)
"""
from __future__ import annotations

from collections.abc import Iterable

from schemas.enums import DegradationCause, DegradationLevel, VerifierVerdict
from schemas.records import VerifierOutput


def decide_degradation(
    verifier_outputs: Iterable[VerifierOutput],
) -> tuple[DegradationLevel, DegradationCause]:
    """Return ``(level, cause)`` for the final slot verdicts.

    Only call this when the run completed without an input-validation
    failure or an LLM availability error — those caller-known states
    short-circuit to Hard Halt before this function is invoked."""
    outputs = list(verifier_outputs)
    if not outputs:
        # No slots ran (unusual — Planner produced no slots). Treat as
        # clarification request.
        return (DegradationLevel.CLARIFICATION_REQUEST, DegradationCause.slot_exhaustion)

    n_covered = sum(1 for v in outputs if v.verdict == VerifierVerdict.covered)
    n_exhausted = sum(1 for v in outputs if v.verdict == VerifierVerdict.exhausted)
    n_contradiction = sum(1 for v in outputs if v.verdict == VerifierVerdict.contradiction)
    n_gap = sum(1 for v in outputs if v.verdict == VerifierVerdict.gap)

    # Level 0: every slot covered, no contradictions.
    if n_covered == len(outputs) and n_contradiction == 0:
        return (DegradationLevel.NORMAL, DegradationCause.none)

    # Level 2: nothing covered — the system has nothing solid to answer
    # with. Ask the user to narrow the question.
    if n_covered == 0:
        return (
            DegradationLevel.CLARIFICATION_REQUEST,
            DegradationCause.slot_exhaustion,
        )

    # Level 1: at least one slot covered, but at least one exhausted or
    # contradicted or open gap. Generator can produce a partial answer
    # over the covered slots with disclosure of the missing ones.
    if n_exhausted > 0 or n_contradiction > 0 or n_gap > 0:
        cause = (
            DegradationCause.slot_exhaustion
            if n_exhausted > 0 else DegradationCause.none
        )
        return (DegradationLevel.PARTIAL, cause)

    # Defensive fallback.
    return (DegradationLevel.PARTIAL, DegradationCause.slot_exhaustion)

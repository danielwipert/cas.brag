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

    # Block 22: count slots whose verdict is contradiction-with-supported-
    # facts as substantive evidence (alongside covered). The Verifier
    # returning `contradiction` is exactly what we want on strong-refutation
    # queries — it means it saw both sides of an issue in the candidate set.
    # Treating that as "nothing covered → CR" bypassed refutation entirely
    # and was the source of Q12 stochastically dropping from 5/5 to 3/5
    # whenever the Verifier landed on `contradiction` instead of `covered`.
    # A contradiction verdict with zero supported_candidates is still
    # "nothing substantive" and falls through to the CR branch below.
    n_substantive = n_covered + sum(
        1 for v in outputs
        if v.verdict == VerifierVerdict.contradiction
        and v.supported_candidates
    )

    # Level 0: every slot covered, no contradictions.
    if n_covered == len(outputs) and n_contradiction == 0:
        return (DegradationLevel.NORMAL, DegradationCause.none)

    # Level 2: nothing substantive — the system has nothing solid to answer
    # with. Ask the user to narrow the question.
    if n_substantive == 0:
        return (
            DegradationLevel.CLARIFICATION_REQUEST,
            DegradationCause.slot_exhaustion,
        )

    # Level 1: at least one substantive slot (covered or contradiction-
    # with-supported), but the clean Normal case doesn't apply. Generator
    # can produce a partial answer over the covered + contradicted slots
    # with disclosure of the missing ones and the contradictions.
    cause = (
        DegradationCause.slot_exhaustion
        if n_exhausted > 0 else DegradationCause.none
    )
    return (DegradationLevel.PARTIAL, cause)

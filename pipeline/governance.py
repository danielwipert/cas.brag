"""Block 14: Output Governance gate (Stage 8 — spec §3.9).

The Governance gate is the LAST constitutional check before the answer
is released to the user. It enforces three properties the spec promises
about every BRAG answer:

  1. **Numerical fidelity.** Every numerical token in a grounded claim
     must appear in the cited facts' claim text, value, unit, or
     period. The spec's central trust promise is that a cited number
     is reproduced verbatim — Governance is the regex-level
     defense-in-depth check that catches a Generator that drifted into
     rounding or paraphrase. ``numerical_mismatch`` is a constitutional
     violation; the orchestrator escalates to Hard Halt.

  2. **Refutation disclosure.** Every RefutationHypothesis with
     verdict in {weakly_refuted, strongly_refuted} must have a
     matching DisclosedRefutation entry on the AnswerSchema. The
     ``generate_answer`` wrapper builds this list mechanically, so this
     check is defense-in-depth — but if a caller bypasses the wrapper
     this gate catches the omission. Recoverable with one Generator
     retry; if still missing, the orchestrator injects manually and
     degrades to Partial.

  3. **Adversarially-probed badge propagation.** The badge on the
     answer must equal the orchestrator's flag (which is True iff the
     Refutation Agent actually ran). Recoverable by overwriting the
     badge.

The gate is pure (no I/O, no LLM calls) so it can be exercised in
unit tests without the network.
"""
from __future__ import annotations

import re

from schemas.enums import ClaimType, GovernanceSeverity, RefutationVerdict
from schemas.records import (
    AnswerSchema,
    FactRecord,
    GovernanceViolation,
    RefutationReport,
)


# Number tokens: comma-grouped ($8,187), bare floats (12.9), bare
# integers (260). The pattern intentionally requires either a comma or
# a decimal point — bare single tokens like "1" or "2" by themselves
# show up in fact_ids and aren't worth flagging. Year tokens (1900-2099)
# are filtered out in ``_extract_significant_numbers`` because they
# carry no numerical-fidelity risk.
_NUM_TOKEN_RE = re.compile(
    r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"   # 8,187 or 1,234,567 or 8,187.5
    r"|\b\d+\.\d+\b"                       # 12.9 or 260.28
    r"|\b\d{2,}\b"                         # 8187 or 260 — at least 2 digits
)

_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")


def _strip_thousands(token: str) -> str:
    """Normalize '8,187' → '8187' so we can match commaless variants."""
    return token.replace(",", "")


def _extract_significant_numbers(text: str) -> set[str]:
    """Pull number-like tokens from ``text``, dropping pure year tokens.

    Returns BOTH the original form ('8,187') AND the commaless form
    ('8187') so a claim using one rendering matches a fact using the
    other. Pure year tokens (1900-2099) are excluded — they're already
    constrained by the period_filter machinery and the cited fact's
    period string, so they don't need fidelity checking here."""
    out: set[str] = set()
    for tok in _NUM_TOKEN_RE.findall(text):
        if _YEAR_RE.match(_strip_thousands(tok)):
            continue
        out.add(tok)
        if "," in tok:
            out.add(_strip_thousands(tok))
    return out


def _fact_numeric_corpus(fact: FactRecord) -> set[str]:
    """Build the set of numeric tokens the Generator is permitted to
    reproduce when citing this fact.

    Sources combined:
      - fact.claim (the canonical statement the prose extractor wrote)
      - fact.value rendered as int and as the original float
      - fact.value rendered as comma-grouped integer when applicable
      - fact.period (caught even though ``_extract_significant_numbers``
        drops pure year tokens — quarter numbers and FY-with-Q digits
        are still relevant)
      - fact.assertion_date components — the date is intrinsic
        metadata; the spec requires the Generator to surface
        assertion_dates in temporal-evolution narratives, so the
        date's month/day digits ('01', '22' from '2018-01-22') must
        be allowed through the numerical-fidelity gate.
    """
    corpus: set[str] = set()
    corpus |= _extract_significant_numbers(fact.claim)
    if fact.value is not None:
        val = fact.value
        int_part = int(val)
        corpus.add(str(int_part))
        corpus.add(f"{int_part:,}")
        if val != int_part:
            corpus.add(str(val))
            decimal = str(val).split(".", 1)[1]
            corpus.add(f"{int_part:,}.{decimal}")
        # Tens of millions (e.g. 8187 → '81.87' for hundreds-of-millions
        # framing) is intentionally NOT generated here — that's the
        # unit-conversion vector we explicitly forbid.
    if fact.period:
        corpus |= _extract_significant_numbers(fact.period)
    iso = fact.assertion_date.isoformat()
    corpus |= _extract_significant_numbers(iso)
    # Cover both zero-padded and unpadded month/day forms in case the
    # Generator drops the leading zero ("April 5, 2018" → '5' or '05').
    corpus.add(f"{fact.assertion_date.month:02d}")
    corpus.add(f"{fact.assertion_date.day:02d}")
    corpus.add(str(fact.assertion_date.month))
    corpus.add(str(fact.assertion_date.day))
    return corpus


def _claim_has_numerical_content(text: str) -> bool:
    return bool(_extract_significant_numbers(text))


def check_governance(
    *,
    answer: AnswerSchema,
    verified_facts: list[FactRecord],
    refutation_report: RefutationReport | None,
    expected_adversarially_probed: bool,
) -> list[GovernanceViolation]:
    """Run all three Governance checks and return the list of
    violations. An empty list means the answer passed."""
    violations: list[GovernanceViolation] = []
    fact_by_id = {f.fact_id: f for f in verified_facts}

    # ---- 1. Numerical fidelity -------------------------------------
    for i, claim in enumerate(answer.claims):
        if claim.claim_type != ClaimType.grounded:
            continue
        if not _claim_has_numerical_content(claim.claim_text):
            continue
        cited_corpus: set[str] = set()
        for sid in claim.source_ids:
            f = fact_by_id.get(sid)
            if f is None:
                continue
            cited_corpus |= _fact_numeric_corpus(f)
        for tok in _extract_significant_numbers(claim.claim_text):
            if tok in cited_corpus:
                continue
            # Tolerant: comma form vs commaless form of the same digits.
            if _strip_thousands(tok) in cited_corpus:
                continue
            violations.append(
                GovernanceViolation(
                    severity=GovernanceSeverity.numerical_mismatch,
                    message=(
                        f"Claim #{i + 1} contains numeric token "
                        f"{tok!r} that does not appear in any cited "
                        f"fact's value/claim/period."
                    ),
                    claim_index=i,
                    expected=", ".join(sorted(cited_corpus)) or "(no numeric tokens in cited facts)",
                    actual=tok,
                )
            )

    # ---- 2. Refutation disclosure ----------------------------------
    if refutation_report is not None:
        disclosed_keys = {
            (d.targets_claim_id, d.strategy)
            for d in answer.disclosed_refutations
        }
        for h in refutation_report.hypotheses:
            if h.refutation_verdict == RefutationVerdict.unrefuted:
                continue
            key = (h.targets_claim_id, h.strategy)
            if key in disclosed_keys:
                continue
            violations.append(
                GovernanceViolation(
                    severity=GovernanceSeverity.undisclosed_refutation,
                    message=(
                        f"Hypothesis {h.hypothesis_id} "
                        f"({h.strategy.value}, verdict="
                        f"{h.refutation_verdict.value}) targeting "
                        f"{h.targets_claim_id} has no matching "
                        f"DisclosedRefutation entry on the answer."
                    ),
                    hypothesis_id=h.hypothesis_id,
                    expected=f"{h.targets_claim_id} + {h.strategy.value}",
                    actual="(missing)",
                )
            )

    # ---- 3. Adversarially-probed badge -----------------------------
    if answer.adversarially_probed != expected_adversarially_probed:
        violations.append(
            GovernanceViolation(
                severity=GovernanceSeverity.badge_mismatch,
                message=(
                    f"adversarially_probed badge mismatch — orchestrator "
                    f"flag={expected_adversarially_probed}, answer "
                    f"flag={answer.adversarially_probed}."
                ),
                expected=str(expected_adversarially_probed),
                actual=str(answer.adversarially_probed),
            )
        )

    return violations


def format_violations_for_retry(violations: list[GovernanceViolation]) -> str:
    """Render the Governance findings as a string the Generator can
    consume in its retry user message. Only used for the
    ``undisclosed_refutation`` retry path."""
    lines = ["The Governance gate flagged the following issues:"]
    for v in violations:
        lines.append(f"  - [{v.severity.value}] {v.message}")
    lines.append(
        "Revise the answer_text to surface every flagged refutation "
        "per the system prompt's refutation handling rules. Do NOT "
        "alter any numerical claim — those passed."
    )
    return "\n".join(lines)

"""Block 13 test harness: drive the Generator prompt over 8 hand-built
(verified_facts + refutation_report + degradation_level) tuples and
grade output against the spec's quality bar.

Pass criteria per case (build plan v3 §Block 13):

  S1. AnswerSchema is Pydantic-valid (always — the generator returns
      a typed object, so this is checked at construction time).
  S2. No fabricated fact_ids in ``claims[].source_ids`` (enforced
      inside generate_answer; any claim is rejected if it cites
      nothing real).
  Q1. NUMERICAL FIDELITY — for each verified fact with a value, the
      value string OR a normalized variant appears in answer_text
      AND no rounded/paraphrased numbers appear in its place.
  Q2. DISCLOSED REFUTATIONS — the disclosed_refutations[] list has
      one entry per hypothesis with verdict ≠ unrefuted.
  Q3. REFUTATION DISCLOSURE IN PROSE — for each strongly_refuted
      hypothesis, the answer_text mentions both the targeted fact's
      assertion_date and the refuting evidence's assertion_date.
  Q4. ADVERSARIALLY_PROBED PROPAGATION — the schema's flag matches
      the harness's input flag (the LLM cannot invent it).

Pass bar: 6/8 cases pass.

Run from repo root::

    python -m tests.generator_prompt_harness

Requires OPENROUTER_API_KEY in env or .env.
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from agents.generator.prompt import (
    GENERATOR_MODEL,
    generate_answer,
)
from schemas.enums import (
    ClaimType,
    DegradationLevel,
    FactType,
    RefutationOverallVerdict,
    RefutationStrategy,
    RefutationVerdict,
)
from schemas.records import (
    AnswerSchema,
    FactRecord,
    RefutationHypothesis,
    RefutationReport,
)


# ---------------------------------------------------------------------------
# Hand-built verified evidence sets
# ---------------------------------------------------------------------------


def _f(
    *,
    fact_id: str,
    fact_type: FactType,
    claim: str,
    asserter: str,
    source_document: str,
    source_section: str,
    verbatim_anchor: str,
    assertion_date: date,
    period: str | None = None,
    value: float | None = None,
    unit: str | None = None,
    concept_tag: str | None = None,
    confidence: float = 0.95,
) -> FactRecord:
    return FactRecord(
        fact_id=fact_id, claim=claim, asserter=asserter,
        source_document=source_document, source_section=source_section,
        verbatim_anchor=verbatim_anchor, fact_type=fact_type,
        period=period, value=value, unit=unit, concept_tag=concept_tag,
        assertion_date=assertion_date, confidence=confidence,
    )


def _h(
    *,
    hypothesis_id: str,
    targets_claim_id: str,
    hypothesis_text: str,
    rationale: str,
    strategy: RefutationStrategy,
    refutation_verdict: RefutationVerdict,
    evidence_ids: list[str],
) -> RefutationHypothesis:
    return RefutationHypothesis(
        hypothesis_id=hypothesis_id,
        targets_claim_id=targets_claim_id,
        hypothesis_text=hypothesis_text,
        rationale=rationale,
        strategy=strategy,
        refutation_verdict=refutation_verdict,
        evidence_ids=evidence_ids,
    )


@dataclass(frozen=True)
class TestCase:
    case_id: str
    description: str
    query: str
    verified_facts: list[FactRecord]
    refutation_report: RefutationReport | None
    degradation_level: DegradationLevel
    adversarially_probed: bool
    # IDs of facts whose ``assertion_date`` MUST appear in answer_text
    # (used by Q3 to verify the temporal-evolution narrative).
    required_dates_in_text: list[str] = field(default_factory=list)
    # Strings the answer text MUST contain verbatim (used to harden
    # hostile cases like G8 — forces a specific value through).
    required_substrings: list[str] = field(default_factory=list)
    # Strings the answer text MUST NOT contain (e.g., rounded forms).
    forbidden_substrings: list[str] = field(default_factory=list)


# Shared fact: the Q4 2022 ad-tier announcement, used by G5 and G6.
_FACT_2022_ADS = _f(
    fact_id="F-PROSE-nflx-q4-2022-letter-ads",
    fact_type=FactType.strategic_claim,
    claim="Netflix announced an ad-supported tier in its Q4 2022 shareholder letter.",
    asserter="Netflix, Inc.",
    source_document="nflx-q4-2022-letter",
    source_section="Strategy",
    verbatim_anchor="Today we announced an ad-supported plan called Basic with Ads.",
    period="2022Q4",
    assertion_date=date(2023, 1, 19),
)

# Shared fact: the 2018 no-ads strategic claim, used by G5 and G6.
_FACT_2018_NO_ADS = _f(
    fact_id="F-PROSE-nflx-q3-2018-letter-no-ads",
    fact_type=FactType.strategic_claim,
    claim="Netflix stated in Q3 2018 that it has no plans to introduce advertising on its service.",
    asserter="Netflix, Inc.",
    source_document="nflx-q3-2018-letter",
    source_section="Member experience",
    verbatim_anchor="We have no plans to introduce advertising on Netflix.",
    assertion_date=date(2018, 10, 16),
)


_TEST_CASES: list[TestCase] = [
    # ---- G1 — Simple clean Normal, single financial_metric -------------
    TestCase(
        case_id="G1_simple_revenue",
        description="Q2 2023 revenue, XBRL financial_metric, no refutation",
        query="What was Netflix's revenue for Q2 2023?",
        verified_facts=[
            _f(
                fact_id="F-XBRL-nflx-10q-2023-q2-Revenues-2023Q2",
                fact_type=FactType.financial_metric,
                claim="Netflix's revenue for Q2 2023 was $8,187 million.",
                asserter="Netflix, Inc.",
                source_document="nflx-10q-2023-q2",
                source_section="Condensed Consolidated Statements of Operations",
                verbatim_anchor="Revenues: $8,187,358 thousand.",
                period="2023Q2",
                value=8187.358,
                unit="USD millions",
                concept_tag="us-gaap:Revenues",
                assertion_date=date(2023, 7, 19),
                confidence=1.0,
            ),
        ],
        refutation_report=RefutationReport(
            run_id="g1",
            model_used="mistralai/mistral-large-2411",
            hypotheses=[
                _h(
                    hypothesis_id="h_1",
                    targets_claim_id="F-XBRL-nflx-10q-2023-q2-Revenues-2023Q2",
                    hypothesis_text="A later Netflix 10-K restated Q2 2023 revenue to a different value.",
                    rationale="If restated, the originally reported figure was wrong.",
                    strategy=RefutationStrategy.restated_value,
                    refutation_verdict=RefutationVerdict.unrefuted,
                    evidence_ids=[],
                ),
            ],
            overall_verdict=RefutationOverallVerdict.answer_strengthened,
        ),
        degradation_level=DegradationLevel.NORMAL,
        adversarially_probed=True,
        required_substrings=["$8,187"],
    ),

    # ---- G2 — Simple multi-fact Normal ---------------------------------
    TestCase(
        case_id="G2_multi_fact",
        description="Q4 2023 paid memberships + paid net adds, no refutation",
        query="What were Netflix's paid memberships and paid net adds at the end of Q4 2023?",
        verified_facts=[
            _f(
                fact_id="F-PROSE-nflx-q4-2023-letter-paid-members",
                fact_type=FactType.operational_metric,
                claim="Netflix reported 260.28 million paid memberships at the end of Q4 2023.",
                asserter="Netflix, Inc.",
                source_document="nflx-q4-2023-letter",
                source_section="Operating Metrics",
                verbatim_anchor="Global paid memberships ended Q4'23 at 260.28m.",
                period="2023Q4",
                value=260.28,
                unit="million",
                assertion_date=date(2024, 1, 23),
            ),
            _f(
                fact_id="F-PROSE-nflx-q4-2023-letter-paid-net-adds",
                fact_type=FactType.operational_metric,
                claim="Netflix added 13.12 million paid net adds in Q4 2023.",
                asserter="Netflix, Inc.",
                source_document="nflx-q4-2023-letter",
                source_section="Operating Metrics",
                verbatim_anchor="Q4'23 paid net additions of 13.12m.",
                period="2023Q4",
                value=13.12,
                unit="million",
                assertion_date=date(2024, 1, 23),
            ),
        ],
        refutation_report=None,
        degradation_level=DegradationLevel.NORMAL,
        adversarially_probed=False,
        required_substrings=["260.28", "13.12"],
    ),

    # ---- G3 — Standard multi-period comparison Normal -----------------
    TestCase(
        case_id="G3_compare",
        description="FY2019 vs FY2023 revenue, no refutation",
        query="Compare Netflix's revenue for FY2019 and FY2023.",
        verified_facts=[
            _f(
                fact_id="F-XBRL-nflx-10k-2019-Revenues-FY2019",
                fact_type=FactType.financial_metric,
                claim="Netflix's revenue for fiscal year 2019 was $20,156 million.",
                asserter="Netflix, Inc.",
                source_document="nflx-10k-2019",
                source_section="Item 8. Financial Statements",
                verbatim_anchor="Revenues: $20,156,447 thousand.",
                period="FY2019",
                value=20156.447,
                unit="USD millions",
                concept_tag="us-gaap:Revenues",
                assertion_date=date(2020, 1, 29),
                confidence=1.0,
            ),
            _f(
                fact_id="F-XBRL-nflx-10k-2023-Revenues-FY2023",
                fact_type=FactType.financial_metric,
                claim="Netflix's revenue for fiscal year 2023 was $33,723 million.",
                asserter="Netflix, Inc.",
                source_document="nflx-10k-2023",
                source_section="Item 8. Financial Statements",
                verbatim_anchor="Revenues: $33,723,297 thousand.",
                period="FY2023",
                value=33723.297,
                unit="USD millions",
                concept_tag="us-gaap:Revenues",
                assertion_date=date(2024, 1, 26),
                confidence=1.0,
            ),
        ],
        refutation_report=None,
        degradation_level=DegradationLevel.NORMAL,
        adversarially_probed=False,
        required_substrings=["$20,156", "$33,723"],
    ),

    # ---- G4 — Standard with weakly_refuted hypothesis ------------------
    TestCase(
        case_id="G4_weak_refutation",
        description="Q3 2023 revenue causal (price increases) + weak alternative_cause",
        query="What drove Netflix's revenue growth in Q3 2023?",
        verified_facts=[
            _f(
                fact_id="F-PROSE-nflx-q3-2023-letter-price-driver",
                fact_type=FactType.causal_explanation,
                claim="Netflix attributed Q3 2023 revenue growth primarily to price increases on certain plans.",
                asserter="Netflix, Inc.",
                source_document="nflx-q3-2023-letter",
                source_section="Financial overview",
                verbatim_anchor="Revenue growth this quarter was driven primarily by recent price changes.",
                period="2023Q3",
                assertion_date=date(2023, 10, 18),
            ),
            _f(
                fact_id="F-PROSE-nflx-q4-2023-letter-sharing-driver",
                fact_type=FactType.causal_explanation,
                claim="Netflix's Q4 2023 letter noted that paid sharing enforcement was a meaningful revenue driver in late 2023.",
                asserter="Netflix, Inc.",
                source_document="nflx-q4-2023-letter",
                source_section="Financial overview",
                verbatim_anchor="Paid sharing enforcement materially contributed to revenue growth.",
                period="2023Q4",
                assertion_date=date(2024, 1, 23),
            ),
        ],
        refutation_report=RefutationReport(
            run_id="g4",
            model_used="mistralai/mistral-large-2411",
            hypotheses=[
                _h(
                    hypothesis_id="h_1",
                    targets_claim_id="F-PROSE-nflx-q3-2023-letter-price-driver",
                    hypothesis_text="Netflix's Q4 2023 letter attributes revenue growth to paid sharing enforcement, not price increases.",
                    rationale="A different cause for the same outcome would weaken the original claim.",
                    strategy=RefutationStrategy.alternative_cause,
                    refutation_verdict=RefutationVerdict.weakly_refuted,
                    evidence_ids=["F-PROSE-nflx-q4-2023-letter-sharing-driver"],
                ),
            ],
            overall_verdict=RefutationOverallVerdict.answer_strengthened,
        ),
        degradation_level=DegradationLevel.NORMAL,
        adversarially_probed=True,
    ),

    # ---- G5 — Strong refutation, Partial, structured disagreement ------
    TestCase(
        case_id="G5_strong_refutation_partial",
        description="2018 no-ads vs Q4 2022 ad-tier; loop did not resolve → Partial",
        query="Did Netflix ever say it had no plans to introduce advertising?",
        verified_facts=[_FACT_2018_NO_ADS],
        refutation_report=RefutationReport(
            run_id="g5",
            model_used="mistralai/mistral-large-2411",
            hypotheses=[
                _h(
                    hypothesis_id="h_1",
                    targets_claim_id=_FACT_2018_NO_ADS.fact_id,
                    hypothesis_text="Netflix's Q4 2022 shareholder letter announced an ad-supported tier, reversing its earlier no-ads position.",
                    rationale="If Netflix introduced ads, it contradicts the earlier no-ads claim.",
                    strategy=RefutationStrategy.later_reversal,
                    refutation_verdict=RefutationVerdict.strongly_refuted,
                    evidence_ids=[_FACT_2022_ADS.fact_id],
                ),
            ],
            overall_verdict=RefutationOverallVerdict.refutation_to_partial,
        ),
        degradation_level=DegradationLevel.PARTIAL,
        adversarially_probed=True,
        required_dates_in_text=[_FACT_2018_NO_ADS.fact_id, _FACT_2022_ADS.fact_id],
    ),

    # ---- G6 — Strong refutation resolved to Normal-with-temporal-evo ---
    TestCase(
        case_id="G6_resolved_temporal_evolution",
        description="Loop resolved: verified set now contains BOTH 2018 + 2022 facts",
        query="What has Netflix said about advertising over time?",
        verified_facts=[_FACT_2018_NO_ADS, _FACT_2022_ADS],
        refutation_report=RefutationReport(
            run_id="g6",
            model_used="mistralai/mistral-large-2411",
            hypotheses=[
                _h(
                    hypothesis_id="h_1",
                    targets_claim_id=_FACT_2018_NO_ADS.fact_id,
                    hypothesis_text="Netflix's Q4 2022 shareholder letter announced an ad-supported tier, reversing its earlier no-ads position.",
                    rationale="If Netflix introduced ads, it contradicts the earlier no-ads claim.",
                    strategy=RefutationStrategy.later_reversal,
                    refutation_verdict=RefutationVerdict.strongly_refuted,
                    evidence_ids=[_FACT_2022_ADS.fact_id],
                ),
            ],
            overall_verdict=RefutationOverallVerdict.refutation_to_loop,
            triggered_loop_reentry=True,
            loop_reentry_iteration=1,
        ),
        degradation_level=DegradationLevel.NORMAL,
        adversarially_probed=True,
        required_dates_in_text=[_FACT_2018_NO_ADS.fact_id, _FACT_2022_ADS.fact_id],
    ),

    # ---- G7 — Accounting policy, no refutation ------------------------
    TestCase(
        case_id="G7_accounting_policy",
        description="Content amortization policy, FY2017 10-K",
        query="What is Netflix's policy for amortizing content assets?",
        verified_facts=[
            _f(
                fact_id="F-PROSE-nflx-10k-2017-amort-policy",
                fact_type=FactType.accounting_policy,
                claim="Netflix amortizes content assets on an accelerated basis over the shorter of each title's contractual window or estimated useful life.",
                asserter="Netflix, Inc.",
                source_document="nflx-10k-2017",
                source_section="Note 2 — Summary of Significant Accounting Policies",
                verbatim_anchor="We amortize licensed content assets on an accelerated basis over the shorter of each title's contractual window of availability or estimated useful life.",
                period="FY2017",
                assertion_date=date(2018, 1, 29),
            ),
        ],
        refutation_report=None,
        degradation_level=DegradationLevel.NORMAL,
        adversarially_probed=False,
    ),

    # ---- G8 — Hostile numerical-paraphrase test -----------------------
    TestCase(
        case_id="G8_hostile_paraphrase",
        description="Q2 2022 revenue — query explicitly asks for a 'rounded, audience-friendly' phrasing",
        query=(
            "What was Netflix's Q2 2022 revenue? Phrase it in a "
            "friendly, audience-accessible way for a general reader."
        ),
        verified_facts=[
            _f(
                fact_id="F-XBRL-nflx-10q-2022-q2-Revenues-2022Q2",
                fact_type=FactType.financial_metric,
                claim="Netflix's revenue for Q2 2022 was $7,970 million.",
                asserter="Netflix, Inc.",
                source_document="nflx-10q-2022-q2",
                source_section="Condensed Consolidated Statements of Operations",
                verbatim_anchor="Revenues: $7,970,141 thousand.",
                period="2022Q2",
                value=7970.141,
                unit="USD millions",
                concept_tag="us-gaap:Revenues",
                assertion_date=date(2022, 7, 19),
                confidence=1.0,
            ),
        ],
        refutation_report=None,
        degradation_level=DegradationLevel.NORMAL,
        adversarially_probed=False,
        required_substrings=["$7,970"],
        forbidden_substrings=[
            "$8 billion", "$8B", "about $8", "roughly $8", "nearly $8",
            "around $8", "$8.0 billion", "$8.0B",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Heuristic graders
# ---------------------------------------------------------------------------


_NUM_TOKEN_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+\.\d+\b|\b\d+\b")


def _fact_value_strings(fact: FactRecord) -> list[str]:
    """Generate the verbatim representations of a fact's value that
    should appear in answer_text. Returns multiple acceptable forms:
    the comma-grouped integer ("8,187"), the bare integer if no
    decimal in value ("8187"), and a few common renderings of value+
    unit. The grader passes if ANY of these forms appears."""
    if fact.value is None:
        return []
    val = fact.value
    out: list[str] = []
    # Match the fact's own claim text — if it contains "$8,187" the
    # answer should too. Pull number-like tokens from the claim.
    tokens = _NUM_TOKEN_RE.findall(fact.claim)
    for t in tokens:
        # Skip raw period numbers like "2023" or "2022" that
        # appear in the claim's date phrasing.
        if re.fullmatch(r"(?:19|20)\d{2}", t):
            continue
        out.append(t)
    # Also accept the numeric value rendered as comma-grouped int and
    # as the float itself. e.g., value=8187.358 → "8,187" "8187"
    # "8,187.358" "8187.358".
    int_part = int(val)
    out.append(f"{int_part:,}")
    out.append(str(int_part))
    if val != int_part:
        out.append(f"{int_part:,}.{str(val).split('.')[-1]}")
        out.append(str(val))
    return out


def _has_numerical_fidelity(answer_text: str, fact: FactRecord) -> bool:
    """True iff at least one verbatim representation of the fact's
    value appears in answer_text. Returns True trivially for facts
    without a numerical value."""
    candidates = _fact_value_strings(fact)
    if not candidates:
        return True
    for c in candidates:
        # Token-boundary match to avoid '187' matching inside '2018'.
        # Use re.escape to handle the comma and dollar-sign chars
        # cleanly. Look for token-boundary on both sides.
        pat = r"(?<![\d.])" + re.escape(c) + r"(?![\d])"
        if re.search(pat, answer_text):
            return True
    return False


def _date_token_variants(d: date) -> list[str]:
    """Return acceptable string renderings of an assertion_date."""
    iso = d.isoformat()
    year = str(d.year)
    return [
        iso,
        f"{d.year}-{d.month:02d}",
        year,
        d.strftime("%B %d, %Y"),
        d.strftime("%B %Y"),
        f"Q{(d.month - 1) // 3 + 1} {d.year}",
        f"Q{(d.month - 1) // 3 + 1} {year[-2:]}",
    ]


def _mentions_date(answer_text: str, d: date) -> bool:
    variants = _date_token_variants(d)
    for v in variants:
        if v in answer_text:
            return True
    return False


@dataclass
class CaseGrade:
    case_id: str
    passed: bool
    elapsed_s: float
    schema_valid: bool
    no_fabricated_ids: bool
    numerical_fidelity: bool
    disclosed_refutations_correct: bool
    refutation_dates_in_prose: bool
    adversarially_probed_correct: bool
    required_substrings_present: bool
    forbidden_substrings_absent: bool
    notes: list[str] = field(default_factory=list)
    answer: AnswerSchema | None = None
    error: str | None = None


def _grade(case: TestCase, answer: AnswerSchema, elapsed_s: float) -> CaseGrade:
    notes: list[str] = []

    # S1: AnswerSchema is Pydantic-valid by construction (generate_answer
    # returns a typed object). Round-trip via model_dump to be safe.
    try:
        AnswerSchema.model_validate(answer.model_dump())
        schema_valid = True
    except Exception as e:
        schema_valid = False
        notes.append(f"schema round-trip failed: {e}")

    # S2: no fabricated fact_ids (enforced inside generate_answer; we
    # confirm here that every cited source_id resolves to a verified fact).
    valid_ids = {f.fact_id for f in case.verified_facts}
    fabricated: list[str] = []
    for c in answer.claims:
        for sid in c.source_ids:
            if sid not in valid_ids:
                fabricated.append(sid)
    no_fabricated = not fabricated
    if not no_fabricated:
        notes.append(f"fabricated fact_ids cited: {fabricated[:3]}")

    # Q1: numerical fidelity — every verified fact with a value must
    # have a verbatim form in answer_text. (Facts without a value
    # are skipped trivially.)
    bad_facts = [
        f.fact_id for f in case.verified_facts
        if not _has_numerical_fidelity(answer.answer_text, f)
    ]
    numerical_fidelity = not bad_facts
    if bad_facts:
        notes.append(
            f"numerical fidelity violated for fact(s) {bad_facts[:3]} — "
            f"value strings missing from answer_text"
        )

    # Q2: disclosed_refutations[] matches the refutation_report (built
    # mechanically by code — should be deterministically correct).
    expected_disclosures = 0
    if case.refutation_report is not None:
        expected_disclosures = sum(
            1 for h in case.refutation_report.hypotheses
            if h.refutation_verdict != RefutationVerdict.unrefuted
        )
    disclosed_refutations_correct = (
        len(answer.disclosed_refutations) == expected_disclosures
    )
    if not disclosed_refutations_correct:
        notes.append(
            f"disclosed_refutations count mismatch: "
            f"expected {expected_disclosures}, got "
            f"{len(answer.disclosed_refutations)}"
        )

    # Q3: refutation disclosure in prose — for strongly_refuted
    # hypotheses, both the targeted fact's assertion_date and the
    # refuting evidence's assertion_date must appear in answer_text.
    fact_by_id = {f.fact_id: f for f in case.verified_facts}
    refutation_dates_in_prose = True
    if case.refutation_report is not None:
        for h in case.refutation_report.hypotheses:
            if h.refutation_verdict != RefutationVerdict.strongly_refuted:
                continue
            targeted = fact_by_id.get(h.targets_claim_id)
            if targeted is not None and not _mentions_date(answer.answer_text, targeted.assertion_date):
                refutation_dates_in_prose = False
                notes.append(
                    f"strong refutation missing targeted assertion_date "
                    f"{targeted.assertion_date} in prose"
                )
            for eid in h.evidence_ids:
                # Refuting evidence might be in the verified set (if
                # the loop resolved) or not (Partial outcome).
                refuting = fact_by_id.get(eid)
                if refuting is None:
                    # G5 path — refuting fact is on the hypothesis
                    # but NOT in the verified set. The case provides
                    # us the refuting fact via the shared constants
                    # at module level. Look it up.
                    refuting = _SHARED_REFUTING_FACTS.get(eid)
                if refuting is None:
                    continue
                if not _mentions_date(answer.answer_text, refuting.assertion_date):
                    refutation_dates_in_prose = False
                    notes.append(
                        f"strong refutation missing refuting fact's "
                        f"assertion_date {refuting.assertion_date} in prose"
                    )

    # Q4: adversarially_probed propagation.
    adversarially_probed_correct = (
        answer.adversarially_probed == case.adversarially_probed
    )
    if not adversarially_probed_correct:
        notes.append(
            f"adversarially_probed mismatch: "
            f"expected {case.adversarially_probed}, got {answer.adversarially_probed}"
        )

    # Required substrings.
    required_substrings_present = True
    for s in case.required_substrings:
        if s not in answer.answer_text:
            required_substrings_present = False
            notes.append(f"required substring missing: {s!r}")

    # Forbidden substrings.
    forbidden_substrings_absent = True
    for s in case.forbidden_substrings:
        if s.lower() in answer.answer_text.lower():
            forbidden_substrings_absent = False
            notes.append(f"forbidden substring present: {s!r}")

    passed = all([
        schema_valid,
        no_fabricated,
        numerical_fidelity,
        disclosed_refutations_correct,
        refutation_dates_in_prose,
        adversarially_probed_correct,
        required_substrings_present,
        forbidden_substrings_absent,
    ])

    return CaseGrade(
        case_id=case.case_id,
        passed=passed,
        elapsed_s=elapsed_s,
        schema_valid=schema_valid,
        no_fabricated_ids=no_fabricated,
        numerical_fidelity=numerical_fidelity,
        disclosed_refutations_correct=disclosed_refutations_correct,
        refutation_dates_in_prose=refutation_dates_in_prose,
        adversarially_probed_correct=adversarially_probed_correct,
        required_substrings_present=required_substrings_present,
        forbidden_substrings_absent=forbidden_substrings_absent,
        notes=notes,
        answer=answer,
    )


# Refuting facts referenced by hypothesis evidence_ids but NOT in the
# verified set (used by G5's Partial path).
_SHARED_REFUTING_FACTS: dict[str, FactRecord] = {
    _FACT_2022_ADS.fact_id: _FACT_2022_ADS,
}


# ---------------------------------------------------------------------------
# Per-case runner + reporting
# ---------------------------------------------------------------------------


def _run_case(case: TestCase) -> CaseGrade:
    t0 = time.time()
    try:
        answer, _resp = generate_answer(
            original_query=case.query,
            verified_facts=case.verified_facts,
            refutation_report=case.refutation_report,
            degradation_level=case.degradation_level,
            adversarially_probed=case.adversarially_probed,
        )
    except Exception as e:
        return CaseGrade(
            case_id=case.case_id,
            passed=False,
            elapsed_s=round(time.time() - t0, 2),
            schema_valid=False,
            no_fabricated_ids=False,
            numerical_fidelity=False,
            disclosed_refutations_correct=False,
            refutation_dates_in_prose=False,
            adversarially_probed_correct=False,
            required_substrings_present=False,
            forbidden_substrings_absent=False,
            error=f"{type(e).__name__}: {e}",
        )
    elapsed = round(time.time() - t0, 2)
    return _grade(case, answer, elapsed)


def _print_grade(case: TestCase, g: CaseGrade) -> None:
    marker = "PASS" if g.passed else "FAIL"
    print()
    print("=" * 78)
    print(f"[{marker}] {g.case_id} — {case.description}")
    print(f"  elapsed: {g.elapsed_s}s")
    if g.error:
        print(f"  ERROR: {g.error}")
        return
    flags = [
        ("schema", g.schema_valid),
        ("no-fab-ids", g.no_fabricated_ids),
        ("num-fidelity", g.numerical_fidelity),
        ("ref-disclosure", g.disclosed_refutations_correct),
        ("ref-dates", g.refutation_dates_in_prose),
        ("adv-probed", g.adversarially_probed_correct),
        ("required", g.required_substrings_present),
        ("forbidden", g.forbidden_substrings_absent),
    ]
    flag_strs = [f"{label}={'OK' if ok else 'FAIL'}" for label, ok in flags]
    print("  flags: " + " | ".join(flag_strs))
    if g.notes:
        for n in g.notes:
            print(f"    · {n}")
    if g.answer is not None:
        text = g.answer.answer_text
        if len(text) > 400:
            text = text[:400] + "…"
        print(f"  answer_text: {text}")
        print(f"  claims: {len(g.answer.claims)}, "
              f"disclosed_refutations: {len(g.answer.disclosed_refutations)}")


def main() -> None:
    print(f"Generator model: {GENERATOR_MODEL}")
    print(f"Test cases:      {len(_TEST_CASES)}\n")

    grades: list[CaseGrade] = []
    for case in _TEST_CASES:
        g = _run_case(case)
        grades.append(g)
        _print_grade(case, g)

    passed = sum(1 for g in grades if g.passed)
    total = len(grades)
    print()
    print("=" * 78)
    print(f"OVERALL: {passed}/{total} cases passed (bar: 6/8)")
    for g in grades:
        marker = "OK  " if g.passed else "FAIL"
        print(f"  {marker}  {g.case_id}")

    summary = {
        "model": GENERATOR_MODEL,
        "total": total,
        "passed": passed,
        "cases": [
            {
                "case_id": g.case_id,
                "passed": g.passed,
                "elapsed_s": g.elapsed_s,
                "error": g.error,
                "flags": {
                    "schema_valid": g.schema_valid,
                    "no_fabricated_ids": g.no_fabricated_ids,
                    "numerical_fidelity": g.numerical_fidelity,
                    "disclosed_refutations_correct": g.disclosed_refutations_correct,
                    "refutation_dates_in_prose": g.refutation_dates_in_prose,
                    "adversarially_probed_correct": g.adversarially_probed_correct,
                    "required_substrings_present": g.required_substrings_present,
                    "forbidden_substrings_absent": g.forbidden_substrings_absent,
                },
                "notes": g.notes,
                "answer": g.answer.model_dump(mode="json") if g.answer else None,
            }
            for g in grades
        ],
    }
    out_path = Path("data/logs/block13_generator_harness.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nSummary -> {out_path}")

    if passed < 6:
        sys.exit(1)


if __name__ == "__main__":
    main()

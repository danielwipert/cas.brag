"""Block 10 test harness: drive the Refutation Agent's hypothesis-
generation prompt over 10 hand-built verified evidence sets, one per
refutation strategy plus three cross-cutting cases, and grade the
output against the spec's quality bar.

Pass criteria per case (build plan v3 §Block 10):

  S1. Strategy matches the targeted fact's fact_type per the table.
      (Enforced inside the prompt's validator, so a case can only
      reach the grader with this satisfied.)
  S2. Hypothesis count matches the complexity tier
      (1 Simple, 2 Standard, 3 Complex). Also enforced upstream.
  Q1. CONCRETE — hypothesis text mentions enough specifics that
      retrieval could find/not-find it. Heuristic: mentions a
      document type or year/quarter token AND is reasonably long.
  Q2. MEANINGFULLY DIFFERENT — hypothesis text is not a paraphrase
      of the targeted claim. Heuristic: token Jaccard overlap below
      a threshold.
  Q3. NETFLIX-ASSERTED — hypothesis would be representable in a
      Netflix-issued document. Heuristic: subject is "Netflix" or
      a Netflix executive name token, not a third party.

The overall pass bar is 8/10 cases passing.

Run from repo root::

    python -m tests.refutation_prompt_harness

Requires OPENROUTER_API_KEY in env or .env.
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import date

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from agents.refutation.prompt import (
    REFUTATION_MODEL,
    generate_hypotheses,
)
from schemas.enums import ComplexityTier, FactType, RefutationStrategy
from schemas.records import FactRecord, RefutationHypothesis


# ---------------------------------------------------------------------------
# Hand-built verified evidence sets
# ---------------------------------------------------------------------------


def _f(  # short ctor for readability in the test table
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
    confidence: float = 0.92,
) -> FactRecord:
    return FactRecord(
        fact_id=fact_id,
        claim=claim,
        asserter=asserter,
        source_document=source_document,
        source_section=source_section,
        verbatim_anchor=verbatim_anchor,
        fact_type=fact_type,
        period=period,
        value=value,
        unit=unit,
        concept_tag=concept_tag,
        assertion_date=assertion_date,
        confidence=confidence,
    )


@dataclass(frozen=True)
class TestCase:
    case_id: str
    description: str
    query: str
    tier: ComplexityTier
    verified_facts: list[FactRecord]
    expected_strategies: list[RefutationStrategy]  # one per hypothesis (order-insensitive)


_TEST_CASES: list[TestCase] = [
    # ---- Single-strategy cases, one per fact_type ------------------------
    TestCase(
        case_id="T1_financial_metric_restated",
        description="financial_metric → restated_value",
        query="What were Netflix's paid memberships at the end of Q1 2022?",
        tier=ComplexityTier.simple,
        verified_facts=[
            _f(
                fact_id="nflx-10q-2022q1::paid-members",
                fact_type=FactType.financial_metric,
                claim="Netflix reported 221.64 million paid memberships at the end of Q1 2022.",
                asserter="Netflix, Inc.",
                source_document="nflx-10q-2022-q1",
                source_section="Operating Highlights",
                verbatim_anchor="Paid memberships at the end of the quarter were 221.64 million.",
                period="2022Q1",
                value=221.64,
                unit="million",
                concept_tag="nflx:PaidMemberships",
                assertion_date=date(2022, 4, 20),
            ),
        ],
        expected_strategies=[RefutationStrategy.restated_value],
    ),
    TestCase(
        case_id="T2_forward_guidance",
        description="forward_guidance → guidance_vs_actual",
        query="What did Netflix project for Q2 2022 paid net additions?",
        tier=ComplexityTier.simple,
        verified_facts=[
            _f(
                fact_id="nflx-q1-2022-letter::guide-q2-pna",
                fact_type=FactType.forward_guidance,
                claim="Netflix forecast paid net additions of approximately +2.5 million for Q2 2022.",
                asserter="Netflix, Inc.",
                source_document="nflx-q1-2022-letter",
                source_section="Outlook",
                verbatim_anchor="We forecast paid net adds of +2.5m for Q2'22.",
                assertion_date=date(2022, 4, 19),
            ),
        ],
        expected_strategies=[RefutationStrategy.guidance_vs_actual],
    ),
    TestCase(
        case_id="T3_strategic_claim_no_ads",
        description="strategic_claim → later_reversal",
        query="Did Netflix have plans to introduce advertising in 2018?",
        tier=ComplexityTier.simple,
        verified_facts=[
            _f(
                fact_id="nflx-q3-2018-letter::no-ads",
                fact_type=FactType.strategic_claim,
                claim="Netflix stated it has no plans to add advertising to its service.",
                asserter="Netflix, Inc.",
                source_document="nflx-q3-2018-letter",
                source_section="Member experience",
                verbatim_anchor="We have no plans to introduce advertising on Netflix.",
                assertion_date=date(2018, 10, 16),
            ),
        ],
        expected_strategies=[RefutationStrategy.later_reversal],
    ),
    TestCase(
        case_id="T4_causal_price",
        description="causal_explanation → alternative_cause",
        query="What drove Netflix's revenue growth in Q3 2023?",
        tier=ComplexityTier.simple,
        verified_facts=[
            _f(
                fact_id="nflx-q3-2023-letter::price-driver",
                fact_type=FactType.causal_explanation,
                claim="Netflix attributed Q3 2023 revenue growth primarily to price increases on certain plans.",
                asserter="Netflix, Inc.",
                source_document="nflx-q3-2023-letter",
                source_section="Financial overview",
                verbatim_anchor="Revenue growth this quarter was driven primarily by recent price changes.",
                period="2023Q3",
                assertion_date=date(2023, 10, 18),
            ),
        ],
        expected_strategies=[RefutationStrategy.alternative_cause],
    ),
    TestCase(
        case_id="T5_risk_password_sharing",
        description="risk_disclosure → materialization",
        query="How does Netflix describe the risk of password sharing in its risk factors?",
        tier=ComplexityTier.simple,
        verified_facts=[
            _f(
                fact_id="nflx-10k-2021::risk-sharing",
                fact_type=FactType.risk_disclosure,
                claim="Netflix's 2021 10-K identifies account password sharing as a risk that may negatively impact membership growth.",
                asserter="Netflix, Inc.",
                source_document="nflx-10k-2021",
                source_section="Item 1A. Risk Factors",
                verbatim_anchor="Account sharing without authorization may negatively impact our ability to grow paid memberships.",
                period="FY2021",
                assertion_date=date(2022, 1, 27),
            ),
        ],
        expected_strategies=[RefutationStrategy.materialization],
    ),
    TestCase(
        case_id="T6_accounting_amortization",
        description="accounting_policy → policy_change",
        query="What is Netflix's policy for amortizing content assets?",
        tier=ComplexityTier.simple,
        verified_facts=[
            _f(
                fact_id="nflx-10k-2017::content-amort",
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
        expected_strategies=[RefutationStrategy.policy_change],
    ),
    TestCase(
        case_id="T7_operational_paid_members",
        description="operational_metric → revised_value",
        query="What were Netflix's paid memberships entering 2022?",
        tier=ComplexityTier.simple,
        verified_facts=[
            _f(
                fact_id="nflx-q4-2021-letter::eop-members",
                fact_type=FactType.operational_metric,
                claim="Netflix reported global paid memberships of 221.84 million at year-end 2021 in its Q4 2021 shareholder letter.",
                asserter="Netflix, Inc.",
                source_document="nflx-q4-2021-letter",
                source_section="Operating Metrics",
                verbatim_anchor="Global paid memberships ended 2021 at 221.84m.",
                period="2021Q4",
                value=221.84,
                unit="million",
                assertion_date=date(2022, 1, 20),
            ),
        ],
        expected_strategies=[RefutationStrategy.revised_value],
    ),
    # ---- Cross-cutting cases ---------------------------------------------
    TestCase(
        case_id="T8_cross_standard_metric_and_strategic",
        description="Standard cross-cutting: financial_metric + strategic_claim",
        query="How has Netflix's stance on advertising evolved given its revenue trajectory?",
        tier=ComplexityTier.standard,
        verified_facts=[
            _f(
                fact_id="nflx-10k-2021::revenue-fy21",
                fact_type=FactType.financial_metric,
                claim="Netflix reported fiscal 2021 revenues of $29.70 billion.",
                asserter="Netflix, Inc.",
                source_document="nflx-10k-2021",
                source_section="Item 8. Financial Statements",
                verbatim_anchor="Revenues for the year ended December 31, 2021 were $29,697,844 thousand.",
                period="FY2021",
                value=29697.844,
                unit="USD millions",
                concept_tag="us-gaap:Revenues",
                assertion_date=date(2022, 1, 27),
            ),
            _f(
                fact_id="nflx-q3-2018-letter::no-ads-stance",
                fact_type=FactType.strategic_claim,
                claim="Netflix stated in Q3 2018 that it had no plans to add advertising to its service.",
                asserter="Netflix, Inc.",
                source_document="nflx-q3-2018-letter",
                source_section="Strategy",
                verbatim_anchor="We have no plans to introduce advertising on Netflix.",
                assertion_date=date(2018, 10, 16),
            ),
        ],
        expected_strategies=[
            RefutationStrategy.restated_value,
            RefutationStrategy.later_reversal,
        ],
    ),
    TestCase(
        case_id="T9_cross_standard_guidance_and_causal",
        description="Standard cross-cutting: forward_guidance + causal_explanation",
        query="What did Netflix expect for Q2 2022, and what drove the actual outcome?",
        tier=ComplexityTier.standard,
        verified_facts=[
            _f(
                fact_id="nflx-q1-2022-letter::q2-revenue-guide",
                fact_type=FactType.forward_guidance,
                claim="Netflix projected Q2 2022 revenue of approximately $8.05 billion.",
                asserter="Netflix, Inc.",
                source_document="nflx-q1-2022-letter",
                source_section="Outlook",
                verbatim_anchor="We forecast Q2'22 revenue of $8,053m.",
                value=8053.0,
                unit="USD millions",
                assertion_date=date(2022, 4, 19),
            ),
            _f(
                fact_id="nflx-q1-2022-letter::why-losses",
                fact_type=FactType.causal_explanation,
                claim="Netflix attributed Q1 2022 subscriber losses primarily to factors including account sharing and increased competition.",
                asserter="Netflix, Inc.",
                source_document="nflx-q1-2022-letter",
                source_section="Operating overview",
                verbatim_anchor="Our revenue growth has slowed considerably... account sharing... competition.",
                period="2022Q1",
                assertion_date=date(2022, 4, 19),
            ),
        ],
        expected_strategies=[
            RefutationStrategy.guidance_vs_actual,
            RefutationStrategy.alternative_cause,
        ],
    ),
    TestCase(
        case_id="T10_cross_complex_three_types",
        description="Complex cross-cutting: risk_disclosure + strategic_claim + financial_metric",
        query="How consistent has Netflix's position on advertising and password sharing been with its results?",
        tier=ComplexityTier.complex,
        verified_facts=[
            _f(
                fact_id="nflx-10k-2020::risk-sharing-2020",
                fact_type=FactType.risk_disclosure,
                claim="Netflix's 2020 10-K disclosed that unauthorized account sharing could adversely affect its business.",
                asserter="Netflix, Inc.",
                source_document="nflx-10k-2020",
                source_section="Item 1A. Risk Factors",
                verbatim_anchor="Unauthorized sharing of accounts could adversely affect our business.",
                period="FY2020",
                assertion_date=date(2021, 1, 28),
            ),
            _f(
                fact_id="nflx-q4-2019-letter::no-ads-2019",
                fact_type=FactType.strategic_claim,
                claim="In Q4 2019, Netflix reiterated that it had no plans to introduce ads on its service.",
                asserter="Netflix, Inc.",
                source_document="nflx-q4-2019-letter",
                source_section="Member experience",
                verbatim_anchor="We continue to have no plans to introduce advertising on Netflix.",
                assertion_date=date(2020, 1, 21),
            ),
            _f(
                fact_id="nflx-10k-2022::revenue-fy22",
                fact_type=FactType.financial_metric,
                claim="Netflix reported fiscal 2022 revenues of $31.62 billion.",
                asserter="Netflix, Inc.",
                source_document="nflx-10k-2022",
                source_section="Item 8. Financial Statements",
                verbatim_anchor="Revenues for the year ended December 31, 2022 were $31,615,550 thousand.",
                period="FY2022",
                value=31615.550,
                unit="USD millions",
                concept_tag="us-gaap:Revenues",
                assertion_date=date(2023, 1, 26),
            ),
        ],
        expected_strategies=[
            RefutationStrategy.materialization,
            RefutationStrategy.later_reversal,
            RefutationStrategy.restated_value,
        ],
    ),
]


# ---------------------------------------------------------------------------
# Heuristic graders for Q1 / Q2 / Q3
# ---------------------------------------------------------------------------


_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_QUARTER_RE = re.compile(r"\b[Qq][1-4]\b|\b(?:first|second|third|fourth)\s+quarter\b", re.IGNORECASE)
_DOC_HINT_RE = re.compile(
    r"\b(?:10-?K|10-?Q|shareholder\s+letter|"
    r"earnings\s+(?:letter|transcript|call|report|release)|"
    r"transcript|filing|press\s+release|annual\s+report|"
    r"quarterly\s+report)\b",
    re.IGNORECASE,
)
_CONTRAST_RE = re.compile(
    r"\b(?:rather\s+than|instead\s+of|reversing|reversed|contradicting|"
    r"contradicts|restated|revised|reverses)\b",
    re.IGNORECASE,
)
_NETFLIX_HINT_RE = re.compile(
    r"\b(?:Netflix|Reed\s+Hastings|Ted\s+Sarandos|Greg\s+Peters|Spencer\s+Neumann|"
    r"the\s+company|management|the\s+letter)\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z]{4,}")


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


@dataclass
class GradeResult:
    concrete: bool
    different: bool
    asserted: bool
    word_count: int
    jaccard: float
    notes: list[str]

    @property
    def passed(self) -> bool:
        return self.concrete and self.different and self.asserted


def _grade_hypothesis(
    h: RefutationHypothesis, targeted_fact: FactRecord
) -> GradeResult:
    notes: list[str] = []
    text = h.hypothesis_text
    wc = sum(1 for _ in _WORD_RE.finditer(text))

    has_year = bool(_YEAR_RE.search(text))
    has_quarter = bool(_QUARTER_RE.search(text))
    has_doc_hint = bool(_DOC_HINT_RE.search(text))
    # CONCRETE: at least one time anchor AND (a document hint OR a fairly
    # long, content-bearing sentence). Length floor at 14 words keeps
    # one-line stub hypotheses ("Netflix later changed its mind") out.
    concrete = (has_year or has_quarter) and (has_doc_hint or wc >= 14)
    if not concrete:
        if not (has_year or has_quarter):
            notes.append("missing year/quarter anchor")
        if not has_doc_hint and wc < 14:
            notes.append(f"no document hint and only {wc} words")

    # MEANINGFULLY DIFFERENT: token Jaccard between hypothesis and the
    # targeted claim is below 0.55. Higher overlap is usually a
    # paraphrase, but hypotheses that explicitly contrast positions
    # ("rather than X", "reversing Y", "restated to Z") legitimately
    # reuse the verified claim's vocabulary while inverting its
    # substance — give those a slightly higher cap (0.65).
    jacc = _jaccard(_tokens(text), _tokens(targeted_fact.claim))
    cap = 0.65 if _CONTRAST_RE.search(text) else 0.55
    different = jacc < cap
    if not different:
        notes.append(f"high token overlap with verified claim ({jacc:.2f}, cap={cap:.2f})")

    # NETFLIX-ASSERTED: the hypothesis must read as a Netflix-issued
    # statement — Netflix as subject or a Netflix executive named.
    asserted = bool(_NETFLIX_HINT_RE.search(text))
    if not asserted:
        notes.append("hypothesis subject is not Netflix / an exec / the company")

    return GradeResult(
        concrete=concrete,
        different=different,
        asserted=asserted,
        word_count=wc,
        jaccard=jacc,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Per-case runner
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    case_id: str
    description: str
    passed: bool
    elapsed_s: float
    hypotheses: list[RefutationHypothesis]
    grades: list[GradeResult]
    error: str | None = None


def _run_case(case: TestCase) -> CaseResult:
    fact_by_id = {f.fact_id: f for f in case.verified_facts}
    t0 = time.time()
    try:
        hypotheses, _resp = generate_hypotheses(
            case.query, case.tier, case.verified_facts
        )
    except Exception as e:
        return CaseResult(
            case_id=case.case_id,
            description=case.description,
            passed=False,
            elapsed_s=round(time.time() - t0, 2),
            hypotheses=[],
            grades=[],
            error=f"{type(e).__name__}: {e}",
        )
    elapsed = round(time.time() - t0, 2)

    grades = [
        _grade_hypothesis(h, fact_by_id[h.targets_claim_id]) for h in hypotheses
    ]

    # Strategy correctness (structural; should always hold since the
    # prompt validator enforces it, but assert defensively).
    strat_ok = True
    for h in hypotheses:
        expected = case.expected_strategies
        if h.strategy not in expected:
            strat_ok = False
            break

    # Case passes iff every hypothesis passes the heuristic grader AND
    # strategy choice is among the expected (i.e., one of the verified
    # facts that this hypothesis is allowed to target).
    case_passed = strat_ok and all(g.passed for g in grades)

    return CaseResult(
        case_id=case.case_id,
        description=case.description,
        passed=case_passed,
        elapsed_s=elapsed,
        hypotheses=hypotheses,
        grades=grades,
        error=None if strat_ok else "strategy not in expected set",
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_case(result: CaseResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"\n{'=' * 78}\n[{status}] {result.case_id} — {result.description}")
    print(f"  elapsed: {result.elapsed_s}s")
    if result.error and not result.hypotheses:
        print(f"  ERROR: {result.error}")
        return
    for i, (h, g) in enumerate(zip(result.hypotheses, result.grades), start=1):
        print(f"  h_{i}: strategy={h.strategy.value}  targets={h.targets_claim_id}")
        print(f"    text: {h.hypothesis_text}")
        print(f"    rationale: {h.rationale}")
        flags = []
        flags.append("CONCRETE" if g.concrete else "not-concrete")
        flags.append("DIFFERENT" if g.different else "not-different")
        flags.append("ASSERTED" if g.asserted else "not-asserted")
        print(
            f"    grade: {' | '.join(flags)}  "
            f"(words={g.word_count}, jaccard={g.jaccard:.2f})"
        )
        if g.notes:
            for n in g.notes:
                print(f"      · {n}")


def main() -> None:
    print(f"Refutation model: {REFUTATION_MODEL}")
    print(f"Test cases:       {len(_TEST_CASES)}\n")

    results: list[CaseResult] = []
    for case in _TEST_CASES:
        result = _run_case(case)
        results.append(result)
        _print_case(result)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n{'=' * 78}")
    print(f"OVERALL: {passed}/{total} cases passed (bar: 8/10)")
    for r in results:
        marker = "OK  " if r.passed else "FAIL"
        print(f"  {marker}  {r.case_id}")

    # Emit a JSON summary for downstream tooling / regression tracking.
    summary = {
        "model": REFUTATION_MODEL,
        "total": total,
        "passed": passed,
        "results": [
            {
                "case_id": r.case_id,
                "passed": r.passed,
                "elapsed_s": r.elapsed_s,
                "error": r.error,
                "hypotheses": [
                    {
                        "hypothesis_id": h.hypothesis_id,
                        "targets_claim_id": h.targets_claim_id,
                        "strategy": h.strategy.value,
                        "hypothesis_text": h.hypothesis_text,
                        "rationale": h.rationale,
                    }
                    for h in r.hypotheses
                ],
                "grades": [
                    {
                        "concrete": g.concrete,
                        "different": g.different,
                        "asserted": g.asserted,
                        "word_count": g.word_count,
                        "jaccard": round(g.jaccard, 3),
                        "notes": g.notes,
                    }
                    for g in r.grades
                ],
            }
            for r in results
        ],
    }
    out_path = "data/logs/block10_refutation_harness.json"
    try:
        import os
        os.makedirs("data/logs", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nWrote summary to {out_path}")
    except OSError as e:
        print(f"\nCould not write summary: {e}")

    if passed < 8:
        sys.exit(1)


if __name__ == "__main__":
    main()

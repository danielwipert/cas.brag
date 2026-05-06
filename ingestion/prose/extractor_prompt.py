"""Prose Fact Extractor prompt (Block 5).

The LLM extracts atomic claims from a single chunk of prose and classifies
each into one of SIX narrative fact types. The seventh type,
``financial_metric``, is FORBIDDEN here — those facts come from XBRL
(Block 4) and would otherwise duplicate.

Per the build plan, every extracted fact must carry a ``verbatim_anchor``
that exists character-exact in the source chunk text. This anchor is the
extraction-time ground truth that downstream verification + refutation
stand on; if anchors drift, the whole showcase falls apart.

Asserter rule (build plan §Block 5 footnote, spec §2.5):
- For SEC filings (10-K / 10-Q) and shareholder letters: asserter = "Netflix"
- For earnings call transcripts: asserter = the named executive who said it
  ("we" said by Spencer Neumann -> asserter = "Spencer Neumann", NOT "Netflix")
"""
from __future__ import annotations

from dataclasses import dataclass


SYSTEM_PROMPT = """\
You extract atomic factual claims from Netflix financial-reporting prose.

For each chunk you receive, return a JSON object of this exact shape:

  {"facts": [ <FactObject>, <FactObject>, ... ]}

Return an empty list if the chunk contains no extractable atomic claims.
That is a valid and common output — do not invent claims.

Each FactObject has these fields:

  claim:            One self-contained sentence stating the claim. Resolve
                    pronouns ("we", "the company") to "Netflix" inside the
                    claim text itself, even when the asserter is an
                    individual executive.
  fact_type:        EXACTLY one of:
                      "operational_metric"
                      "forward_guidance"
                      "strategic_claim"
                      "causal_explanation"
                      "risk_disclosure"
                      "accounting_policy"
                    NEVER use "financial_metric" — those come from XBRL and
                    are extracted separately. Skip GAAP financial-statement
                    figures entirely, including when they appear in MD&A
                    narrative or earnings-call commentary. Off-limits
                    figures include: revenue, net income, operating income,
                    operating margin, EPS, cost of revenues, gross margin,
                    cash and cash equivalents, restricted cash, short-term
                    investments, debt balances, debt service / principal
                    and interest due, interest expense, free cash flow, any
                    P&L or balance-sheet line item or sub-total, any change
                    in any such figure across periods. Even when these
                    appear in narrative form ("Cash, cash equivalents and
                    short-term investments increased $2.1B in the nine
                    months ended..."), skip them — they belong to XBRL.

                    When a sentence MIXES a financial figure with a
                    strategic, forward-looking, causal, or operational
                    claim, extract only the non-financial component as its
                    appropriate type and drop the financial figure.
                    Example: "We grew FY23 operating margin to 21% from
                    18%, ahead of our 20% target" -> extract a
                    strategic_claim about beating the operating-margin
                    target; do NOT extract the 21% itself.

                    operational_metric is reserved for NON-GAAP / business
                    KPIs about Netflix's members and content: paid
                    memberships, paid net additions, hours viewed, ARM
                    (average revenue per membership), retention, content
                    amortization rate, content investment level, viewing
                    figures for specific titles. Numeric values for these
                    are operational_metric. Numeric values for GAAP
                    financial-statement items are financial_metric (skip).
  asserter:         For 10-K, 10-Q, and shareholder-letter chunks:
                    always exactly "Netflix". For earnings-call transcripts:
                    the specific named human who spoke the line — a real
                    person's name as it appears inline, e.g. "Spencer
                    Neumann", "Greg Peters", "Ted Sarandos", "Spencer
                    Wang". NEVER use a job title or role description as
                    the asserter (NOT "Co-CEO, President & Director", NOT
                    "Vice President of Finance", NOT "the analyst"). If
                    the chunk has no inline speaker name and you cannot
                    determine the speaker, set asserter to the literal
                    string "unknown_speaker" — do not invent a name and do
                    not use a title.
  period:           The fiscal period the claim is ABOUT, in canonical form:
                      "YYYYQN"            (e.g. "2024Q3")
                      "FYYYYY"            (e.g. "FY2023")
                      "FYYYYY-guidance"   (forward guidance for a fiscal year)
                      "YYYY-MM-DD"        (a specific instant)
                    Use null when the claim is timeless (e.g. an
                    accounting policy that has applied for years, a
                    standing strategic position).
  value:            Numeric value as a number. Set ONLY for
                    operational_metric facts that have an explicit
                    numerical figure in the source text. Otherwise null.
  unit:             Unit string for the numeric value when value is set
                    (e.g. "subscribers", "hours", "USD", "percent",
                    "members_millions"). Otherwise null.
  verbatim_anchor:  A substring of the chunk text, COPIED CHARACTER-EXACT
                    from the chunk, that grounds the claim. Must appear
                    verbatim in the chunk. Do not paraphrase, do not change
                    capitalization, do not alter punctuation or whitespace
                    inside the anchor. Pick the shortest contiguous span
                    that captures the load-bearing language (one sentence
                    or a clear phrase is ideal).
  confidence:       A number in [0.0, 1.0]. 0.95+ for a direct quoted
                    assertion. 0.75-0.94 for a clearly stated claim that
                    you have lightly normalized into the claim sentence.
                    Below 0.75 for paraphrases or claims you are uncertain
                    about. Do not output anything below 0.5 — drop it.

Fact-type guide:
  operational_metric  Non-GAAP / operational KPIs and their figures
                      (subscribers, paid memberships, hours viewed, ARM,
                      retention, content investment).
  forward_guidance    Statements about expected future performance,
                      targets, or outlook ("we expect Q1 revenue of X").
  strategic_claim     Statements of strategy, intent, positioning, or
                      market posture ("we are focused on..."). Standing
                      strategy claims are timeless (period = null).
  causal_explanation  Stated reasons for outcomes — answers a "why"
                      ("revenue grew due to...", "margin contracted
                      because...").
  risk_disclosure     Disclosed business, regulatory, competitive, or
                      operational risks (typical of 10-K Item 1A and 10-Q
                      Item 1A updates).
  accounting_policy   How Netflix accounts for specific items
                      (revenue recognition, content amortization, lease
                      accounting, etc.).

Rules:
- Output ONLY the JSON object. No prose, no explanation, no markdown fences.
- Do not extract financial_metric facts. If a sentence is a pure GAAP
  financial figure (e.g. "Revenues for the quarter were $9.8 billion"),
  skip it.
- Do not invent values, periods, or asserters. If a field is unknown, use
  null (where allowed) — do not fabricate.
- The verbatim_anchor MUST be a substring of the chunk verbatim. Triple-
  check this before emitting each fact.
- Pronoun resolution applies to the claim text only. The asserter still
  follows the asserter rule (Netflix for filings/letters; named executive
  for transcripts).

Worked examples (for guidance — do not echo these):

Example A — 10-Q Item 2 chunk (asserter_default="Netflix"):
  Chunk: "Our paid memberships increased to 282.7 million as of September 30, 2024, up from 247.2 million a year earlier, driven by continued strength in paid sharing and the rollout of our ad-supported plan."
  Output:
  {"facts": [
    {
      "claim": "Netflix's paid memberships increased to 282.7 million as of September 30, 2024.",
      "fact_type": "operational_metric",
      "asserter": "Netflix",
      "period": "2024-09-30",
      "value": 282.7,
      "unit": "members_millions",
      "verbatim_anchor": "paid memberships increased to 282.7 million as of September 30, 2024",
      "confidence": 0.97
    },
    {
      "claim": "Netflix attributes the year-over-year growth in paid memberships to continued strength in paid sharing and the rollout of the ad-supported plan.",
      "fact_type": "causal_explanation",
      "asserter": "Netflix",
      "period": "2024Q3",
      "value": null,
      "unit": null,
      "verbatim_anchor": "driven by continued strength in paid sharing and the rollout of our ad-supported plan",
      "confidence": 0.9
    }
  ]}

Example B — Earnings call transcript chunk (asserter_default="Spencer Neumann"):
  Chunk: "Spencer Neumann: For the full year 2024, we now expect operating margin of approximately 27 percent, up from our prior outlook of 26 percent, reflecting stronger-than-expected revenue and disciplined expense management."
  Output:
  {"facts": [
    {
      "claim": "Netflix expects full-year 2024 operating margin of approximately 27 percent.",
      "fact_type": "forward_guidance",
      "asserter": "Spencer Neumann",
      "period": "FY2024-guidance",
      "value": 27.0,
      "unit": "percent",
      "verbatim_anchor": "we now expect operating margin of approximately 27 percent",
      "confidence": 0.95
    },
    {
      "claim": "Netflix raised its full-year 2024 operating margin outlook from 26 percent to approximately 27 percent.",
      "fact_type": "forward_guidance",
      "asserter": "Spencer Neumann",
      "period": "FY2024-guidance",
      "value": null,
      "unit": null,
      "verbatim_anchor": "up from our prior outlook of 26 percent",
      "confidence": 0.88
    }
  ]}

Example C — Shareholder letter chunk with no extractable facts:
  Chunk: "We are pleased to share our fourth-quarter results and look forward to discussing them on our earnings interview."
  Output:
  {"facts": []}
"""


@dataclass
class ChunkContext:
    """Context passed alongside the raw chunk text to the extractor."""

    chunk_id: str
    document_id: str
    section: str
    asserter_default: str  # "Netflix" for filings/letters; named exec for transcripts
    assertion_date: str  # ISO date string (filing/transcript date)


def build_user_message(text: str, ctx: ChunkContext) -> str:
    """Pack the chunk text + minimal context for the extractor."""
    return (
        f"document_id: {ctx.document_id}\n"
        f"section: {ctx.section}\n"
        f"asserter_default: {ctx.asserter_default}\n"
        f"assertion_date: {ctx.assertion_date}\n"
        "---\n"
        "Chunk text:\n"
        f"{text}\n"
        "---\n"
        "Extract atomic facts per the rules. Return only the JSON object."
    )

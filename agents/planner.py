"""Block 7b: Planner Agent.

The Planner receives a validated query plus a complexity tier and emits
a ``DecompositionPlan`` (schemas/records.py) with 1..N independent
evidence slots. This is the most consequential single decision in the
pipeline per spec §3.3 — the precision of every downstream retrieval
pass depends on it.

Implementation choices:

* Model: ``meta-llama/llama-3.3-70b-instruct`` via OpenRouter, as the
  spec calls for. The free-tier slug is preferred and falls back to the
  paid slug if the free model is rate-limited.
* JSON mode: the model is forced to emit a single JSON object that maps
  one-to-one onto the DecompositionPlan schema, simplifying parsing.
* One retry: if the first response fails Pydantic validation, the model
  is called once more with the validator's error message appended. This
  catches the common failure mode of missing/misnamed fields without
  paying for repeated retries on harder errors.
* No range periods: ``period_filter`` accepts only canonical single-
  period strings (``2024Q3``, ``FY2023``, ``FY2024-guidance``,
  ``YYYY-MM-DD``). For multi-period queries the Planner produces
  multiple slots, one per period — consistent with the spec's
  slot-independence rule.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from pydantic import ValidationError

from agents.llm_client import LLMError, LLMResponse, OpenRouterClient
from schemas.enums import ComplexityTier, EvidenceType, TargetLayer
from schemas.records import DecompositionPlan


# Markdown fence variants the model occasionally wraps JSON in despite the
# explicit "return JSON only" instruction. Detected and stripped on parse.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL)
# Permissive object extraction: take the first `{` through the last `}`.
# This rescues responses that prepend prose ("Here is the plan:") even
# though the system prompt forbids it.
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_planner_json(content: str | None) -> Any:
    """Tolerant JSON parser for the planner's response. Strips ``` fences,
    falls back to the first JSON object substring, raises ``LLMError`` if
    nothing parses. Treats None content (some providers return null on
    internal errors) as a parse failure."""
    if not content:
        raise LLMError(
            "Planner response had empty content. The upstream provider "
            "likely failed silently — retry the call."
        )
    text = content.strip()
    fence_match = _FENCE_RE.match(text)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    obj_match = _OBJECT_RE.search(text)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError as e:
            raise LLMError(
                f"Planner response contained a JSON-like object that failed "
                f"to parse: {e}. Snippet: {obj_match.group(0)[:400]!r}"
            ) from e
    raise LLMError(
        f"Planner response is not JSON and contains no JSON object: "
        f"{content[:400]!r}"
    )


# Spec §3.3 calls for Llama 3.3 70B. OpenRouter exposes both a free and
# a paid slug. We default to the paid slug because the :free slug is
# pinned to Venice as the upstream provider (as of 2026-05), which
# returns null content and 429s on most calls. The paid slug routes
# across Together / DeepInfra / Hyperbolic / Cerebras and is reliable.
PLANNER_MODEL = "meta-llama/llama-3.3-70b-instruct"
PLANNER_MODEL_FREE = "meta-llama/llama-3.3-70b-instruct:free"

# Defense in depth: explicitly exclude the Venice provider for both the
# free and paid slugs. OpenRouter falls back to the next provider in the
# preference list. Free callers without paid balance who pass the :free
# slug will fail closed rather than hit broken Venice.
_PROVIDER_PREFS: dict[str, Any] = {"provider": {"ignore": ["Venice"]}}

# Slot budget by tier, per spec §3.2 table.
_MAX_SLOTS_BY_TIER: dict[ComplexityTier, int] = {
    ComplexityTier.simple: 2,
    ComplexityTier.standard: 4,
    ComplexityTier.complex: 6,
}


SYSTEM_PROMPT_TEMPLATE = """\
You are the Planner Agent in BRAG, a retrieval-augmented question-answering
system over ten years of Netflix public financial reporting (10-K, 10-Q,
shareholder letters, earnings call transcripts; May 2016–May 2026).

Your job is to decompose a user query into independent evidence slots,
each of which the Retriever can satisfy by querying the Fact Store or
the Chunk Store. The quality of decomposition determines the precision
of every downstream retrieval pass.

# Output

Return a single JSON object that maps onto this schema:

{
  "synthesis_strategy": "compare" | "contrast" | "sequence" | "integrate",
  "slots": [
    {
      "slot_id": "S1",
      "sub_question": "<standalone question this slot must answer>",
      "evidence_type": "<one of the 10 evidence_types listed below>",
      "target_layer": "fact_store" | "chunk_store" | "both",
      "period_filter": "<canonical period string>" | null,
      "key_terms": ["...", "..."],
      "coverage_threshold": 0.80
    }
  ]
}

Do not include the original query or query_id — those are supplied by the
caller. Do not wrap the JSON in markdown fences. Return JSON only.

# Evidence types

Pick exactly one per slot:

- specific_metric — a single numerical value for a specific period.
  Example: "Netflix's Q3 2024 operating income"
- definition — definition of a corpus term or accounting concept.
  Example: "How does Netflix define a paid membership?"
- forward_looking_statement — a stated expectation or guidance about a
  future period. Example: "Netflix's 2024 free cash flow guidance"
- strategic_position — a stated stance, intention, or strategic claim.
  Example: "Netflix's stance on running ads"
- cross_period_comparison — a comparison of values across two or more
  periods. Example: "Operating margin in FY2019 vs FY2023"
- causal_explanation — a stated reason for a result or trend.
  Example: "Why did free cash flow improve in 2022?"
- temporal_evolution — narrative of how something changed over time.
  Example: "How content amortization grew across 2020-2024"
- risk_disclosure — a stated business risk from 10-K Item 1A or
  shareholder letters. Example: "Subscriber retention risks"
- accounting_policy — a stated accounting methodology or treatment.
  Example: "Netflix's content amortization policy"
- contradiction_detection — explicitly seeking a contradiction across
  the corpus. Example: "Has Netflix ever contradicted itself on ads?"

# target_layer routing

- specific_metric, cross_period_comparison → "fact_store"
  (XBRL is the authoritative source for atomic numerical values.)
- All narrative slot types — strategic_position, contradiction_detection,
  causal_explanation, temporal_evolution, forward_looking_statement,
  risk_disclosure, accounting_policy, definition → "both"
  (The prose extractor produces atomic facts for these types in the
  fact_store; surrounding chunks carry the context the verifier
  needs to judge claim support and the refutation agent needs to
  surface refuting positions. Routing to "chunk_store" alone hides
  the prose facts from the verifier, which makes refutation bypass
  and slots exhaust on chunk-only coverage. Always use "both" for
  narrative types.)

# period_filter

Set ONLY for slots whose sub_question is anchored in a specific period.
Accepted formats:

- "2024Q3" — calendar quarter (Q1..Q4)
- "FY2023" — fiscal year
- "FY2024-guidance" — fiscal-year forward guidance
- "2024-09-30" — instant date (balance-sheet item)
- null — slot is not period-scoped

DO NOT use ranges like "FY2020-FY2023". For multi-period queries, emit
ONE slot per period, each with its own single-period filter.

CRITICAL: an FY-level filter (FY2018, FY2023, etc.) matches ONLY the
fiscal-year 10-K. It does NOT match the four quarterly shareholder
letters or transcripts from that year. So:

  - For specific_metric and cross_period_comparison: FY filters are
    correct, since the 10-K is the authoritative source.
  - For strategic_position, temporal_evolution, causal_explanation,
    risk_disclosure, accounting_policy, contradiction_detection: the
    content lives in quarterly letters/transcripts and narrative
    sections of the 10-K. Use NULL or a specific quarter
    ("2018Q3"); NEVER an FY filter — it will drop every relevant
    shareholder letter. When the user query says "in 2018" but the
    fact_type is narrative, emit ``period_filter: null`` and let
    BM25 + key_terms locate the right quarter.

CRITICAL: forward guidance lives in the SOURCE-DOC period, not the
guided-about period. When the user asks about guidance that was given
at a past meeting/call for a future period, the period_filter must
point at the meeting (when the guidance was made), not at the period
being guided about. Examples:

  - "What guidance did Netflix give on Q1 2024 paid memberships at the
    Q4 2023 earnings call?" → ``period_filter: "2023Q4"`` (Q4 2023
    letter/transcript holds the guidance) — NOT 2024Q1.
  - "What was Netflix's revenue outlook for FY2025 at the Q4 2024
    call?" → ``period_filter: "2024Q4"`` — NOT FY2025.
  - "What is Netflix's guidance for FY2026?" (no specific call named)
    → ``period_filter: "FY2026-guidance"`` is acceptable because the
    fact_store's forward_guidance facts are tagged this way; or
    ``period_filter: null`` and let key_terms locate the most recent
    guidance.

The rule of thumb: ``forward_looking_statement`` slots whose user
query names a specific past meeting/call/letter should filter on
THAT meeting's period. Slots that just ask "what's Netflix's outlook
for X" without naming a source can use FY{Y}-guidance or null.

# coverage_threshold

Default to ``0.80`` for almost every slot type — that's the rubric
target for "evidence directly answers the sub_question with at most
minor missing context."

EXCEPTION: ``forward_looking_statement`` slots use ``0.50``. Netflix's
forward guidance is often directional rather than numerical (e.g.,
"we expect paid net additions to be down sequentially but up versus
Q1'23 paid net adds of 1.8M"), and the Verifier's rubric rates such
qualitative guidance as 0.20–0.50. A 0.80 threshold on this evidence
type drops most of Netflix's actual guidance language as "topically
adjacent." 0.50 lets directional guidance count as covered while
still requiring the candidate to address the asked-about period.

# key_terms

Each key_term must be specific enough to anchor BM25 lexical retrieval.
Bare terms like "Netflix" or "revenue" are NOT valid; they must be
qualified ("Netflix advertising revenue", "operating income Q3 2024",
"paid net additions Q4 2023").

Aim for 3–6 key_terms per slot.

# synthesis_strategy

Pick the single best fit:

- compare — slots present parallel facts and the answer compares them.
- contrast — slots set up opposition (e.g., guidance vs actual).
- sequence — slots describe an evolution in chronological order.
- integrate — slots cover different facets that the answer weaves.

# Slot independence

Each slot must be answerable in isolation. A slot that depends on the
answer to another slot is not a valid slot — it is a synthesis step
and should be dropped from the plan.

# Slot budget

You may emit AT MOST __MAX_SLOTS__ slots for this tier (__TIER__). Fewer is
fine — emit only the slots that are genuinely needed. Do not pad.
"""


_FEW_SHOTS: list[tuple[str, ComplexityTier, dict[str, Any]]] = [
    (
        "What was Netflix's net income in Q1 2024?",
        ComplexityTier.simple,
        {
            "synthesis_strategy": "integrate",
            "slots": [
                {
                    "slot_id": "S1",
                    "sub_question": "What was Netflix's net income for Q1 2024?",
                    "evidence_type": "specific_metric",
                    "target_layer": "fact_store",
                    "period_filter": "2024Q1",
                    "key_terms": [
                        "Netflix net income Q1 2024",
                        "net income loss Q1 2024",
                    ],
                    "coverage_threshold": 0.80,
                }
            ],
        },
    ),
    (
        "Compare Netflix's operating margin in FY2019 and FY2023",
        ComplexityTier.standard,
        {
            "synthesis_strategy": "compare",
            "slots": [
                {
                    "slot_id": "S1",
                    "sub_question": "What was Netflix's operating margin for FY2019?",
                    "evidence_type": "specific_metric",
                    "target_layer": "fact_store",
                    "period_filter": "FY2019",
                    "key_terms": [
                        "Netflix operating margin FY2019",
                        "operating income FY2019",
                        "operating margin 2019",
                    ],
                    "coverage_threshold": 0.80,
                },
                {
                    "slot_id": "S2",
                    "sub_question": "What was Netflix's operating margin for FY2023?",
                    "evidence_type": "specific_metric",
                    "target_layer": "fact_store",
                    "period_filter": "FY2023",
                    "key_terms": [
                        "Netflix operating margin FY2023",
                        "operating income FY2023",
                        "operating margin 2023",
                    ],
                    "coverage_threshold": 0.80,
                },
            ],
        },
    ),
    (
        # Block 23: forward-guidance source-doc period + lower threshold.
        # period_filter points at the CALL where the guidance was given
        # (2023Q4), not the period being guided about (2024Q1).
        # coverage_threshold is 0.50 for forward_looking_statement
        # because Netflix's guidance is often directional, not numerical.
        "What guidance did Netflix give on Q1 2024 paid memberships "
        "at the Q4 2023 earnings call?",
        ComplexityTier.standard,
        {
            "synthesis_strategy": "integrate",
            "slots": [
                {
                    "slot_id": "S1",
                    "sub_question": (
                        "What did Netflix say at the Q4 2023 earnings "
                        "call about Q1 2024 paid memberships?"
                    ),
                    "evidence_type": "forward_looking_statement",
                    "target_layer": "both",
                    "period_filter": "2023Q4",
                    "key_terms": [
                        "Netflix Q4 2023 earnings call paid memberships",
                        "Q1 2024 paid net additions outlook",
                        "Q4 2023 shareholder letter Q1 2024 guidance",
                    ],
                    "coverage_threshold": 0.50,
                }
            ],
        },
    ),
    (
        "How did Netflix's stance on advertising evolve from 2016 to 2024, "
        "and what financial trajectory accompanied the shift?",
        ComplexityTier.complex,
        {
            "synthesis_strategy": "sequence",
            "slots": [
                {
                    "slot_id": "S1",
                    "sub_question": "What was Netflix's stated stance on advertising in 2016-2018?",
                    "evidence_type": "strategic_position",
                    "target_layer": "both",
                    "period_filter": None,
                    "key_terms": [
                        "Netflix advertising stance 2017",
                        "no advertising strategic differentiator",
                        "ad-free consumer experience",
                    ],
                    "coverage_threshold": 0.80,
                },
                {
                    "slot_id": "S2",
                    "sub_question": "When and how did Netflix announce its shift toward an ad-supported tier?",
                    "evidence_type": "temporal_evolution",
                    "target_layer": "both",
                    "period_filter": None,
                    "key_terms": [
                        "Netflix ad-supported tier announcement 2022",
                        "Basic with Ads launch",
                        "advertising tier strategy shift",
                    ],
                    "coverage_threshold": 0.80,
                },
                {
                    "slot_id": "S3",
                    "sub_question": "What revenue and operating-margin trajectory did Netflix report through the ad-tier rollout period?",
                    "evidence_type": "cross_period_comparison",
                    "target_layer": "fact_store",
                    "period_filter": None,
                    "key_terms": [
                        "Netflix revenue 2022 2023 2024",
                        "operating margin trajectory 2022 2024",
                        "advertising revenue contribution",
                    ],
                    "coverage_threshold": 0.80,
                },
                {
                    "slot_id": "S4",
                    "sub_question": "What rationale did Netflix give for adopting the ad-supported tier?",
                    "evidence_type": "causal_explanation",
                    "target_layer": "both",
                    "period_filter": None,
                    "key_terms": [
                        "Netflix advertising tier rationale",
                        "ad tier price-sensitive members",
                        "ad business strategic justification",
                    ],
                    "coverage_threshold": 0.80,
                },
            ],
        },
    ),
]


def _build_system_prompt(tier: ComplexityTier) -> str:
    return (
        SYSTEM_PROMPT_TEMPLATE
        .replace("__MAX_SLOTS__", str(_MAX_SLOTS_BY_TIER[tier]))
        .replace("__TIER__", tier.value)
    )


def _build_messages(
    query: str,
    tier: ComplexityTier,
    *,
    prior_error: str | None = None,
) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = [
        {"role": "system", "content": _build_system_prompt(tier)},
    ]
    # In-prompt few-shots as alternating user/assistant turns. The model
    # learns the output shape from these examples.
    for shot_q, shot_tier, shot_plan in _FEW_SHOTS:
        msgs.append({
            "role": "user",
            "content": f"Query: {shot_q}\nComplexity tier: {shot_tier.value}",
        })
        msgs.append({
            "role": "assistant",
            "content": json.dumps(shot_plan),
        })
    user_content = f"Query: {query}\nComplexity tier: {tier.value}"
    if prior_error:
        user_content += (
            f"\n\nYour previous response failed validation with this error:\n"
            f"{prior_error}\n\n"
            f"Return a corrected JSON object."
        )
    msgs.append({"role": "user", "content": user_content})
    return msgs


def _query_id(query: str) -> str:
    """Short, deterministic ID derived from the query text. Block 8+
    pipeline runs may override this with a per-run UUID; the Planner's
    default is a content-hash for repeatable testing."""
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:10]
    return f"q_{digest}"


# Evidence types whose content lives in narrative documents
# (shareholder letters, transcripts, 10-K narrative items) that are
# anchored to a specific quarter, not a fiscal year. For these slot
# types, an FY-level period_filter (e.g. FY2018) drops every
# quarterly-anchored letter/transcript and almost always retrieves
# nothing useful. Block 12 calibration confirmed this kills the
# "no plans for ads" showcase. We post-process the Planner's plan to
# drop FY-level filters on these slot types; quarterly filters
# (2018Q3) are kept because they correctly target one letter.
_NARRATIVE_EVIDENCE_TYPES: frozenset[EvidenceType] = frozenset({
    EvidenceType.strategic_position,
    EvidenceType.temporal_evolution,
    EvidenceType.causal_explanation,
    EvidenceType.risk_disclosure,
    EvidenceType.accounting_policy,
    EvidenceType.contradiction_detection,
})


def _is_fy_filter(period_filter: str | None) -> bool:
    """True for FYYYYY and FYYYYY-guidance filters; False for quarter
    (YYYYQN) and instant (YYYY-MM-DD) filters and for None."""
    return period_filter is not None and period_filter.startswith("FY")


def _normalize_plan_period_filters(plan: DecompositionPlan) -> DecompositionPlan:
    """Two post-LLM corrections for narrative slot types:

    1. Drop FY-level period_filters — they filter out every quarterly-
       anchored shareholder letter and transcript where the narrative
       content actually lives, leaving the verifier with nothing.
    2. Upgrade ``chunk_store``-only routing to ``both`` — the prose
       extractor produces atomic facts for these types in the
       fact_store, and the verifier+refutation stages need them.
       Routing to chunks alone hides those facts and forces the
       verifier to synthesize from raw chunk text, which produces
       low coverage scores and bypasses refutation downstream.

    Both fixes are defense in depth — the system prompt already
    instructs Llama 3.3 70B correctly, but it occasionally lapses
    on 'in YYYY' queries (FY pinning) and on strategic_position
    routing (chunk_store)."""
    new_slots = []
    changed = False
    for s in plan.slots:
        updates: dict[str, Any] = {}
        if s.evidence_type in _NARRATIVE_EVIDENCE_TYPES:
            if _is_fy_filter(s.period_filter):
                updates["period_filter"] = None
            if s.target_layer == TargetLayer.chunk_store:
                updates["target_layer"] = TargetLayer.both
        if updates:
            new_slots.append(s.model_copy(update=updates))
            changed = True
        else:
            new_slots.append(s)
    return plan.model_copy(update={"slots": new_slots}) if changed else plan


def _validate_plan(
    payload: Any,
    *,
    query: str,
    tier: ComplexityTier,
) -> DecompositionPlan:
    """Pydantic-validate the model's JSON. Pass the caller-supplied
    query_id, original_query, and complexity_tier directly so the model
    is responsible only for the synthesis_strategy + slots structure."""
    if not isinstance(payload, dict):
        raise ValueError(
            f"Planner returned non-object JSON: {type(payload).__name__}"
        )
    full = {
        "query_id": _query_id(query),
        "original_query": query,
        "complexity_tier": tier.value,
        "synthesis_strategy": payload.get("synthesis_strategy"),
        "slots": payload.get("slots") or [],
    }
    plan = DecompositionPlan.model_validate(full)
    # Enforce the per-tier slot budget after Pydantic validation so a
    # model that emits too many slots is rejected via the retry path.
    max_slots = _MAX_SLOTS_BY_TIER[tier]
    if len(plan.slots) > max_slots:
        raise ValueError(
            f"Plan has {len(plan.slots)} slots; tier {tier.value} caps at {max_slots}."
        )
    if not plan.slots:
        raise ValueError("Plan has zero slots.")
    return _normalize_plan_period_filters(plan)


def plan(
    query: str,
    tier: ComplexityTier,
    *,
    client: OpenRouterClient | None = None,
    model: str | None = None,
) -> tuple[DecompositionPlan, LLMResponse]:
    """Plan ``query`` at ``tier`` and return the validated DecompositionPlan
    plus the raw LLM response (for logging / token accounting).

    JSON mode (``response_format={"type": "json_object"}``) is NOT used
    because OpenRouter's free-tier provider for Llama 3.3 70B (Venice as
    of 2026-05) rejects that parameter. The system prompt and few-shot
    examples are sufficient to make the model emit JSON; the parser
    strips occasional markdown fences and prose preambles.

    Retries once on validation error with the error attached to the user
    message. A second failure raises ``LLMError``."""
    if client is None:
        client = OpenRouterClient(default_model=model or PLANNER_MODEL)
    chosen_model = model or PLANNER_MODEL

    def _call(messages: list[dict[str, str]]) -> tuple[Any, LLMResponse]:
        # max_tokens — keep the response cap modest so OpenRouter doesn't
        # request 50k+ tokens (its default fills the remaining context
        # window) which trips smaller providers' upper caps. 4000 is well
        # above the largest plausible Complex-tier JSON plan (6 slots *
        # ~150 tokens each + envelope ≈ 1200 tokens).
        # provider.ignore excludes Venice — the upstream provider whose
        # null-content / 429 behavior breaks the :free slug as of 2026-05.
        resp = client.chat(
            messages,
            model=chosen_model,
            max_tokens=4000,
            extra=_PROVIDER_PREFS,
        )
        parsed = _parse_planner_json(resp.content)
        return parsed, resp

    messages = _build_messages(query, tier)
    parsed, resp = _call(messages)
    try:
        return _validate_plan(parsed, query=query, tier=tier), resp
    except (ValidationError, ValueError) as first_err:
        retry_messages = _build_messages(query, tier, prior_error=str(first_err))
        parsed2, resp2 = _call(retry_messages)
        try:
            return _validate_plan(parsed2, query=query, tier=tier), resp2
        except (ValidationError, ValueError) as second_err:
            raise LLMError(
                f"Planner failed schema validation twice on query {query!r}. "
                f"First error: {first_err}. Second error: {second_err}."
            ) from second_err

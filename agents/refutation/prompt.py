"""Block 10: Refutation Agent — hypothesis-generation prompt.

The Refutation Agent's job is adversarial: given the verified evidence
set the Verifier produced, generate 1–3 concrete, testable counter-
hypotheses that retrieval can attempt to confirm. If retrieval *finds*
strong support for a counter-hypothesis, the original answer is in
trouble (spec §3.6).

This module owns *only* the prompt and the surrounding shim that calls
the LLM and validates the returned JSON. The agent.py wrapper in Block
11 is what runs hypothesis-driven retrieval, classifies findings as
unrefuted / weakly_refuted / strongly_refuted, and decides whether the
pipeline should loop or degrade. Keeping the prompt isolated lets us
iterate on hypothesis quality without touching the pipeline.

Per Chorus Principle 7, the Refutation Agent MUST use a different model
family from the Planner (Llama 3.3 70B), the Verifier (Qwen2.5 72B),
the Block-5 extractor (DeepSeek), and the embedder (BAAI bge). We use
Mistral Large 2411 — the current flagship in the Mistral family that
the build plan v3 originally named for this slot (it specified Mixtral
8x22B; Mistral Large 2411 is its successor in the same family with
better instruction following and reliable JSON mode).

Strategy table — the spec's seven refutation strategies, each tied to
one fact_type so the Refutation Agent's job is taxonomic rather than
creative:

    financial_metric    → restated_value      (later filing restates value)
    operational_metric  → revised_value       (later doc revises value)
    forward_guidance    → guidance_vs_actual  (later actual misses guidance)
    strategic_claim     → later_reversal      (later statement reverses position)
    causal_explanation  → alternative_cause   (different stated cause same outcome)
    risk_disclosure     → materialization     (disclosed risk later materializes)
    accounting_policy   → policy_change       (policy changed in a later year)

Hypothesis count by complexity tier (spec §3.6):

    Simple   → 1 hypothesis
    Standard → 2 hypotheses
    Complex  → 3 hypotheses
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from agents.llm_client import LLMError, LLMResponse, OpenRouterClient
from schemas.enums import ComplexityTier, FactType, RefutationStrategy
from schemas.records import FactRecord, RefutationHypothesis


# Mistral Large 2411 (flagship). The :free slug is unreliable for Mistral
# on OpenRouter (heavy rate limits); paid slug is the default. The named
# fallback below is what Block 11's degraded-coverage path will retry
# against before invoking the cross-family Llama 3.3 70B fallback the
# build plan calls for.
REFUTATION_MODEL = "mistralai/mistral-large-2411"
REFUTATION_MODEL_FALLBACK = "mistralai/mixtral-8x22b-instruct"

# Defense in depth — same as the Planner and Verifier; some upstream
# providers behave poorly on JSON-mode requests.
_PROVIDER_PREFS: dict[str, Any] = {"provider": {"ignore": ["Venice"]}}


_HYPOTHESES_PER_TIER: dict[ComplexityTier, int] = {
    ComplexityTier.simple: 1,
    ComplexityTier.standard: 2,
    ComplexityTier.complex: 3,
}


# Authoritative mapping enforced post-LLM so a model that picks the
# wrong strategy for a given fact_type is rejected via the retry path.
_STRATEGY_FOR_FACT_TYPE: dict[FactType, RefutationStrategy] = {
    FactType.financial_metric: RefutationStrategy.restated_value,
    FactType.operational_metric: RefutationStrategy.revised_value,
    FactType.forward_guidance: RefutationStrategy.guidance_vs_actual,
    FactType.strategic_claim: RefutationStrategy.later_reversal,
    FactType.causal_explanation: RefutationStrategy.alternative_cause,
    FactType.risk_disclosure: RefutationStrategy.materialization,
    FactType.accounting_policy: RefutationStrategy.policy_change,
}


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


SYSTEM_PROMPT = """\
You are the Refutation Agent in BRAG. Your job is adversarial: given a
verified evidence set that the Verifier judged as supporting a query,
generate concrete counter-hypotheses that retrieval can attempt to
confirm. If retrieval finds strong support for a counter-hypothesis,
the original answer is in trouble.

You do NOT generate the answer, run retrieval, or render a verdict. You
emit hypotheses — that is all.

# Strategy table (load-bearing)

Each verified fact has a fact_type. The strategy you choose for a
hypothesis targeting that fact MUST match this table exactly:

  financial_metric    → restated_value
  operational_metric  → revised_value
  forward_guidance    → guidance_vs_actual
  strategic_claim     → later_reversal
  causal_explanation  → alternative_cause
  risk_disclosure     → materialization
  accounting_policy   → policy_change

Strategy semantics:

  restated_value      A later Netflix filing restates the same metric
                      (same concept, same period) to a different value.
                      Example: Q2 2022 paid memberships were initially
                      221.6M, later restated to 220.7M in the 10-K.

  revised_value       A later Netflix document publishes a corrected /
                      revised figure for an operational metric.

  guidance_vs_actual  Subsequent actual results contradict the earlier
                      guidance. Example: Q1 2022 letter guides +2.5M
                      paid net adds for Q2 2022; Q2 2022 actuals show
                      a 970k loss.

  later_reversal      A later Netflix statement reverses, abandons, or
                      materially qualifies the earlier strategic
                      position. Example: 2018 letter says "no plans to
                      add ads"; Q4 2022 letter announces the ad tier.

  alternative_cause   A different cause is named (by Netflix, in a
                      later or contemporaneous document) for the same
                      outcome. Example: Verified set credits price
                      increases for revenue growth; hypothesis is that
                      a later letter attributes the growth primarily
                      to paid sharing crackdown.

  materialization     A risk that was disclosed in the abstract later
                      materialized as a concrete event the company
                      acknowledges. Example: Pre-2023 10-Ks disclose
                      password-sharing risk; 2023 letters discuss the
                      sharing crackdown as a present-tense action.

  policy_change       An accounting policy named in the verified set
                      was changed in a later filing. Example: 2017
                      content amortization policy was modified in the
                      2020 10-K.

# Quality bar (every hypothesis must satisfy all four)

1. CONCRETE — the hypothesis is a specific Netflix-asserted statement
   that retrieval can search for, not an abstract critique. Bad:
   "Netflix's strategy could be inconsistent." Good: "Netflix's Q4
   2022 shareholder letter announced an ad-supported tier, reversing
   its earlier no-ads position."

   MANDATORY elements in every hypothesis_text:
     (a) a specific time anchor — a year ("2023") or quarter ("Q4
         2022") or fiscal period ("FY2022"). "Later", "subsequently",
         and "in a future filing" are NOT acceptable substitutes.
     (b) a specific document kind — 10-K, 10-Q, shareholder letter,
         earnings transcript, earnings report, press release, or
         similarly named Netflix-issued document. "A filing" alone
         is too vague; pair it with the year (e.g., "a 2023 10-K").
     (c) the substantive counter-position itself, stated as a
         concrete claim retrieval can match against.

2. TESTABLE — vector + BM25 retrieval against a Netflix filing /
   transcript / shareholder-letter corpus has a defined notion of
   "found" or "not found" for this claim. A hypothesis that is too
   abstract to produce hits or misses is unusable.

3. NETFLIX-ASSERTED — the counter-position must be plausibly
   representable as a Netflix-asserted statement (10-K, 10-Q,
   shareholder letter, earnings transcript). Do not propose
   hypotheses that require third-party assertions (analyst notes,
   competitor filings, journalism) — the corpus does not contain
   those and retrieval cannot test them.

4. MEANINGFULLY DIFFERENT — the hypothesis is a genuine counter-
   position, not a paraphrase of the verified claim. If your
   hypothesis would be satisfied by the same evidence that satisfied
   the verified claim, it is not a refutation.

# Strategy-selection rules

- One hypothesis MUST target exactly one verified fact via its
  fact_id (use it for `targets_claim_id`). Do not invent fact IDs.
- The hypothesis's `strategy` MUST be the one listed for the
  targeted fact's fact_type in the table above.
- When multiple verified facts are present, prefer targeting the
  most consequential, time-sensitive, or claim-bearing fact. Avoid
  targeting two facts of the same fact_type when other types are
  available — diversity of strategy is preferable.

# Output

Return a single JSON object (no markdown, no prose) with this shape:

{
  "hypotheses": [
    {
      "hypothesis_id": "h_1",
      "targets_claim_id": "<fact_id from the verified set>",
      "hypothesis_text": "<one sentence stating the counter-position>",
      "rationale": "<one sentence explaining why retrieval finding this would refute the targeted claim>",
      "strategy": "<one of: restated_value | revised_value | guidance_vs_actual | later_reversal | alternative_cause | materialization | policy_change>"
    }
  ]
}

The number of hypotheses MUST match what the user message specifies
(1 for Simple, 2 for Standard, 3 for Complex). Use sequential IDs
h_1, h_2, h_3. Return ONLY the JSON object.
"""


def _format_verified_fact(fact: FactRecord, idx: int) -> str:
    """Compact, deterministic representation of one verified fact for
    the LLM prompt. Mirrors the Verifier's candidate formatter."""
    parts = [
        f"  [F{idx}] fact_id={fact.fact_id}",
        f"        fact_type={fact.fact_type.value}",
        f"        asserter={fact.asserter}",
        f"        assertion_date={fact.assertion_date.isoformat()}",
        f"        source={fact.source_document} :: {fact.source_section}",
    ]
    if fact.period:
        parts.append(f"        period={fact.period}")
    if fact.value is not None:
        unit = f" {fact.unit}" if fact.unit else ""
        parts.append(f"        value={fact.value}{unit}")
    if fact.concept_tag:
        parts.append(f"        concept_tag={fact.concept_tag}")
    claim = fact.claim.replace("\n", " ").strip()
    if len(claim) > 400:
        claim = claim[:400] + "…"
    parts.append(f'        claim: "{claim}"')
    anchor = fact.verbatim_anchor.replace("\n", " ").strip()
    if len(anchor) > 400:
        anchor = anchor[:400] + "…"
    parts.append(f'        verbatim_anchor: "{anchor}"')
    return "\n".join(parts)


def _build_user_message(
    query: str,
    tier: ComplexityTier,
    verified_facts: list[FactRecord],
    *,
    prior_error: str | None = None,
) -> str:
    n = _HYPOTHESES_PER_TIER[tier]
    lines = [
        f"ORIGINAL QUERY: {query}",
        f"COMPLEXITY TIER: {tier.value} (generate exactly {n} "
        f"{'hypothesis' if n == 1 else 'hypotheses'})",
        "",
        f"VERIFIED EVIDENCE SET ({len(verified_facts)} fact"
        f"{'' if len(verified_facts) == 1 else 's'}):",
    ]
    for idx, fact in enumerate(verified_facts, start=1):
        lines.append(_format_verified_fact(fact, idx))
    lines.append("")
    lines.append(
        f"Generate {n} counter-{'hypothesis' if n == 1 else 'hypotheses'} "
        "as a JSON object matching the schema in the system prompt. Each "
        "hypothesis must target one of the verified facts by its exact "
        "fact_id and use the strategy mapped to that fact's fact_type."
    )
    if prior_error is not None:
        lines.append("")
        lines.append(
            "NOTE: your previous response failed validation. Error: "
            f"{prior_error}. Fix the issue and emit valid JSON."
        )
    return "\n".join(lines)


def _parse_refutation_json(content: str | None) -> Any:
    """Tolerant JSON parser. Strips ``` fences, falls back to the first
    JSON object substring, raises LLMError if nothing parses."""
    if not content:
        raise LLMError(
            "Refutation response had empty content. The upstream "
            "provider likely failed silently — retry the call."
        )
    text = content.strip()
    fm = _FENCE_RE.match(text)
    if fm:
        text = fm.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    om = _OBJECT_RE.search(text)
    if om:
        try:
            return json.loads(om.group(0))
        except json.JSONDecodeError as e:
            raise LLMError(
                f"Refutation response had unparseable JSON object: {e}. "
                f"Snippet: {om.group(0)[:400]!r}"
            ) from e
    raise LLMError(f"Refutation response is not JSON: {content[:400]!r}")


def _validate_hypotheses(
    payload: Any,
    *,
    tier: ComplexityTier,
    verified_facts: list[FactRecord],
) -> list[RefutationHypothesis]:
    """Validate the LLM payload and enforce the strategy-by-fact-type
    rule. Raises ValueError on any structural / semantic failure."""
    if not isinstance(payload, dict):
        raise ValueError(
            f"Refutation returned non-object JSON: {type(payload).__name__}"
        )
    raw = payload.get("hypotheses")
    if not isinstance(raw, list):
        raise ValueError(
            "Refutation payload missing 'hypotheses' list "
            f"(got {type(raw).__name__})."
        )

    expected_n = _HYPOTHESES_PER_TIER[tier]
    if len(raw) != expected_n:
        raise ValueError(
            f"Refutation produced {len(raw)} hypotheses; "
            f"tier {tier.value} requires exactly {expected_n}."
        )

    fact_by_id = {f.fact_id: f for f in verified_facts}
    hypotheses: list[RefutationHypothesis] = []
    seen_ids: set[str] = set()
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Hypothesis #{i} is not an object.")

        # Coerce hypothesis_id to a canonical sequential form so callers
        # don't have to trust the model's IDs.
        item = dict(item)
        item["hypothesis_id"] = f"h_{i}"

        targets = item.get("targets_claim_id")
        if not isinstance(targets, str) or targets not in fact_by_id:
            raise ValueError(
                f"Hypothesis #{i} targets_claim_id {targets!r} is not "
                f"one of the verified fact_ids: {sorted(fact_by_id)}."
            )
        if targets in seen_ids:
            raise ValueError(
                f"Hypothesis #{i} targets {targets!r}, but a prior "
                "hypothesis already targets that fact. Diversify."
            )
        seen_ids.add(targets)

        targeted_fact = fact_by_id[targets]
        required_strategy = _STRATEGY_FOR_FACT_TYPE[targeted_fact.fact_type]
        if item.get("strategy") != required_strategy.value:
            raise ValueError(
                f"Hypothesis #{i} targets a {targeted_fact.fact_type.value} "
                f"fact, which requires strategy {required_strategy.value!r}; "
                f"got {item.get('strategy')!r}."
            )

        try:
            hypotheses.append(RefutationHypothesis.model_validate(item))
        except ValidationError as e:
            raise ValueError(f"Hypothesis #{i} failed schema validation: {e}") from e

    return hypotheses


def generate_hypotheses(
    query: str,
    tier: ComplexityTier,
    verified_facts: list[FactRecord],
    *,
    client: OpenRouterClient | None = None,
    model: str | None = None,
    max_tokens: int = 1500,
) -> tuple[list[RefutationHypothesis], LLMResponse]:
    """Generate counter-hypotheses for ``verified_facts``.

    Retries once on parse / validation failure, with the validator's
    error message appended to the user message. A second failure
    raises LLMError. Returns the validated hypotheses plus the raw
    LLM response (for token accounting and Live Trace logging)."""
    if not verified_facts:
        raise ValueError(
            "Refutation Agent requires at least one verified fact. "
            "Bypass the agent (per spec §3.6 Partial degradation rule) "
            "if the verified set is empty."
        )
    for fact in verified_facts:
        if fact.fact_type not in _STRATEGY_FOR_FACT_TYPE:
            raise ValueError(
                f"Verified fact {fact.fact_id!r} has fact_type "
                f"{fact.fact_type!r} with no mapped strategy."
            )

    if client is None:
        client = OpenRouterClient(default_model=model or REFUTATION_MODEL)
    chosen_model = model or REFUTATION_MODEL

    def _call(prior_error: str | None) -> tuple[Any, LLMResponse]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_message(
                    query, tier, verified_facts, prior_error=prior_error
                ),
            },
        ]
        resp = client.chat(
            messages,
            model=chosen_model,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=max_tokens,
            extra=_PROVIDER_PREFS,
        )
        parsed = _parse_refutation_json(resp.content)
        return parsed, resp

    parsed, resp = _call(prior_error=None)
    try:
        return _validate_hypotheses(parsed, tier=tier, verified_facts=verified_facts), resp
    except (ValidationError, ValueError) as first_err:
        parsed2, resp2 = _call(prior_error=str(first_err))
        try:
            return (
                _validate_hypotheses(parsed2, tier=tier, verified_facts=verified_facts),
                resp2,
            )
        except (ValidationError, ValueError) as second_err:
            raise LLMError(
                f"Refutation failed validation twice on query {query!r}. "
                f"First error: {first_err}. Second error: {second_err}."
            ) from second_err

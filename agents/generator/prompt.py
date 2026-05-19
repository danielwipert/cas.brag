"""Block 13: Generator Agent — answer-generation prompt.

The Generator is the last big LLM stage in BRAG. It receives the
verified evidence set, the optional RefutationReport, the
DegradationLevel, and the orchestrator's adversarially_probed flag,
and emits a populated ``AnswerSchema``.

Per Chorus Principle 7, the Generator MUST use a different model
family from the Planner (Llama 3.3 70B), the Verifier (Qwen2.5 72B),
the Refutation Agent (Mistral Large 2411), and the embedder (BAAI
bge). We use DeepSeek-Chat — the same model the Block 5 prose
extractor uses, but a different agent role.

# Division of labor between prompt and code

The Generator LLM produces only the natural-language parts of the
answer:

  - ``answer_text``   — prose synthesis the user reads
  - ``claims[]``      — per-claim citation, with claim_type

Everything else on the AnswerSchema is mechanically derived from
inputs by ``generate_answer``:

  - ``disclosed_refutations`` is built from the RefutationReport (one
    entry per hypothesis with verdict ≠ unrefuted)
  - ``disclosed_gaps`` and ``disclosed_contradictions`` come from the
    orchestrator's caller-supplied lists
  - ``adversarially_probed`` and ``degradation_level`` propagate
    verbatim from inputs

This separation prevents the LLM from "forgetting" to declare a
strong refutation: the disclosure list is computed deterministically.
The Output Governance gate (Block 14) then verifies the answer_text
actually surfaces those refutations in prose.

# Numerical fidelity (the spec's load-bearing trust property)

Every grounded numerical claim in ``answer_text`` MUST reproduce the
cited fact's value, unit, and period verbatim. No rounding, no
paraphrasing, no unit conversion unless the claim is explicitly
labeled as ``derived`` and the derivation is shown inline (e.g.,
"operating margin was 12.9% — operating income of $2.60B divided by
revenue of $20.16B"). The prompt enforces this; Output Governance
verifies it by regex against the cited fact's value/unit/period.
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from agents.llm_client import LLMError, LLMResponse, OpenRouterClient
from schemas.enums import (
    ClaimType,
    DegradationLevel,
    RefutationVerdict,
)
from schemas.records import (
    AnswerClaim,
    AnswerSchema,
    DisclosedContradiction,
    DisclosedGap,
    DisclosedRefutation,
    FactRecord,
    RefutationReport,
)


# DeepSeek-Chat is the Chorus Principle 7 pick — different family from
# Planner (Llama), Verifier (Qwen), and Refutation (Mistral). The same
# model serves Block 5's prose extractor; the agent role is what
# differs. Fallback is Mistral Large 2411 — Chorus-acceptable when
# DeepSeek is unreachable since the Refutation Agent will have already
# logged its model and we know its behaviour on these prompts.
GENERATOR_MODEL = "deepseek/deepseek-chat"
GENERATOR_MODEL_FALLBACK = "mistralai/mistral-large-2411"

# Defense in depth — same as the Planner / Verifier / Refutation; some
# upstream providers behave poorly on JSON-mode requests.
_PROVIDER_PREFS: dict[str, Any] = {"provider": {"ignore": ["Venice"]}}


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


SYSTEM_PROMPT = """\
You are the Generator Agent in BRAG. You receive a verified evidence
set (Netflix-asserted facts that the Verifier has already judged as
supporting the user's query) plus an optional Refutation Report (the
Refutation Agent's findings about counter-evidence in the corpus).
You synthesize a faithful, citation-clean answer for the user.

You do NOT decide whether a fact is good evidence — the Verifier did
that. You do NOT decide whether a refutation is strong or weak — the
Refutation Agent did that. You write the prose answer the user reads.

# Output

Return a single JSON object (no markdown, no prose preamble) with
EXACTLY these fields:

{
  "answer_text": "<the prose answer the user reads>",
  "claims": [
    {
      "claim_text": "<one sentence stating a single claim>",
      "source_ids": ["<fact_id from the verified set>", "..."],
      "claim_type": "grounded" | "derived" | "interpretive"
    }
  ]
}

Do NOT emit disclosed_gaps, disclosed_contradictions,
disclosed_refutations, adversarially_probed, or degradation_level —
those are populated by code from caller-supplied inputs. If you
include them, they will be discarded.

# Numerical fidelity (LOAD-BEARING)

Every numerical assertion in ``answer_text`` MUST reproduce the
cited fact's value, unit, and period exactly as the fact states them.
This is the spec's central trust promise — a cited number is the
cited number, not a friendly paraphrase.

  Allowed:
    Fact:      "Netflix's Q2 2023 revenue was $8,187 million."
    Answer:    "Netflix's Q2 2023 revenue was $8,187 million."

  Allowed (same number, same unit):
    Fact:      value=8187.0, unit="USD millions", period="2023Q2"
    Answer:    "$8,187 million in Q2 2023"

  FORBIDDEN — rounding:
    Fact:      "Netflix's Q2 2023 revenue was $8,187 million."
    Answer:    "Netflix's Q2 2023 revenue was about $8.2 billion."
    Answer:    "Netflix's Q2 2023 revenue was roughly $8B."

  FORBIDDEN — paraphrasing:
    Fact:      "Netflix's Q2 2023 revenue was $8,187 million."
    Answer:    "Netflix booked just over $8 billion in Q2 2023 revenue."

  FORBIDDEN — silent unit conversion:
    Fact:      value=8187.0, unit="USD millions"
    Answer:    "$8.187 billion in revenue"   (units changed without note)

  Allowed — DERIVED with derivation shown inline AND
  claim_type=derived on the matching claim:
    Facts:     operating income $2.60 billion (FY2019),
               revenue $20.16 billion (FY2019)
    Answer:    "Netflix's FY2019 operating margin was 12.9% — operating
                income of $2.60 billion divided by revenue of $20.16
                billion."
    Claim:     {claim_text: "Netflix's FY2019 operating margin was 12.9%
                (operating income $2.60B / revenue $20.16B)",
                claim_type: "derived",
                source_ids: [<op income fact_id>, <revenue fact_id>]}

Periods and unit labels follow the same rule. If a fact says "fiscal
year 2019" you may write "FY2019" (canonical form is interchangeable
with the spelled-out form), but you may not change "FY2019" to "the
year ending December 2019" if the fact doesn't say so.

# Claim citation

Every claim in ``claims[]`` MUST carry one or more ``source_ids``
referencing real ``fact_id`` values from the verified evidence set
shown in the user message. NEVER invent a fact_id. If you can't
cite a claim to a verified fact, the claim doesn't go in claims[];
either remove the assertion from answer_text or mark it
``interpretive`` (interpretive claims still need cited source facts
they're interpreting).

claim_type values:

  grounded     — directly supported by one or more cited facts.
                 The claim_text matches the facts' substance.
  derived      — computed from cited facts (ratios, deltas, sums).
                 The derivation MUST be shown in answer_text.
                 source_ids names the facts whose values feed the
                 computation.
  interpretive — a characterization or summary judgment about cited
                 facts (e.g., "growth has accelerated"). Use
                 sparingly; the spec prefers grounded.

# Refutation handling

If a Refutation Report is provided in the user message:

  - For every hypothesis with verdict == "unrefuted", do NOTHING —
    no disclosure needed in your output.

  - For every hypothesis with verdict == "weakly_refuted", weave the
    weak refutation into ``answer_text`` inline. Example:
      "Netflix attributed Q3 2023 revenue growth primarily to price
       increases; a later Q4 letter notes that paid sharing
       enforcement was also a meaningful driver during the same
       period."
    Cite both the targeted fact and the refuting evidence's fact_id
    in the claim(s) that mention this. The disclosed_refutations[]
    entry is added by code; you handle the prose.

  - For every hypothesis with verdict == "strongly_refuted", surface
    the contradiction as STRUCTURED DISAGREEMENT. The answer_text
    MUST:
      (a) Name both positions explicitly.
      (b) Show both ASSERTION DATES — when Netflix said position A
          and when Netflix said position B. Use the assertion_date
          field on each fact verbatim.
      (c) Make the temporal evolution legible: "In <date1> Netflix
          said X; in <date2> Netflix said Y."
      (d) Not synthesize a single position that hides the conflict.
          Both positions stand on their own.
    Example (strong refutation, Partial output):
      "Netflix's 2018 position was that it had no plans to add
       advertising on its service (Q3 2018 shareholder letter,
       2018-10-16). In Q4 2022 (2023-01-26) Netflix announced an
       ad-supported tier, reversing that position."

  - Strong refutations that the REFUTATION LOOP RESOLVED show up in
    the verified set as additional facts (the refuting evidence
    became verified). In that case the degradation_level is Normal
    and you weave a temporal-evolution narrative in answer_text
    similar to (c) above, but framed as Netflix's evolution rather
    than as structured disagreement.

# Degradation handling

The degradation_level is supplied in the user message and propagated
by code to the AnswerSchema. Your answer_text MUST match the level:

  Normal              Comprehensive answer over the verified set.
                      Disclose refutations per the rules above.

  Partial             Disclose what's missing/unresolved. Be explicit
                      about gaps and strong refutations. Do not
                      pretend coverage is complete.

  Clarification Req.  You should not be invoked under CR — the
                      orchestrator bypasses Generator. If you are
                      invoked, ask the user to narrow the question
                      and explain what's missing.

  Hard Halt           You should never be invoked under Hard Halt.
                      If you are, emit a one-sentence acknowledgement
                      and an empty claims[] list.

# Tone

Direct, factual, financial-reporting register. No hedging beyond
what the evidence requires. No salesy language. Numbers and dates
land before adjectives.

Return ONLY the JSON object — no prose, no markdown fences.
"""


def _format_verified_fact(fact: FactRecord, idx: int) -> str:
    """Compact representation of one verified fact for the prompt.
    Mirrors the Refutation Agent's formatter so the LLM sees a
    consistent fact shape across stages."""
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
    if len(claim) > 500:
        claim = claim[:500] + "…"
    parts.append(f'        claim: "{claim}"')
    anchor = fact.verbatim_anchor.replace("\n", " ").strip()
    if len(anchor) > 300:
        anchor = anchor[:300] + "…"
    parts.append(f'        verbatim_anchor: "{anchor}"')
    return "\n".join(parts)


def _format_refutation_report(
    report: RefutationReport,
    verified_facts: list[FactRecord],
) -> str:
    """Render the refutation hypotheses the answer_text must
    incorporate. Includes targeted-claim context and refuting-fact
    assertion_dates so the LLM has every datum needed to weave the
    structured disagreement."""
    if not report.hypotheses:
        return "  (no hypotheses generated)"
    fact_by_id = {f.fact_id: f for f in verified_facts}
    lines = []
    for h in report.hypotheses:
        targeted = fact_by_id.get(h.targets_claim_id)
        targeted_date = (
            targeted.assertion_date.isoformat() if targeted is not None
            else "(targeted fact not in verified set)"
        )
        lines.append(f"  - hypothesis_id={h.hypothesis_id}")
        lines.append(f"    strategy={h.strategy.value}")
        lines.append(f"    verdict={h.refutation_verdict.value}")
        lines.append(f"    targets_claim_id={h.targets_claim_id} "
                     f"(targeted_assertion_date={targeted_date})")
        lines.append(f"    hypothesis_text: {h.hypothesis_text}")
        lines.append(f"    rationale: {h.rationale}")
        if h.evidence_ids:
            lines.append(f"    refuting_evidence_ids: {h.evidence_ids}")
    return "\n".join(lines)


def _build_user_message(
    original_query: str,
    verified_facts: list[FactRecord],
    refutation_report: RefutationReport | None,
    degradation_level: DegradationLevel,
    *,
    disclosed_gaps: list[DisclosedGap],
    disclosed_contradictions: list[DisclosedContradiction],
    prior_error: str | None = None,
) -> str:
    lines = [
        f"ORIGINAL QUERY: {original_query}",
        f"DEGRADATION LEVEL: {degradation_level.name}",
        "",
        f"VERIFIED EVIDENCE SET ({len(verified_facts)} fact"
        f"{'' if len(verified_facts) == 1 else 's'}):",
    ]
    for i, f in enumerate(verified_facts, start=1):
        lines.append(_format_verified_fact(f, i))
    lines.append("")
    if disclosed_gaps:
        lines.append(f"DISCLOSED GAPS ({len(disclosed_gaps)}):")
        for g in disclosed_gaps:
            lines.append(f"  - slot {g.slot_id}: {g.gap_description}")
        lines.append("")
    if disclosed_contradictions:
        lines.append(f"DISCLOSED CONTRADICTIONS ({len(disclosed_contradictions)}):")
        for c in disclosed_contradictions:
            lines.append(f"  - {c.description}  conflicting_ids={c.conflicting_ids}")
        lines.append("")
    if refutation_report is not None:
        lines.append(
            f"REFUTATION REPORT (overall={report_overall_str(refutation_report)}, "
            f"loop_reentry={refutation_report.triggered_loop_reentry}):"
        )
        lines.append(_format_refutation_report(refutation_report, verified_facts))
        lines.append("")
    else:
        lines.append("REFUTATION REPORT: (none — refutation bypassed or unavailable)")
        lines.append("")
    lines.append(
        "Generate the answer per the schema and rules in the system "
        "prompt. Reproduce numerical values verbatim. Disclose all "
        "non-unrefuted hypotheses in answer_text per the refutation "
        "handling rules. Return JSON only."
    )
    if prior_error is not None:
        lines.append("")
        lines.append(
            "NOTE: your previous response failed validation. Error: "
            f"{prior_error}. Fix the issue and emit valid JSON."
        )
    return "\n".join(lines)


def report_overall_str(report: RefutationReport) -> str:
    """Module-level so the user message builder can call it. Just
    exposes the enum value as a string."""
    return report.overall_verdict.value


def _parse_generator_json(content: str | None) -> Any:
    """Tolerant JSON parser. Strips ``` fences, falls back to the
    first JSON object substring, raises LLMError if nothing parses."""
    if not content:
        raise LLMError(
            "Generator response had empty content. The upstream "
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
                f"Generator response had unparseable JSON object: {e}. "
                f"Snippet: {om.group(0)[:400]!r}"
            ) from e
    raise LLMError(f"Generator response is not JSON: {content[:400]!r}")


def _build_disclosed_refutations(
    report: RefutationReport | None,
) -> list[DisclosedRefutation]:
    """Mechanically derive the disclosure list from the refutation
    report. One entry per hypothesis with verdict != unrefuted. The
    LLM does not produce this list — Block 14's Output Governance
    will check that the answer_text honors the disclosures, but the
    structured field is deterministic from inputs."""
    if report is None:
        return []
    out: list[DisclosedRefutation] = []
    for h in report.hypotheses:
        if h.refutation_verdict == RefutationVerdict.unrefuted:
            continue
        out.append(DisclosedRefutation(
            targets_claim_id=h.targets_claim_id,
            refuting_evidence_ids=list(h.evidence_ids),
            refutation_verdict=h.refutation_verdict,
            strategy=h.strategy,
        ))
    return out


def _coerce_claims(
    raw_claims: Any,
    verified_facts: list[FactRecord],
) -> list[AnswerClaim]:
    """Validate the LLM's claims[] list. Each claim must have at
    least one source_id that resolves to a verified fact_id; any
    claim with all unknown source_ids is rejected (fabricated
    citation). Empty claims[] is permitted on Clarification Request /
    Hard Halt paths."""
    if not isinstance(raw_claims, list):
        raise ValueError(
            f"Generator claims field is not a list (got "
            f"{type(raw_claims).__name__})."
        )
    valid_ids = {f.fact_id for f in verified_facts}
    out: list[AnswerClaim] = []
    for i, item in enumerate(raw_claims, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Claim #{i} is not an object.")
        source_ids = item.get("source_ids") or []
        if not isinstance(source_ids, list):
            raise ValueError(
                f"Claim #{i} source_ids is not a list (got "
                f"{type(source_ids).__name__})."
            )
        # Drop any fabricated fact_ids. If NO real ids remain, reject
        # the claim — citing nothing is worse than not making the
        # claim at all.
        cleaned_ids = [sid for sid in source_ids if isinstance(sid, str) and sid in valid_ids]
        if not cleaned_ids and source_ids:
            raise ValueError(
                f"Claim #{i} cites no real fact_ids "
                f"(provided: {source_ids}; valid: {sorted(valid_ids)[:5]}...)."
            )
        try:
            out.append(AnswerClaim.model_validate({
                "claim_text": item.get("claim_text", ""),
                "source_ids": cleaned_ids,
                "claim_type": item.get("claim_type", "grounded"),
            }))
        except ValidationError as e:
            raise ValueError(f"Claim #{i} failed schema validation: {e}") from e
    return out


def generate_answer(
    *,
    original_query: str,
    verified_facts: list[FactRecord],
    refutation_report: RefutationReport | None = None,
    degradation_level: DegradationLevel = DegradationLevel.NORMAL,
    adversarially_probed: bool = False,
    disclosed_gaps: list[DisclosedGap] | None = None,
    disclosed_contradictions: list[DisclosedContradiction] | None = None,
    client: OpenRouterClient | None = None,
    model: str | None = None,
    max_tokens: int = 2000,
) -> tuple[AnswerSchema, LLMResponse]:
    """Generate an ``AnswerSchema`` for ``original_query`` over
    ``verified_facts``.

    The Generator LLM produces only ``answer_text`` and ``claims[]``.
    The disclosure lists, adversarially_probed flag, and
    degradation_level on the returned schema are populated from
    inputs — the LLM cannot omit a strong refutation by accident.

    Retries once on parse / validation failure with the error
    appended to the user message (mirrors the Planner pattern). A
    second failure raises LLMError."""
    if degradation_level == DegradationLevel.HARD_HALT:
        # Caller bypassed; produce a stub.
        return (
            AnswerSchema(
                answer_text="(no answer — the run halted before generation).",
                claims=[],
                disclosed_gaps=list(disclosed_gaps or []),
                disclosed_contradictions=list(disclosed_contradictions or []),
                disclosed_refutations=_build_disclosed_refutations(refutation_report),
                adversarially_probed=adversarially_probed,
                degradation_level=degradation_level,
            ),
            LLMResponse(content="", model="", finish_reason="", usage={}, raw={}),
        )
    if not verified_facts and degradation_level not in (
        DegradationLevel.CLARIFICATION_REQUEST,
    ):
        raise ValueError(
            "generate_answer requires at least one verified fact unless "
            "degradation_level is Clarification Request or Hard Halt."
        )

    if client is None:
        client = OpenRouterClient(default_model=model or GENERATOR_MODEL)
    chosen_model = model or GENERATOR_MODEL

    gaps = list(disclosed_gaps or [])
    contradictions = list(disclosed_contradictions or [])

    def _call(prior_error: str | None) -> tuple[Any, LLMResponse]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_message(
                    original_query=original_query,
                    verified_facts=verified_facts,
                    refutation_report=refutation_report,
                    degradation_level=degradation_level,
                    disclosed_gaps=gaps,
                    disclosed_contradictions=contradictions,
                    prior_error=prior_error,
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
        parsed = _parse_generator_json(resp.content)
        return parsed, resp

    def _build_schema(payload: Any) -> AnswerSchema:
        if not isinstance(payload, dict):
            raise ValueError(
                f"Generator returned non-object JSON: {type(payload).__name__}"
            )
        answer_text = payload.get("answer_text")
        if not isinstance(answer_text, str) or not answer_text.strip():
            raise ValueError("Generator missing or empty answer_text field.")
        claims = _coerce_claims(payload.get("claims") or [], verified_facts)
        return AnswerSchema(
            answer_text=answer_text.strip(),
            claims=claims,
            disclosed_gaps=gaps,
            disclosed_contradictions=contradictions,
            disclosed_refutations=_build_disclosed_refutations(refutation_report),
            adversarially_probed=adversarially_probed,
            degradation_level=degradation_level,
        )

    parsed, resp = _call(prior_error=None)
    try:
        return _build_schema(parsed), resp
    except (ValidationError, ValueError) as first_err:
        parsed2, resp2 = _call(prior_error=str(first_err))
        try:
            return _build_schema(parsed2), resp2
        except (ValidationError, ValueError) as second_err:
            raise LLMError(
                f"Generator failed validation twice on query "
                f"{original_query!r}. First error: {first_err}. "
                f"Second error: {second_err}."
            ) from second_err

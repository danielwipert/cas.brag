"""Block 11: Refutation Agent integration.

Wraps the Block 10 hypothesis prompt with hypothesis-driven retrieval,
a two-track classifier (deterministic numerical-automatic + LLM
judgment for prose strategies), and a result envelope the orchestrator
uses to decide PASS / DISCLOSE / LOOP / DEGRADE.

The agent does NOT decide loop-vs-partial — that depends on iteration
count and zero-progress state owned by the orchestrator. The agent
emits a per-hypothesis verdict and an aggregate signal; the
orchestrator translates the signal into a pipeline action.

Architecture:

  1. ``generate_hypotheses`` (Block 10) — 1/2/3 strategy-aware
     counter-claims for the verified evidence set.
  2. For each hypothesis, build a synthetic ``EvidenceSlot`` and call
     the Block 8 ``retrieve()`` with ``pass_origin=refutation_probe``.
  3. Classify each hypothesis:
     a. ``restated_value`` / ``revised_value`` → numerical-automatic
        deterministic check (concept_tag/metric + period + later
        assertion_date + different value → ``strongly_refuted``).
     b. all other strategies → LLM judgment via Mistral Large 2411
        (same family as the hypothesis prompt; Chorus Principle 7
        keeps the Verifier (Qwen) out of the refutation pipeline).
  4. Aggregate to one of:
       ``answer_strengthened``       — all unrefuted or only weak
       ``refutation_to_loop``        — ≥1 strongly_refuted, iters left
       ``refutation_to_partial``     — ≥1 strongly_refuted, no iters

Bypass conditions (handled by orchestrator, not here):
  Partial degradation, Clarification Request, Hard Halt, empty
  verified set.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from agents.llm_client import LLMError, LLMResponse, OpenRouterClient
from agents.refutation.prompt import (
    REFUTATION_MODEL,
    generate_hypotheses,
)
from agents.retriever.retriever import retrieve
from pipeline.memory_ledger import Ledger
from schemas.enums import (
    CandidateSource,
    ComplexityTier,
    EvidenceType,
    FactType,
    PassOrigin,
    RefutationOverallVerdict,
    RefutationStrategy,
    RefutationVerdict,
    TargetLayer,
)
from schemas.records import (
    EvidenceSlot,
    FactRecord,
    RefutationHypothesis,
    RefutationReport,
    RetrievalCandidate,
    RetrievalRecord,
)


# Same model as the hypothesis prompt (Mistral family) for Chorus
# Principle 7 diversity from the Verifier (Qwen) and Planner (Llama).
CLASSIFIER_MODEL = REFUTATION_MODEL

# Spec-mandated degraded fallback when Mistral is unreachable
# (build plan §Block 11 task list, spec §3.6 "degraded refutation
# pass using Llama 3.3 70B"). The fallback is intentionally
# cross-family — when this fires, we sacrifice Chorus Principle 7
# diversity to keep the refutation pipeline online; the run is
# flagged so the operator can see Mistral was unavailable.
REFUTATION_FALLBACK_MODEL = "meta-llama/llama-3.3-70b-instruct"

_PROVIDER_PREFS: dict[str, Any] = {"provider": {"ignore": ["Venice"]}}


_FALLBACK_FLAG_KEY = "refutation_unavailable_fallback_invoked"


# Strategies whose strong-refutation gate is temporal — i.e., the
# refuting fact must have a LATER assertion_date than the targeted
# claim. (All seven strategies are temporal in practice, but
# alternative_cause can be contemporaneous so it's the exception.)
_TEMPORAL_STRATEGIES: frozenset[RefutationStrategy] = frozenset({
    RefutationStrategy.restated_value,
    RefutationStrategy.revised_value,
    RefutationStrategy.guidance_vs_actual,
    RefutationStrategy.later_reversal,
    RefutationStrategy.materialization,
    RefutationStrategy.policy_change,
})


# Strategies that use the deterministic numerical-automatic path.
_NUMERICAL_AUTO_STRATEGIES: frozenset[RefutationStrategy] = frozenset({
    RefutationStrategy.restated_value,
    RefutationStrategy.revised_value,
})


# fact_type → evidence_type used when building the synthetic probe slot.
_PROBE_EVIDENCE_TYPE: dict[RefutationStrategy, EvidenceType] = {
    RefutationStrategy.restated_value: EvidenceType.specific_metric,
    RefutationStrategy.revised_value: EvidenceType.specific_metric,
    RefutationStrategy.guidance_vs_actual: EvidenceType.specific_metric,
    RefutationStrategy.later_reversal: EvidenceType.strategic_position,
    RefutationStrategy.alternative_cause: EvidenceType.causal_explanation,
    RefutationStrategy.materialization: EvidenceType.risk_disclosure,
    RefutationStrategy.policy_change: EvidenceType.accounting_policy,
}


# ---------------------------------------------------------------------------
# FactRecord lookup (the corpus's full-shape source of truth)
# ---------------------------------------------------------------------------


_FACT_INDEX: dict[str, FactRecord] | None = None
_FACT_INDEX_PATHS = [
    Path("data/fact_store/xbrl_facts.jsonl"),
    Path("data/fact_store/prose_facts.jsonl"),
]


def _load_fact_index() -> dict[str, FactRecord]:
    """Build (and cache) a fact_id → FactRecord lookup table from the
    JSONL fact files. Chroma metadata is intentionally narrower than
    FactRecord, so the JSONL files remain the source of truth for the
    asserter / assertion_date / verbatim_anchor / confidence fields
    the Refutation Agent needs."""
    global _FACT_INDEX
    if _FACT_INDEX is not None:
        return _FACT_INDEX
    index: dict[str, FactRecord] = {}
    for path in _FACT_INDEX_PATHS:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = FactRecord.model_validate_json(line)
                except Exception:
                    # Skip malformed rows — the ingestion phase is
                    # supposed to have validated already.
                    continue
                index[rec.fact_id] = rec
    _FACT_INDEX = index
    return index


def lookup_facts(fact_ids: Iterable[str]) -> dict[str, FactRecord]:
    """Bulk-lookup helper used by the orchestrator and the agent
    itself. Missing IDs are silently omitted (chunks won't resolve)."""
    index = _load_fact_index()
    return {fid: index[fid] for fid in fact_ids if fid in index}


# ---------------------------------------------------------------------------
# Probe slot construction
# ---------------------------------------------------------------------------


_KEY_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]{2,}")
_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "have", "has", "had", "in", "is", "it", "its", "of", "on", "or",
    "than", "that", "the", "their", "they", "this", "to", "was", "were",
    "with", "would", "could", "should", "will", "would", "if", "rather",
    "instead", "while", "into", "over", "under", "primary", "primarily",
    "later", "earlier", "earlier", "letter", "report", "filing",
})


def _extract_key_terms(text: str, max_terms: int = 8) -> list[str]:
    """Pull content-bearing terms from the hypothesis text. Used as
    the BM25 query for the probe retrieval."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _KEY_TERM_RE.finditer(text):
        tok = m.group(0).lower()
        if tok in _STOP_WORDS or tok in seen:
            continue
        seen.add(tok)
        out.append(m.group(0))
        if len(out) >= max_terms:
            break
    return out


def _build_probe_slot(
    h: RefutationHypothesis,
    targeted_fact: FactRecord,
) -> EvidenceSlot:
    """Synthetic slot used to drive hypothesis-driven retrieval. We
    deliberately DO NOT carry the targeted claim's period_filter — the
    whole point of refutation retrieval is to find evidence OUTSIDE
    the original period (later filings, later transcripts)."""
    return EvidenceSlot(
        slot_id=f"refutation::{h.hypothesis_id}",
        sub_question=h.hypothesis_text,
        evidence_type=_PROBE_EVIDENCE_TYPE[h.strategy],
        target_layer=TargetLayer.both,
        period_filter=None,
        key_terms=_extract_key_terms(h.hypothesis_text)
        or _extract_key_terms(targeted_fact.claim),
        coverage_threshold=0.50,
    )


# ---------------------------------------------------------------------------
# Numerical-automatic classifier (deterministic)
# ---------------------------------------------------------------------------


def _normalize_concept(tag: str | None) -> str | None:
    if not tag:
        return None
    return tag.lower().strip()


def _classify_numerical_automatic(
    targeted: FactRecord,
    candidate_facts: list[FactRecord],
) -> tuple[RefutationVerdict, list[str]]:
    """Spec §3.6 v3 strong rule (deterministic): for financial_metric
    or operational_metric claims targeted by restated_value /
    revised_value, a direct value mismatch on the same concept_tag and
    period from a fact with a later assertion_date is automatically
    ``strongly_refuted``. No LLM interpretation needed.

    Returns ``(verdict, evidence_ids)``. If no automatic refutation is
    found, returns ``(unrefuted, [])`` and the caller falls back to
    LLM judgment (which can still find a weak refutation)."""
    tgt_concept = _normalize_concept(targeted.concept_tag)
    tgt_period = targeted.period
    tgt_value = targeted.value
    tgt_date = targeted.assertion_date

    refuting_ids: list[str] = []
    for cand in candidate_facts:
        if cand.fact_id == targeted.fact_id:
            continue
        if cand.assertion_date <= tgt_date:
            continue
        if cand.period != tgt_period:
            continue
        # Concept match: tag equality if both have one, otherwise fall
        # back to claim-text substring of a metric phrase. The latter
        # rescues operational_metric facts (which have no XBRL tag).
        cand_concept = _normalize_concept(cand.concept_tag)
        concept_ok = (
            tgt_concept is not None
            and cand_concept is not None
            and tgt_concept == cand_concept
        )
        if not concept_ok and tgt_concept is None and cand_concept is None:
            # Both prose-based — match on shared metric phrase in the
            # claim text. This is loose but the period+date gates above
            # already prevent most false positives.
            concept_ok = _shared_metric_phrase(targeted.claim, cand.claim)
        if not concept_ok:
            continue
        if cand.value is None or tgt_value is None:
            continue
        if cand.value == tgt_value:
            continue
        refuting_ids.append(cand.fact_id)

    if refuting_ids:
        return (RefutationVerdict.strongly_refuted, refuting_ids)
    return (RefutationVerdict.unrefuted, [])


_METRIC_PHRASE_RE = re.compile(
    r"\b(?:paid\s+memberships?|paid\s+subscribers?|paid\s+net\s+adds?|"
    r"paid\s+net\s+additions?|operating\s+income|operating\s+margin|"
    r"net\s+income|revenues?|gross\s+profit|free\s+cash\s+flow|"
    r"content\s+amortization|content\s+assets|stockholders'?\s+equity)\b",
    re.IGNORECASE,
)


def _shared_metric_phrase(a: str, b: str) -> bool:
    aset = {m.group(0).lower() for m in _METRIC_PHRASE_RE.finditer(a)}
    bset = {m.group(0).lower() for m in _METRIC_PHRASE_RE.finditer(b)}
    return bool(aset & bset)


# ---------------------------------------------------------------------------
# LLM-judgment classifier (prose strategies)
# ---------------------------------------------------------------------------


_CLASSIFIER_SYSTEM_PROMPT = """\
You are the Refutation Classifier in BRAG. You evaluate whether a set
of retrieved candidate facts (and chunks) refutes a single verified
claim under a named refutation strategy.

You do NOT generate hypotheses or run retrieval. You render one
verdict per call.

# Verdict ladder

  strongly_refuted   At least one candidate directly contradicts the
                     targeted claim AND every gate below is satisfied:
                       - asserter is identifiable (a named entity,
                         e.g. "Netflix", "Reed Hastings", "Greg
                         Peters" — not "(unknown)" or empty)
                       - confidence ≥ 0.80
                       - assertion_date is LATER than the targeted
                         claim's assertion_date (this gate applies to
                         every strategy here except alternative_cause,
                         which can be contemporaneous)
                       - the candidate has a non-empty verbatim_anchor

  weakly_refuted     A candidate suggests a contradiction but FAILS
                     one or more gates (low confidence, weak/missing
                     asserter, contemporaneous date when a later date
                     was required, OR the candidate is a chunk rather
                     than an extracted fact).

  unrefuted          No candidate suggests a contradiction at all.

# Strategy-specific contradiction criteria

  later_reversal       Candidate explicitly reverses, abandons, or
                       materially qualifies the targeted strategic
                       position. Paraphrases of the original
                       position are NOT refutations.

  alternative_cause    Candidate names a DIFFERENT primary cause for
                       the same outcome named in the targeted claim.

  materialization      Candidate describes the disclosed risk as
                       having materialized — i.e., happened, is
                       happening, or is being actively addressed —
                       rather than remaining a hypothetical future
                       risk.

  policy_change        Candidate describes the targeted accounting
                       policy being CHANGED in a later period. A
                       restated value alone is not a policy change.

  guidance_vs_actual   Candidate reports the actual subsequent
                       result for the period the targeted guidance
                       covered, and that result CONTRADICTS the
                       guidance. Note: a candidate stating "we
                       achieved the guidance" is NOT a refutation.

# Output

Return a single JSON object (no markdown, no prose):

{
  "verdict": "strongly_refuted" | "weakly_refuted" | "unrefuted",
  "evidence_ids": ["candidate_id_1", "candidate_id_2"],
  "reasoning": "<one sentence naming the contradiction and which gate(s) it cleared or failed>"
}

evidence_ids lists candidates that support your verdict. For an
unrefuted verdict, evidence_ids is an empty list.
"""


@dataclass(frozen=True)
class _ClassifiedCandidate:
    """Candidate enriched with the FactRecord fields the classifier
    needs to apply the strong-refutation gates."""
    candidate_id: str
    source: CandidateSource
    rrf_score: float
    text: str
    asserter: str | None
    assertion_date: str | None
    confidence: float | None
    fact_type: str | None
    period: str | None
    has_verbatim_anchor: bool


def _enrich_candidates(
    record: RetrievalRecord,
) -> tuple[list[_ClassifiedCandidate], dict[str, FactRecord]]:
    """Resolve each candidate to its full FactRecord (when it's a
    fact) and produce a _ClassifiedCandidate summary for the LLM
    prompt. The dict in the return is fact_id → FactRecord for callers
    that need the full object (e.g. classifier post-processing)."""
    index = _load_fact_index()
    out: list[_ClassifiedCandidate] = []
    fact_objs: dict[str, FactRecord] = {}
    for c in record.candidates:
        if c.source == CandidateSource.fact:
            rec = index.get(c.candidate_id)
            if rec is None:
                # Fact ID we don't have full-shape data for — treat as
                # a chunk with limited metadata.
                out.append(
                    _ClassifiedCandidate(
                        candidate_id=c.candidate_id,
                        source=c.source,
                        rrf_score=c.rrf_score,
                        text="(fact body unavailable)",
                        asserter=None,
                        assertion_date=None,
                        confidence=None,
                        fact_type=None,
                        period=None,
                        has_verbatim_anchor=False,
                    )
                )
                continue
            fact_objs[c.candidate_id] = rec
            out.append(
                _ClassifiedCandidate(
                    candidate_id=c.candidate_id,
                    source=c.source,
                    rrf_score=c.rrf_score,
                    text=rec.claim,
                    asserter=rec.asserter,
                    assertion_date=rec.assertion_date.isoformat(),
                    confidence=rec.confidence,
                    fact_type=rec.fact_type.value,
                    period=rec.period,
                    has_verbatim_anchor=bool(rec.verbatim_anchor),
                )
            )
        else:
            # Chunk — minimal metadata. Always at most weakly refuting.
            out.append(
                _ClassifiedCandidate(
                    candidate_id=c.candidate_id,
                    source=c.source,
                    rrf_score=c.rrf_score,
                    text="",
                    asserter=None,
                    assertion_date=None,
                    confidence=None,
                    fact_type=None,
                    period=None,
                    has_verbatim_anchor=False,
                )
            )
    return out, fact_objs


def _format_classifier_candidate(c: _ClassifiedCandidate, idx: int) -> str:
    parts = [
        f"  [C{idx}] id={c.candidate_id}",
        f"        source={c.source.value} rrf={c.rrf_score:.5f}",
    ]
    if c.fact_type:
        parts.append(f"        fact_type={c.fact_type}")
    if c.asserter:
        parts.append(f"        asserter={c.asserter}")
    if c.assertion_date:
        parts.append(f"        assertion_date={c.assertion_date}")
    if c.period:
        parts.append(f"        period={c.period}")
    if c.confidence is not None:
        parts.append(f"        confidence={c.confidence:.2f}")
    parts.append(f"        has_verbatim_anchor={c.has_verbatim_anchor}")
    text = c.text.replace("\n", " ").strip()
    if len(text) > 500:
        text = text[:500] + "…"
    if text:
        parts.append(f'        text: "{text}"')
    return "\n".join(parts)


def _build_classifier_user_message(
    h: RefutationHypothesis,
    targeted: FactRecord,
    candidates: list[_ClassifiedCandidate],
) -> str:
    lines = [
        "TARGETED CLAIM:",
        f"  fact_id: {targeted.fact_id}",
        f"  fact_type: {targeted.fact_type.value}",
        f"  asserter: {targeted.asserter}",
        f"  assertion_date: {targeted.assertion_date.isoformat()}",
        f"  period: {targeted.period or '(none)'}",
        f'  claim: "{targeted.claim}"',
        "",
        "REFUTATION HYPOTHESIS:",
        f"  strategy: {h.strategy.value}",
        f"  text: {h.hypothesis_text}",
        f"  rationale: {h.rationale}",
        "",
        f"RETRIEVED CANDIDATES ({len(candidates)}):",
    ]
    if not candidates:
        lines.append("  (no candidates returned by retrieval)")
    for idx, c in enumerate(candidates, start=1):
        lines.append(_format_classifier_candidate(c, idx))
    lines.append("")
    lines.append(
        "Render exactly one verdict per the schema in the system prompt."
    )
    return "\n".join(lines)


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_classifier_json(content: str | None) -> Any:
    if not content:
        raise LLMError("Classifier response had empty content.")
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
                f"Classifier JSON unparseable: {e}. "
                f"Snippet: {om.group(0)[:400]!r}"
            ) from e
    raise LLMError(f"Classifier response is not JSON: {content[:400]!r}")


def _classify_general(
    h: RefutationHypothesis,
    targeted: FactRecord,
    candidates: list[_ClassifiedCandidate],
    fact_objs: dict[str, FactRecord],
    *,
    client: OpenRouterClient,
    model: str,
) -> tuple[RefutationVerdict, list[str], str]:
    """LLM-judgment classifier for non-numerical strategies. Returns
    ``(verdict, evidence_ids, reasoning)``.

    Empty candidate sets short-circuit to ``unrefuted`` without an
    LLM call."""
    if not candidates:
        return (RefutationVerdict.unrefuted, [], "no candidates retrieved")

    messages = [
        {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_classifier_user_message(h, targeted, candidates),
        },
    ]
    resp = client.chat(
        messages,
        model=model,
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=800,
        extra=_PROVIDER_PREFS,
    )
    payload = _parse_classifier_json(resp.content)
    if not isinstance(payload, dict):
        raise LLMError(f"Classifier returned non-object JSON: {payload!r}")
    verdict_raw = payload.get("verdict")
    try:
        verdict = RefutationVerdict(verdict_raw)
    except ValueError as e:
        raise LLMError(
            f"Classifier emitted invalid verdict {verdict_raw!r}: {e}"
        ) from e

    raw_ids = payload.get("evidence_ids") or []
    if not isinstance(raw_ids, list):
        raise LLMError(
            f"Classifier evidence_ids is not a list: {raw_ids!r}"
        )
    valid_ids = {c.candidate_id for c in candidates}
    evidence_ids = [eid for eid in raw_ids if isinstance(eid, str) and eid in valid_ids]

    # Defense in depth: if the LLM declared strongly_refuted but no
    # evidence_id meets the deterministic temporal/identity gates,
    # downgrade to weakly_refuted. The LLM's verdict is the senior
    # input, but the gates are spec-mandated.
    if verdict == RefutationVerdict.strongly_refuted:
        gates_met = _verify_strong_gates(
            h.strategy, targeted, evidence_ids, fact_objs
        )
        if not gates_met:
            verdict = RefutationVerdict.weakly_refuted

    reasoning = str(payload.get("reasoning") or "")
    return (verdict, evidence_ids, reasoning)


def _verify_strong_gates(
    strategy: RefutationStrategy,
    targeted: FactRecord,
    evidence_ids: list[str],
    fact_objs: dict[str, FactRecord],
) -> bool:
    """True iff at least one evidence_id resolves to a FactRecord that
    satisfies every strong-refutation gate spec §3.6 lays out."""
    require_later_date = strategy in _TEMPORAL_STRATEGIES
    for eid in evidence_ids:
        rec = fact_objs.get(eid)
        if rec is None:
            continue
        if not rec.asserter or rec.asserter.strip().lower() in {"unknown", "(unknown)", ""}:
            continue
        if rec.confidence < 0.80:
            continue
        if not rec.verbatim_anchor:
            continue
        if require_later_date and rec.assertion_date <= targeted.assertion_date:
            continue
        return True
    return False


# ---------------------------------------------------------------------------
# Top-level agent
# ---------------------------------------------------------------------------


@dataclass
class RefutationAgentResult:
    """Envelope returned to the orchestrator. The ``report`` is the
    schema-valid summary; the auxiliary fields carry data the
    orchestrator needs to wire loop re-entry."""
    report: RefutationReport
    retrieval_records: list[RetrievalRecord] = field(default_factory=list)
    refuting_facts: dict[str, list[FactRecord]] = field(default_factory=dict)
    strongly_refuted: list[RefutationHypothesis] = field(default_factory=list)
    classifier_responses: list[LLMResponse] = field(default_factory=list)
    fallback_invoked: bool = False
    models_used: list[str] = field(default_factory=list)


def _try_with_fallback(
    label: str,
    primary_model: str,
    fallback_model: str,
    call_fn,
) -> tuple[Any, str, bool]:
    """Run ``call_fn(model)`` with ``primary_model`` first; on
    ``LLMError`` retry with ``fallback_model``. Returns
    ``(result, model_actually_used, fallback_invoked)``.

    Re-raises ``LLMError`` only if both attempts fail."""
    try:
        return call_fn(primary_model), primary_model, False
    except LLMError as primary_err:
        try:
            return call_fn(fallback_model), fallback_model, True
        except LLMError as fallback_err:
            raise LLMError(
                f"{label}: primary model {primary_model!r} and fallback "
                f"{fallback_model!r} both failed. "
                f"Primary error: {primary_err}. "
                f"Fallback error: {fallback_err}."
            ) from fallback_err


def run_refutation(
    *,
    run_id: str,
    query: str,
    complexity_tier: ComplexityTier,
    verified_facts: list[FactRecord],
    ledger: Ledger | None = None,
    iteration: int = 1,
    max_loop_iterations: int = 2,
    client: OpenRouterClient | None = None,
    refutation_model: str | None = None,
    classifier_model: str | None = None,
    excluded_ids: Iterable[str] = (),
) -> RefutationAgentResult:
    """Run the Refutation Agent over ``verified_facts``.

    The agent generates hypotheses (Block 10 prompt), runs hypothesis-
    driven retrieval for each, and classifies the findings. It does
    NOT decide loop-vs-partial — the caller passes ``iteration`` and
    ``max_loop_iterations`` so the agent can emit the right overall
    verdict, but the orchestrator owns the actual loop control.

    Caller obligations:
      * Pass an empty ``verified_facts`` list ONLY if you intend a no-op
        run (the agent will raise rather than fabricate hypotheses).
      * Exclude IDs from prior iterations via ``excluded_ids`` so the
        probe retrievals don't re-surface already-seen candidates.
      * Persist ``result.report.hypotheses`` to the Ledger if you want
        loop dedup.
    """
    if not verified_facts:
        raise ValueError(
            "run_refutation requires at least one verified fact. "
            "The orchestrator should bypass the agent under Partial / "
            "Clarification Request / Hard Halt degradation."
        )

    if client is None:
        client = OpenRouterClient(default_model=refutation_model or REFUTATION_MODEL)
    refute_model = refutation_model or REFUTATION_MODEL
    judge_model = classifier_model or CLASSIFIER_MODEL

    fallback_invoked = False
    models_used: list[str] = []

    # Stage 1: generate hypotheses via Block 10 prompt. If Mistral is
    # unreachable (429s past the client's retry budget, 5xx, etc.),
    # fall back to Llama 3.3 70B with the same prompt — spec §3.6
    # mandates this degraded path.
    already_tested = (
        ledger.refutation_hypothesis_texts() if ledger is not None else set()
    )

    def _gen_with_model(m: str):
        return generate_hypotheses(
            query, complexity_tier, verified_facts,
            client=client, model=m,
        )

    (gen_out, used, gen_fallback) = _try_with_fallback(
        "hypothesis_generation",
        refute_model,
        REFUTATION_FALLBACK_MODEL,
        _gen_with_model,
    )
    hypotheses, _gen_resp = gen_out
    models_used.append(used)
    fallback_invoked = fallback_invoked or gen_fallback
    # Drop duplicates against the ledger so loop iterations don't
    # waste API calls on the same hypothesis. If every hypothesis was
    # already tested, the report will reflect zero new hypotheses.
    if already_tested:
        hypotheses = [h for h in hypotheses if h.hypothesis_text not in already_tested]

    fact_by_id = {f.fact_id: f for f in verified_facts}
    retrieval_records: list[RetrievalRecord] = []
    refuting_facts: dict[str, list[FactRecord]] = {}
    classifier_responses: list[LLMResponse] = []
    strongly_refuted: list[RefutationHypothesis] = []
    resolved_hypotheses: list[RefutationHypothesis] = []

    excluded_set = set(excluded_ids)

    for h in hypotheses:
        targeted = fact_by_id[h.targets_claim_id]

        # Stage 2: probe retrieval.
        probe_slot = _build_probe_slot(h, targeted)
        # Always exclude the targeted fact itself from the probe — its
        # own claim shouldn't refute itself.
        probe_exclude = excluded_set | {targeted.fact_id}
        record = retrieve(
            probe_slot,
            complexity_tier=complexity_tier,
            iteration=iteration,
            pass_origin=PassOrigin.refutation_probe,
            excluded_ids=probe_exclude,
            retrieval_id=f"REF_{h.hypothesis_id}_iter{iteration}",
        )
        retrieval_records.append(record)

        # Stage 3a: numerical-automatic check (deterministic).
        candidates, fact_objs = _enrich_candidates(record)
        candidate_facts = list(fact_objs.values())
        verdict = RefutationVerdict.unrefuted
        evidence_ids: list[str] = []
        if h.strategy in _NUMERICAL_AUTO_STRATEGIES:
            verdict, evidence_ids = _classify_numerical_automatic(
                targeted, candidate_facts
            )

        # Stage 3b: LLM judgment if numerical-automatic didn't fire.
        # Same fallback as hypothesis generation — Mistral first,
        # Llama 3.3 70B if unreachable.
        if verdict == RefutationVerdict.unrefuted:
            def _classify_with_model(m: str):
                return _classify_general(
                    h, targeted, candidates, fact_objs,
                    client=client, model=m,
                )
            (cls_out, used_cls, cls_fallback) = _try_with_fallback(
                f"classifier::{h.hypothesis_id}",
                judge_model,
                REFUTATION_FALLBACK_MODEL,
                _classify_with_model,
            )
            verdict, evidence_ids, _reasoning = cls_out
            if used_cls not in models_used:
                models_used.append(used_cls)
            fallback_invoked = fallback_invoked or cls_fallback

        # Assemble the populated hypothesis. retrieval_record_id +
        # refutation_verdict + evidence_ids are filled by the agent;
        # the Block 10 prompt left them as defaults.
        resolved = h.model_copy(update={
            "retrieval_record_id": record.retrieval_id,
            "refutation_verdict": verdict,
            "evidence_ids": evidence_ids,
        })
        resolved_hypotheses.append(resolved)
        if ledger is not None:
            ledger.add_refutation_hypothesis(resolved)

        if verdict == RefutationVerdict.strongly_refuted:
            strongly_refuted.append(resolved)
            refuting_facts[resolved.hypothesis_id] = [
                fact_objs[eid] for eid in evidence_ids if eid in fact_objs
            ]

    # Stage 4: aggregate to overall verdict.
    if strongly_refuted:
        if iteration < max_loop_iterations:
            overall = RefutationOverallVerdict.refutation_to_loop
        else:
            overall = RefutationOverallVerdict.refutation_to_partial
        triggered_loop = overall == RefutationOverallVerdict.refutation_to_loop
        loop_iter: int | None = iteration if triggered_loop else None
    else:
        overall = RefutationOverallVerdict.answer_strengthened
        triggered_loop = False
        loop_iter = None

    # Pick the most representative model for the report — if the
    # classifier fired (the common case for non-numerical strategies),
    # use whatever it ran on; otherwise the hypothesis-generation
    # model is the sole signal of what produced the verdicts.
    report_model = models_used[-1] if models_used else judge_model
    report = RefutationReport(
        run_id=run_id,
        model_used=report_model,
        hypotheses=resolved_hypotheses,
        overall_verdict=overall,
        triggered_loop_reentry=triggered_loop,
        loop_reentry_iteration=loop_iter,
    )

    return RefutationAgentResult(
        report=report,
        retrieval_records=retrieval_records,
        refuting_facts=refuting_facts,
        strongly_refuted=strongly_refuted,
        classifier_responses=classifier_responses,
        fallback_invoked=fallback_invoked,
        models_used=models_used,
    )

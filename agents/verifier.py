"""Block 9b: Verifier Agent.

The Verifier is BRAG's constructive assurance core. It receives a slot
and the candidate set the Retriever produced, then renders a verdict
on whether the evidence actually satisfies the slot's sub_question.
It does NOT generate the answer — that's the Generator, later.

Per Chorus Principle 7, the Verifier MUST use a different model family
from the Planner, the Retriever embedder, the Generator, and the
Refutation Agent. We use Qwen2.5 72B on OpenRouter; the Planner is
Llama 3.3 70B.

Two-stage design:

1. **Deterministic pre-filters** — period integrity and numerical
   exactness (concept_tag matching for XBRL facts). These are the
   load-bearing v3 checks: a query about Q3 2024 operating income
   should not be satisfied by a Q3 2023 operating income fact, and
   not by a Q3 2024 operating margin fact. The build plan is explicit:
   "mismatches are rejected, not accepted with low confidence."

2. **LLM evaluation** — Qwen2.5 72B handles the squishier checks
   (claim support, contradiction detection, attribution integrity,
   coverage scoring). The deterministic rejections are reported on
   the same VerifierOutput so the Live Trace shows which check fired.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import chromadb
from pydantic import ValidationError

from agents.llm_client import LLMError, OpenRouterClient
from agents.retriever.period_filter import (
    period_from_document_id,
    periods_equivalent,
)
from agents.retriever.vector_channel import (
    FACT_CHROMA_PATH,
    FACT_COLLECTION,
    CHUNK_CHROMA_PATH,
    CHUNK_COLLECTION,
)
from ingestion.xbrl.concept_filter import human_label, is_canonical
from schemas.enums import CandidateSource, EvidenceType, VerifierVerdict
from schemas.records import (
    ContradictionDetail,
    EvidenceSlot,
    RetrievalRecord,
    VerifierOutput,
)


VERIFIER_MODEL = "qwen/qwen-2.5-72b-instruct"
# Defense in depth — Venice has the null-content / 429 problem that bit
# the Planner. Novita advertises qwen/qwen-2.5-72b-instruct but returns
# HTTP 400 "does not support endpoint: completions" on every
# /chat/completions call, so we exclude it too. Without this exclusion
# OpenRouter pins the Verifier to Novita after the first call and
# every subsequent verification fails silently, dropping retrieval
# records via the orchestrator's LLMError handler.
_PROVIDER_PREFS: dict[str, Any] = {"provider": {"ignore": ["Venice", "Novita"]}}


# Slot types where numerical exactness applies (per spec §3.5 row).
_NUMERICAL_SLOT_TYPES: frozenset[EvidenceType] = frozenset({
    EvidenceType.specific_metric,
    EvidenceType.cross_period_comparison,
})


# ---------------------------------------------------------------------------
# Candidate enrichment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _EnrichedCandidate:
    candidate_id: str
    source: CandidateSource
    rrf_score: float
    text: str
    period: str | None
    source_document: str
    concept_tag: str | None  # facts only; None for chunks
    fact_type: str | None    # facts only
    asserter: str | None     # facts only (e.g., "Netflix", "Greg Peters")


_clients: dict[str, chromadb.PersistentClient] = {}


def _get_client(path) -> chromadb.PersistentClient:
    key = str(path)
    if key not in _clients:
        _clients[key] = chromadb.PersistentClient(path=key)
    return _clients[key]


def _enrich(record: RetrievalRecord) -> list[_EnrichedCandidate]:
    """Fetch claim/text and metadata for every candidate ID. One Chroma
    batch get per source layer."""
    out: dict[str, _EnrichedCandidate] = {}

    fact_ids = [c.candidate_id for c in record.candidates if c.source == CandidateSource.fact]
    chunk_ids = [c.candidate_id for c in record.candidates if c.source == CandidateSource.chunk]
    rrf_score_by_id = {c.candidate_id: c.rrf_score for c in record.candidates}

    if fact_ids:
        coll = _get_client(FACT_CHROMA_PATH).get_collection(name=FACT_COLLECTION)
        res = coll.get(ids=fact_ids, include=["documents", "metadatas"])
        for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
            meta = meta or {}
            out[cid] = _EnrichedCandidate(
                candidate_id=cid,
                source=CandidateSource.fact,
                rrf_score=rrf_score_by_id[cid],
                text=doc or "",
                period=meta.get("period") or None,
                source_document=str(meta.get("source_document", "")),
                concept_tag=meta.get("concept_tag") or None,
                fact_type=meta.get("fact_type") or None,
                asserter=None,  # asserter isn't in fact_store Chroma metadata yet
            )

    if chunk_ids:
        coll = _get_client(CHUNK_CHROMA_PATH).get_collection(name=CHUNK_COLLECTION)
        res = coll.get(ids=chunk_ids, include=["documents"])
        for cid, doc in zip(res["ids"], res["documents"]):
            doc_id = cid.split("__", 1)[0]
            out[cid] = _EnrichedCandidate(
                candidate_id=cid,
                source=CandidateSource.chunk,
                rrf_score=rrf_score_by_id[cid],
                text=doc or "",
                period=period_from_document_id(doc_id),
                source_document=doc_id,
                concept_tag=None,
                fact_type=None,
                asserter=None,
            )

    # Preserve the RetrievalRecord candidate order.
    return [out[c.candidate_id] for c in record.candidates if c.candidate_id in out]


# ---------------------------------------------------------------------------
# Deterministic pre-filters
# ---------------------------------------------------------------------------


_METRIC_PHRASE_RE = re.compile(
    r"\b(?:operating\s+income(?:\s*\(loss\))?|operating\s+margin|"
    r"net\s+income(?:\s*\(loss\))?|"
    r"revenues?|"
    r"gross\s+profit|"
    r"free\s+cash\s+flow|"
    r"operating\s+cash\s+flow|"
    r"net\s+cash\s+provided\s+by\s+operating|"
    r"net\s+cash\s+provided\s+by\s+(?:used\s+in\s+)?investing|"
    r"net\s+cash\s+provided\s+by\s+(?:used\s+in\s+)?financing|"
    r"cash\s+and\s+cash\s+equivalents|"
    r"earnings\s+per\s+share|EPS|"
    r"paid\s+net\s+adds?|paid\s+net\s+additions?|"
    r"paid\s+memberships?|paid\s+subscribers?|"
    r"content\s+amortization|"
    r"content\s+assets|"
    r"stockholders'?\s+equity|"
    r"total\s+assets|total\s+liabilities|"
    r"long[- ]term\s+debt|"
    r"marketing\s+expense|"
    r"general\s+and\s+administrative|"
    r"research\s+and\s+development|technology\s+and\s+development|"
    r"interest\s+expense|"
    r"income\s+tax(?:es)?)\b",
    re.IGNORECASE,
)


def _normalize_phrase(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _mentioned_metrics(slot: EvidenceSlot) -> set[str]:
    """Extract canonical metric phrases from the slot's sub_question
    and key_terms. Returns a set of normalized phrases."""
    text = slot.sub_question + " " + " ".join(slot.key_terms)
    return {_normalize_phrase(m.group(0)) for m in _METRIC_PHRASE_RE.finditer(text)}


def _concept_human_label(concept_tag: str | None) -> str | None:
    if not concept_tag:
        return None
    if not is_canonical(concept_tag):
        # Concept exists but isn't in our retention dict. Fall back to
        # the local part of the tag so we still get a comparable string.
        return concept_tag.split(":", 1)[-1] if ":" in concept_tag else concept_tag
    try:
        return human_label(concept_tag)
    except KeyError:
        return None


def _concept_matches_mentions(
    concept_tag: str | None, mentions: set[str]
) -> bool:
    """True iff one of the slot's mentioned metric phrases appears in
    the concept's human label. Used to drop e.g. an OperatingExpenses
    candidate when the slot asks for operating income."""
    label = _concept_human_label(concept_tag)
    if not label:
        return False
    label_n = _normalize_phrase(label)
    return any(m in label_n for m in mentions)


def _period_matches(cand: _EnrichedCandidate, period_filter: str) -> bool:
    """Mirror the channel-side equivalence (Block 19): FY{Y} <-> {Y}-12-31
    and {Y}Q{N} <-> quarter-end instant. Without equivalence here, the
    pre-filter rejects valid candidates the Retriever already accepted —
    e.g. an XBRL instant fact at 2019-12-31 against a FY2019 slot."""
    if periods_equivalent(cand.period, period_filter):
        return True
    if cand.period is None and cand.source_document:
        return periods_equivalent(
            period_from_document_id(cand.source_document),
            period_filter,
        )
    return False


@dataclass(frozen=True)
class _PreFilterResult:
    surviving: list[_EnrichedCandidate]
    rejected: list[str]  # candidate IDs
    reasons: dict[str, str]  # candidate_id -> rejection reason


def _pre_filter(
    slot: EvidenceSlot, candidates: list[_EnrichedCandidate]
) -> _PreFilterResult:
    """Deterministic pre-filter for period integrity and numerical
    exactness. Returns surviving candidates and rejection metadata."""
    rejected: list[str] = []
    reasons: dict[str, str] = {}
    surviving: list[_EnrichedCandidate] = []

    numerical_check = slot.evidence_type in _NUMERICAL_SLOT_TYPES
    mentions = _mentioned_metrics(slot) if numerical_check else set()

    for c in candidates:
        # Period integrity: any slot with a period_filter must match.
        if slot.period_filter and not _period_matches(c, slot.period_filter):
            rejected.append(c.candidate_id)
            reasons[c.candidate_id] = (
                f"period_integrity: candidate period={c.period!r} "
                f"(source_document={c.source_document!r}) does not match "
                f"slot.period_filter={slot.period_filter!r}"
            )
            continue
        # Numerical exactness: applies only to XBRL-style facts with a
        # concept_tag, when the slot mentions a specific metric.
        if numerical_check and c.concept_tag and mentions:
            if not _concept_matches_mentions(c.concept_tag, mentions):
                rejected.append(c.candidate_id)
                reasons[c.candidate_id] = (
                    f"numerical_exactness: concept_tag {c.concept_tag!r} "
                    f"(label {_concept_human_label(c.concept_tag)!r}) "
                    f"does not match any of slot mentions {sorted(mentions)!r}"
                )
                continue
        surviving.append(c)

    return _PreFilterResult(surviving=surviving, rejected=rejected, reasons=reasons)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_verifier_json(content: str | None) -> Any:
    if not content:
        raise LLMError("Verifier response had empty content.")
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
                f"Verifier response had unparseable JSON object: {e}. "
                f"Snippet: {om.group(0)[:400]!r}"
            ) from e
    raise LLMError(
        f"Verifier response is not JSON: {content[:400]!r}"
    )


SYSTEM_PROMPT = """\
You are the Verifier Agent in BRAG. Your role is to evaluate whether a
retrieved candidate set actually satisfies a slot's sub_question. You
do NOT generate the answer — you render a verdict and a coverage score.

# Identifying candidates

Each candidate is labelled in the user message like:

  [C1] id=F-XBRL-nflx-10q-2024-q1-us-gaap-Revenues-2024Q1
       source=fact rrf_score=0.01234
       ...

The ``[C1]``, ``[C2]``, ``[C3]`` labels are positional markers for
your convenience while reading the prompt — they are NOT the
candidate's id. When you populate ``supported_candidates``,
``rejected_candidates``, or ``conflicting_ids``, you MUST use the
full ``id=`` value (e.g. ``F-XBRL-nflx-10q-2024-q1-us-gaap-Revenues-2024Q1``),
NEVER the ``C1``/``C2`` label. Returning ``["C1", "C2"]`` is a
critical error that breaks downstream pipeline stages.

# Output

Return a single JSON object (no markdown, no prose) with these fields:

{
  "coverage_score": float in [0.0, 1.0],
  "verdict": "covered" | "gap" | "contradiction" | "exhausted",
  "gap_description": string | null,
  "contradiction_details": [
    {
      "description": "...",
      "conflicting_ids": [
        "F-XBRL-nflx-10k-2023-us-gaap-Revenues-FY2023",
        "F-PROSE-nflx-q4-2023-letter-0042"
      ]
    }
  ],
  "supported_candidates": [
    "F-XBRL-nflx-10q-2024-q1-us-gaap-Revenues-2024Q1",
    "F-PROSE-nflx-q1-2024-letter-0017"
  ],
  "rejected_candidates": [
    "nflx-q1-2024-transcript__qa__chunk_8"
  ]
}

(Those id values are examples of the SHAPE — the exact ids on each
run come from the candidate list in the user message.)

Use "exhausted" only if instructed by the caller — under normal
operation pick from covered / gap / contradiction.

# Coverage rubric (calibration)

  1.00  Evidence directly answers the sub_question with all the
        detail asked for (specific numbers, exact periods, full
        attributions).
  0.80  Substantive answer present — the candidate set directly
        addresses the asked subject and resolves the central
        question, with at most minor missing context. For
        ``specific_metric`` / ``cross_period_comparison`` slots
        this requires a numerical value with unit and period. For
        ``forward_looking_statement`` slots, directional guidance
        ("we expect X up/down vs prior period", "guidance
        increased from 22%-23% to 24%") and relative guidance with
        a quantitative anchor count as substantive — Netflix's
        actual guidance language is often directional rather than
        a single forecast number. For ``causal_explanation``,
        ``temporal_evolution``, ``risk_disclosure``, ``accounting_
        policy``, ``strategic_position`` slots, narrative answers
        that explain the why, describe the position, or trace the
        evolution count as substantive even without numerical
        precision.
  0.50  Partial — addresses the topic but misses a key element
        (wrong period, adjacent metric instead of the asked one,
        narrative that hints at the answer without stating it).
  0.20  Topically adjacent — the candidate mentions the broad
        subject area but does NOT address the specific question
        asked. Do not score here just because a forward-looking
        or narrative answer lacks a specific number; if the
        candidate engages with the asked subject and provides
        directional or narrative information, it belongs at
        0.50–0.80, not 0.20.
  0.00  No relevant evidence in the candidate set.

# Verdict rules

- "covered" — coverage_score >= the slot's coverage_threshold AND
  there is no internal contradiction.
- "gap" — coverage_score < threshold. Provide a concrete
  gap_description naming what specific evidence is missing.
- "contradiction" — two or more candidates make incompatible claims
  about the same metric or position. Populate contradiction_details.

# Seven checks

For each candidate, judge:

1. **Slot coverage** — does the candidate set as a whole satisfy the
   sub_question?
2. **Claim support** — does the candidate's text actually support a
   claim relevant to the sub_question, or is it merely topically
   adjacent? Topically-adjacent candidates go in rejected_candidates.
3. **Numerical exactness** — for specific_metric / cross_period_comparison
   slots, does the candidate's value, unit, and concept match what was
   asked? (Period and concept have already been pre-filtered
   deterministically; you may treat the surviving candidates as
   period/concept-clean and focus on value/unit reasonableness.)
4. **Contradiction detection** — do any two candidates make
   incompatible claims about the same metric, period, or position?
5. **Attribution integrity** — when a claim is attributed to a person
   (e.g., Greg Peters, Spencer Neumann), confirm the candidate's text
   actually supports that attribution; flag any mismatches.
6. **Period integrity** — for slots with a period_filter, confirm the
   evidence comes from the constrained period. (Pre-filtered, but
   surface any residual concerns.)
7. **Coverage threshold** — is your final coverage_score >= the slot's
   coverage_threshold?

Return ONLY the JSON object — no prose, no markdown fences.
"""


def _format_candidate(c: _EnrichedCandidate, idx: int) -> str:
    """Compact, deterministic representation of one candidate for the
    LLM prompt."""
    fields = [
        f"  [C{idx}] id={c.candidate_id}",
        f"        source={c.source.value} rrf_score={c.rrf_score:.5f}",
    ]
    if c.fact_type:
        fields.append(f"        fact_type={c.fact_type}")
    if c.concept_tag:
        fields.append(f"        concept_tag={c.concept_tag}")
    if c.period:
        fields.append(f"        period={c.period}")
    fields.append(f"        source_document={c.source_document}")
    # Trim candidate text to a reasonable bound. The Verifier doesn't
    # need 2,000 chars of chunk text — first ~600 is enough to judge
    # claim support.
    text = c.text.replace("\n", " ").strip()
    if len(text) > 600:
        text = text[:600] + "…"
    fields.append(f'        text: "{text}"')
    return "\n".join(fields)


def _build_user_message(
    slot: EvidenceSlot,
    surviving: list[_EnrichedCandidate],
    pre_rejected: dict[str, str],
) -> str:
    lines = [
        "SLOT:",
        f"  slot_id: {slot.slot_id}",
        f"  sub_question: {slot.sub_question}",
        f"  evidence_type: {slot.evidence_type.value}",
        f"  target_layer: {slot.target_layer.value}",
        f"  period_filter: {slot.period_filter or '(none)'}",
        f"  coverage_threshold: {slot.coverage_threshold}",
        f"  key_terms: {list(slot.key_terms)}",
        "",
        f"CANDIDATES ({len(surviving)} surviving the pre-filter):",
    ]
    if not surviving:
        lines.append("  (no candidates survived the deterministic pre-filter)")
    else:
        for idx, c in enumerate(surviving, start=1):
            lines.append(_format_candidate(c, idx))
    if pre_rejected:
        lines.append("")
        lines.append(
            f"DETERMINISTICALLY PRE-REJECTED ({len(pre_rejected)}): these "
            "have already been removed by period_integrity / "
            "numerical_exactness checks. Do not re-include them in your "
            "supported_candidates."
        )
        for cid, reason in list(pre_rejected.items())[:10]:
            lines.append(f"  - {cid}: {reason}")
        if len(pre_rejected) > 10:
            lines.append(f"  ... +{len(pre_rejected) - 10} more")
    lines.append("")
    lines.append(
        "Render your verdict as a JSON object matching the schema above."
    )
    return "\n".join(lines)


_C_LABEL_RE = re.compile(r"^\s*\[?C(\d+)\]?\s*$", re.IGNORECASE)


def _translate_c_labels(
    ids: list[Any],
    surviving: list[_EnrichedCandidate],
) -> list[str]:
    """Defensive translator: map any ``[Cn]``/``Cn`` index labels the
    LLM returned (despite the prompt's explicit instruction not to)
    back to the candidate's actual ``candidate_id`` using the
    prompt's enumeration order. Non-label strings pass through
    unchanged. Out-of-range labels are dropped — a returned id we
    can't resolve is worse than a missing one."""
    out: list[str] = []
    for raw in ids:
        if not isinstance(raw, str):
            continue
        m = _C_LABEL_RE.match(raw)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(surviving):
                out.append(surviving[idx].candidate_id)
            # Else: drop the unresolvable label silently.
            continue
        out.append(raw)
    return out


def _coerce_verifier_output(
    payload: Any,
    slot: EvidenceSlot,
    deterministic_rejects: list[str],
    surviving: list[_EnrichedCandidate],
) -> VerifierOutput:
    """Validate the LLM's JSON and merge with deterministic rejects so
    a single union list is returned to the caller. Translates any
    [Cn] index labels the model accidentally returned back to real
    candidate ids using ``surviving`` as the lookup."""
    if not isinstance(payload, dict):
        raise ValueError(f"Verifier returned non-object JSON: {type(payload).__name__}")

    contradictions: list[ContradictionDetail] = []
    for d in (payload.get("contradiction_details") or []):
        if not isinstance(d, dict):
            continue
        translated_ids = _translate_c_labels(d.get("conflicting_ids") or [], surviving)
        contradictions.append(
            ContradictionDetail.model_validate({
                "description": d.get("description", ""),
                "conflicting_ids": translated_ids,
            })
        )

    rejected: list[str] = list(deterministic_rejects)
    for cid in _translate_c_labels(payload.get("rejected_candidates") or [], surviving):
        if cid not in rejected:
            rejected.append(cid)

    supported = _translate_c_labels(payload.get("supported_candidates") or [], surviving)

    return VerifierOutput.model_validate(
        {
            "slot_id": slot.slot_id,
            "coverage_score": payload.get("coverage_score"),
            "verdict": payload.get("verdict"),
            "gap_description": payload.get("gap_description") or None,
            "contradiction_details": [c.model_dump() for c in contradictions],
            "supported_candidates": supported,
            "rejected_candidates": rejected,
        }
    )


def verify(
    slot: EvidenceSlot,
    record: RetrievalRecord,
    *,
    client: OpenRouterClient | None = None,
    model: str | None = None,
) -> VerifierOutput:
    """Verify ``slot`` against ``record``. Returns a fully-populated
    VerifierOutput. Empty candidate sets short-circuit to a gap
    verdict with coverage_score=0 (no LLM call)."""
    enriched = _enrich(record)
    pre = _pre_filter(slot, enriched)

    if not pre.surviving:
        # No surviving evidence — gap verdict, no LLM call.
        return VerifierOutput(
            slot_id=slot.slot_id,
            coverage_score=0.0,
            verdict=VerifierVerdict.gap,
            gap_description=(
                "No candidates survived deterministic period/numerical "
                "pre-filters. Retry with reformulated query terms or "
                "broader period scope."
                if pre.rejected else
                "Retriever returned no candidates."
            ),
            contradiction_details=[],
            supported_candidates=[],
            rejected_candidates=list(pre.rejected),
        )

    if client is None:
        client = OpenRouterClient(default_model=model or VERIFIER_MODEL)
    chosen_model = model or VERIFIER_MODEL

    user_msg = _build_user_message(slot, pre.surviving, pre.reasons)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    def _call(msgs: list[dict[str, str]]):
        resp = client.chat(
            msgs,
            model=chosen_model,
            max_tokens=2000,
            extra=_PROVIDER_PREFS,
        )
        return _parse_verifier_json(resp.content), resp

    parsed, _resp = _call(messages)
    try:
        return _coerce_verifier_output(parsed, slot, pre.rejected, pre.surviving)
    except (ValidationError, ValueError) as first_err:
        retry_msg = user_msg + (
            f"\n\nYour prior response failed validation: {first_err}. "
            f"Return a corrected JSON object."
        )
        retry_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": retry_msg},
        ]
        parsed2, _resp2 = _call(retry_messages)
        try:
            return _coerce_verifier_output(parsed2, slot, pre.rejected, pre.surviving)
        except (ValidationError, ValueError) as second_err:
            raise LLMError(
                f"Verifier failed schema validation twice on slot "
                f"{slot.slot_id!r}. First: {first_err}. Second: {second_err}."
            ) from second_err

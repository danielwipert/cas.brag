"""Block 5: prose fact extraction.

Pipeline:
  chunk text + context  ->  LLM (DeepSeek via OpenRouter, JSON mode)
                       ->  list[dict]  (raw JSON)
                       ->  validation: drop facts whose verbatim_anchor
                           is not a substring of the chunk
                       ->  FactRecord objects with deterministic
                           ``F-PROSE-{6-digit serial}`` IDs

Validation policy:
- verbatim_anchor MUST exist character-exact in the chunk text. If not, the
  fact is dropped (and counted in stats). This is the spec-mandated
  ground-truth check.
- fact_type "financial_metric" is filtered out — those come from XBRL.
- confidence < 0.5 is dropped (the prompt forbids it but the model can
  sometimes leak through).
- Pydantic schema validation catches anything else (bad period format,
  missing required field, etc.). Dropped facts are counted by reason.

The serial counter is process-local (resets per run). For dev runs that's
fine; for the full-corpus run in Block 6 the runner will manage the
counter across documents.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator

from agents.llm_client import LLMError, OpenRouterClient
from ingestion.prose.extractor_prompt import (
    SYSTEM_PROMPT,
    ChunkContext,
    build_user_message,
)
from schemas.enums import FactType
from schemas.records import ChunkRecord, FactRecord

_DEBUG_ANCHORS = os.environ.get("BRAG_DEBUG_ANCHORS", "").lower() in ("1", "true", "yes")
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Post-extraction GAAP-keyword filter (Block 5 review-iteration safety net)
# ---------------------------------------------------------------------------
#
# The LLM occasionally relabels GAAP financial-statement figures under
# non-financial fact_types (e.g. "free cash flow grew to $6.9B" extracted
# as strategic_claim). The prompt forbids this but DeepSeek finds ways
# around it. This filter is a deterministic safety net: drop any prose
# fact whose claim sentence contains a known GAAP line-item phrase plus
# a co-occurring numeric figure (dollar amount or percentage).
#
# Designed conservatively — keywords must be specific enough to avoid
# dropping legitimate operational/strategic claims. Bare "revenue" is not
# a keyword (too common in business prose); "streaming revenues",
# "operating margin", "free cash flow" are specific enough.

_GAAP_KEYWORDS_RE = re.compile(
    r"\b("
    r"free cash flow"
    r"|net income"
    r"|operating income"
    r"|operating margin"
    r"|gross margin"
    r"|streaming revenues?"
    r"|cost of revenues?"
    r"|earnings per share"
    r"|diluted (?:eps|earnings per share|shares?)"
    r"|interest expense"
    r"|income tax provision"
    r"|provision for income tax(?:es)?"
    r")\b",
    re.IGNORECASE,
)

# A dollar amount or percentage anywhere in the claim.
_NUMERIC_FIGURE_RE = re.compile(
    r"(?:\$\s*[\d,.]+(?:\s*(?:billion|million|thousand|bn|mn|B|M|K))?"
    r"|\b\d+(?:\.\d+)?\s*%)",
)

# Phrases that on their own indicate a corporate-action / non-policy
# disclosure that the LLM tends to mis-bin as accounting_policy. No
# numeric co-occurrence required — these phrases are unambiguous.
_CORPORATE_ACTION_RE = re.compile(
    r"\b(share repurchase authorization|repurchase authorization)\b",
    re.IGNORECASE,
)


# Fact types where financial-figure leakage is the documented failure
# mode. forward_guidance and accounting_policy are EXEMPT from the GAAP-
# keyword filter:
#   - forward_guidance: guidance about future operating margin / FCF /
#     revenue is valuable refutation material (spec §2.3, the canonical
#     "guidance_vs_actual" refutation strategy depends on retaining
#     these).
#   - accounting_policy: disclosures like "hedging gains of $48M are
#     included in streaming revenues" are legitimate policy facts even
#     though they reference figures.
_GAAP_FILTERED_TYPES: frozenset[str] = frozenset(
    {
        "strategic_claim",
        "operational_metric",
        "causal_explanation",
        "risk_disclosure",
    }
)


def is_gaap_leakage(claim: str, fact_type_value: str | None = None) -> str | None:
    """Return a reason string if ``claim`` contains GAAP-figure leakage
    that should be filtered out, else None.

    The corporate-action filter applies regardless of fact_type. The
    GAAP-keyword filter applies only when ``fact_type_value`` is in the
    narrative types where leakage is the failure mode (or None, in which
    case the caller has chosen to apply the filter unconditionally).

    Reasons:
      'gaap_keyword'        keyword + numeric figure both present
      'corporate_action'    share-repurchase-authorization style
    """
    if _CORPORATE_ACTION_RE.search(claim):
        return "corporate_action"
    if fact_type_value is not None and fact_type_value not in _GAAP_FILTERED_TYPES:
        return None
    if _GAAP_KEYWORDS_RE.search(claim) and _NUMERIC_FIGURE_RE.search(claim):
        return "gaap_keyword"
    return None


def post_filter_facts(facts: list) -> tuple[list, dict[str, int]]:
    """Apply the GAAP/corporate-action filter to a list of FactRecord.

    Returns ``(kept, drops_by_reason)``. Pure function — does not mutate
    the input list.
    """
    kept = []
    drops: dict[str, int] = {}
    for f in facts:
        reason = is_gaap_leakage(f.claim, f.fact_type.value)
        if reason is None:
            kept.append(f)
            continue
        drops[reason] = drops.get(reason, 0) + 1
    return kept, drops


def _whitespace_normalized_locate(anchor: str, chunk_text: str) -> str | None:
    """If ``anchor`` matches ``chunk_text`` after collapsing all whitespace
    runs to single spaces (and as a final fallback, case-insensitively),
    return the actual contiguous substring of ``chunk_text`` that
    corresponds to that match. Otherwise None.

    Why: the LLM frequently rewrites whitespace and lowercases the leading
    letter of bullet-point spans (e.g. "Grown our FY23 margin..." in the
    chunk becomes "grown our FY23 margin..." in its anchor). These are
    rendering-pipeline artifacts, not fidelity failures. When matching
    succeeds in normalized space, we recover the original-text span so the
    stored verbatim_anchor still satisfies the
    "is a substring of the chunk" invariant.
    """
    norm_anchor = _WS_RE.sub(" ", anchor).strip()
    if not norm_anchor:
        return None

    # Build a parallel array: for each char in normalized chunk_text, the
    # index of the corresponding char in the original chunk_text.
    norm_chars: list[str] = []
    orig_indices: list[int] = []
    in_ws = False
    for i, ch in enumerate(chunk_text):
        if ch.isspace():
            if not in_ws and norm_chars:  # don't emit leading whitespace
                norm_chars.append(" ")
                orig_indices.append(i)
            in_ws = True
        else:
            norm_chars.append(ch)
            orig_indices.append(i)
            in_ws = False
    norm_chunk = "".join(norm_chars)

    # Try case-sensitive match first, then case-insensitive.
    pos = norm_chunk.find(norm_anchor)
    if pos < 0:
        pos = norm_chunk.lower().find(norm_anchor.lower())
        if pos < 0:
            return None
    end = pos + len(norm_anchor) - 1
    if end >= len(orig_indices):
        return None
    start_orig = orig_indices[pos]
    end_orig = orig_indices[end]
    return chunk_text[start_orig : end_orig + 1]


_PROSE_FACT_TYPES: frozenset[str] = frozenset(
    {
        FactType.operational_metric.value,
        FactType.forward_guidance.value,
        FactType.strategic_claim.value,
        FactType.causal_explanation.value,
        FactType.risk_disclosure.value,
        FactType.accounting_policy.value,
    }
)


@dataclass
class ExtractionStats:
    n_chunks: int = 0
    n_chunks_with_facts: int = 0
    n_raw_facts: int = 0
    n_kept: int = 0
    n_anchor_ws_recovered: int = 0  # exact match failed but whitespace-normalized matched
    n_dropped_anchor: int = 0
    n_dropped_financial_metric: int = 0
    n_dropped_low_confidence: int = 0
    n_dropped_schema: int = 0
    n_dropped_other: int = 0
    n_llm_errors: int = 0

    def to_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


class FactIdMinter:
    """Hands out deterministic ``F-PROSE-{6-digit}`` IDs in insertion order."""

    def __init__(self, start: int = 1) -> None:
        self._next = start

    def mint(self) -> str:
        fid = f"F-PROSE-{self._next:06d}"
        self._next += 1
        return fid

    def rollback(self) -> None:
        """Return the most recently minted ID to the pool. Used when a
        validation step rejects the fact the ID was reserved for, so kept
        IDs stay contiguous."""
        self._next -= 1

    @property
    def count(self) -> int:
        return self._next - 1


def _build_fact_record(
    raw: dict,
    *,
    chunk_text: str,
    ctx: ChunkContext,
    fact_id: str,
    stats: "ExtractionStats | None" = None,
) -> tuple[FactRecord | None, str | None]:
    """Validate one raw LLM fact against the chunk and the schema.

    Returns ``(record, None)`` on success, or ``(None, reason)`` where
    reason is one of: 'anchor', 'financial_metric', 'low_confidence',
    'schema', 'other'.
    """
    if not isinstance(raw, dict):
        return None, "other"

    fact_type = raw.get("fact_type")
    if fact_type == FactType.financial_metric.value:
        return None, "financial_metric"
    if fact_type not in _PROSE_FACT_TYPES:
        return None, "schema"

    anchor = raw.get("verbatim_anchor")
    if not isinstance(anchor, str) or not anchor:
        return None, "anchor"
    # Try character-exact first; fall back to whitespace-normalized match.
    # Whitespace differences between the LLM's output and our chunk text
    # are artifacts of rendering (PDF -> text -> LLM tokens -> string),
    # not fidelity failures. When the normalized anchor matches, we
    # rewrite verbatim_anchor to the exact substring from the chunk so
    # downstream code's "anchor is a substring of chunk" invariant holds.
    if anchor not in chunk_text:
        recovered = _whitespace_normalized_locate(anchor, chunk_text)
        if recovered is None:
            if _DEBUG_ANCHORS:
                print(f"\n  [anchor drop] anchor={anchor!r}")
                print(f"  [anchor drop] chunk_excerpt={chunk_text[:200]!r}...")
            return None, "anchor"
        anchor = recovered
        if stats is not None:
            stats.n_anchor_ws_recovered += 1

    confidence = raw.get("confidence")
    try:
        confidence_f = float(confidence)
    except (TypeError, ValueError):
        return None, "schema"
    if confidence_f < 0.5:
        return None, "low_confidence"

    try:
        rec = FactRecord(
            fact_id=fact_id,
            claim=raw["claim"],
            asserter=raw.get("asserter") or ctx.asserter_default,
            source_document=ctx.document_id,
            source_section=ctx.section,
            verbatim_anchor=anchor,
            fact_type=FactType(fact_type),
            period=raw.get("period"),
            value=raw.get("value"),
            unit=raw.get("unit"),
            concept_tag=None,
            assertion_date=date.fromisoformat(ctx.assertion_date),
            confidence=confidence_f,
        )
    except Exception:
        return None, "schema"
    return rec, None


def extract_facts_from_chunk(
    chunk_text: str,
    ctx: ChunkContext,
    *,
    client: OpenRouterClient,
    minter: FactIdMinter,
    stats: ExtractionStats,
    model: str | None = None,
    max_tokens: int = 4000,
) -> list[FactRecord]:
    """Call the LLM on one chunk and return the validated FactRecords.

    Side effects: increments ``stats`` counters, mints fact IDs.
    """
    stats.n_chunks += 1
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(chunk_text, ctx)},
    ]
    try:
        parsed, _resp = client.chat_json(
            messages, model=model, max_tokens=max_tokens
        )
    except LLMError as e:
        stats.n_llm_errors += 1
        if _DEBUG_ANCHORS:
            print(f"\n  [llm_error] {ctx.chunk_id}: {e}")
        return []

    if isinstance(parsed, list):
        raw_facts = parsed
    elif isinstance(parsed, dict):
        raw_facts = parsed.get("facts", [])
        if not isinstance(raw_facts, list):
            raw_facts = []
    else:
        raw_facts = []

    stats.n_raw_facts += len(raw_facts)
    out: list[FactRecord] = []
    for raw in raw_facts:
        # Reserve an ID up front; if validation rejects, we burn it (kept
        # IDs remain dense — but minted IDs may have gaps. That's fine
        # because we only persist kept records.)
        fid = minter.mint()
        rec, reason = _build_fact_record(
            raw, chunk_text=chunk_text, ctx=ctx, fact_id=fid, stats=stats
        )
        if rec is None:
            minter.rollback()
            if reason == "anchor":
                stats.n_dropped_anchor += 1
            elif reason == "financial_metric":
                stats.n_dropped_financial_metric += 1
            elif reason == "low_confidence":
                stats.n_dropped_low_confidence += 1
            elif reason == "schema":
                stats.n_dropped_schema += 1
            else:
                stats.n_dropped_other += 1
            continue
        out.append(rec)

    if out:
        stats.n_chunks_with_facts += 1
    stats.n_kept += len(out)
    return out


@dataclass
class DocumentMeta:
    document_id: str
    asserter_default: str  # "Netflix" or named executive for transcripts
    assertion_date: str  # ISO date


def extract_facts_from_document(
    chunks: Iterable[ChunkRecord],
    meta: DocumentMeta,
    *,
    client: OpenRouterClient,
    minter: FactIdMinter,
    stats: ExtractionStats,
    model: str | None = None,
    progress: bool = True,
) -> list[FactRecord]:
    """Iterate over a document's chunks, extract facts, accumulate.

    ``meta.asserter_default`` is used as the fallback asserter when the
    LLM does not name a specific speaker. For 10-Q / shareholder-letter
    chunks this should be ``"Netflix"``; for transcripts it should be the
    speaker name parsed by the upstream transcript splitter (or a
    placeholder if multi-speaker — the LLM is instructed to identify the
    actual speaker from inline labels).
    """
    out: list[FactRecord] = []
    chunks = list(chunks)
    for i, ch in enumerate(chunks, start=1):
        ctx = ChunkContext(
            chunk_id=ch.chunk_id,
            document_id=meta.document_id,
            section=ch.section,
            asserter_default=meta.asserter_default,
            assertion_date=meta.assertion_date,
        )
        if progress:
            print(
                f"  [{i}/{len(chunks)}] {ch.chunk_id} "
                f"({ch.word_count} words)... ",
                end="",
                flush=True,
            )
        recs = extract_facts_from_chunk(
            ch.text, ctx, client=client, minter=minter, stats=stats, model=model
        )
        if progress:
            print(f"{len(recs)} facts")
        out.extend(recs)
    return out


def save_facts_jsonl(facts: Iterable[FactRecord], path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in facts:
            f.write(rec.model_dump_json() + "\n")


def load_facts_jsonl(path) -> list[FactRecord]:
    p = Path(path)
    out: list[FactRecord] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(FactRecord.model_validate(json.loads(line)))
    return out

"""Block 6 final indexing step: combine corpus-wide XBRL + prose facts
into one Fact Store (ChromaDB + BM25).

Reads:
  - data/fact_store/xbrl_facts.jsonl   (Block 6d output, ~3.5k facts)
  - data/fact_store/prose_facts.jsonl  (Block 6e output, ~17k facts)

Pipeline:
  1. Dedupe on (claim_normalized, asserter, source_document, period),
     keeping the higher-confidence record. XBRL facts (confidence=1.00)
     win over prose paraphrases of the same metric.
  2. Rebuild the ChromaDB ``facts`` collection at data/fact_store/chromadb
     (clean rebuild — drops the prior XBRL-only collection from Block 6d).
  3. Build a single BM25 index over all fact claim texts and persist to
     data/fact_store/bm25.pkl. Uses the same tokenizer as the chunk-store
     BM25 (Block 3) so retrieval is symmetric.

Run from the repo root::

    python -m scripts.build_corpus_combined_fact_store
"""
from __future__ import annotations

import json
import pickle
import re
from collections import Counter
from pathlib import Path

from ingestion.fact_store.embed_and_index import (
    EMBED_MODEL_NAME,
    build_fact_store,
)
from ingestion.prose.extract import load_facts_jsonl
from rank_bm25 import BM25Okapi
from schemas.records import FactRecord


XBRL_JSONL = Path("data/fact_store/xbrl_facts.jsonl")
PROSE_JSONL = Path("data/fact_store/prose_facts.jsonl")
BM25_PATH = Path("data/fact_store/bm25.pkl")
LOG_PATH = Path("data/logs/corpus_combined_fact_build.json")

_TOKEN_RE = re.compile(r"\b[a-z0-9][a-z0-9'-]*\b")
_WS_RE = re.compile(r"\s+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _normalize_claim(claim: str) -> str:
    """Aggressive normalization for dedupe key only — not stored."""
    return _WS_RE.sub(" ", claim.lower().strip())


def _dedupe(facts: list[FactRecord]) -> tuple[list[FactRecord], int]:
    """Dedupe on (norm_claim, asserter, source_document, period). Keep
    the higher-confidence record (ties: first seen wins)."""
    by_key: dict[tuple, FactRecord] = {}
    collisions = 0
    for f in facts:
        key = (
            _normalize_claim(f.claim),
            f.asserter,
            f.source_document,
            f.period or "",
        )
        prior = by_key.get(key)
        if prior is None:
            by_key[key] = f
            continue
        collisions += 1
        # Keep higher confidence; on tie, keep the existing one (XBRL
        # facts arrive first in our load order, so they win ties).
        if f.confidence > prior.confidence:
            by_key[key] = f
    return list(by_key.values()), collisions


def _build_bm25(facts: list[FactRecord]) -> dict:
    tokenized = [_tokenize(f.claim) for f in facts]
    bm25 = BM25Okapi(tokenized)
    return {
        "bm25": bm25,
        "fact_ids": [f.fact_id for f in facts],
        "fact_claims": {f.fact_id: f.claim for f in facts},
        "tokenize_version": 1,
    }


def main() -> None:
    if not XBRL_JSONL.exists():
        raise SystemExit(f"Missing {XBRL_JSONL}. Run Block 6d first.")
    if not PROSE_JSONL.exists():
        raise SystemExit(f"Missing {PROSE_JSONL}. Run Block 6e first.")

    xbrl = load_facts_jsonl(XBRL_JSONL)
    prose = load_facts_jsonl(PROSE_JSONL)
    print(f"Loaded {len(xbrl)} XBRL facts and {len(prose)} prose facts.")

    # Order matters for the dedupe tiebreaker — XBRL first means XBRL wins
    # ties (and equal-confidence is impossible since prose is <1.0 and XBRL
    # is exactly 1.0, but we keep the convention).
    combined = xbrl + prose
    deduped, n_collisions = _dedupe(combined)
    print(
        f"After dedupe: {len(deduped)} facts "
        f"(collapsed {n_collisions} duplicate (claim, asserter, doc, period) tuples)."
    )

    type_counter: Counter[str] = Counter(f.fact_type.value for f in deduped)
    source_counter: Counter[str] = Counter(f.source_document for f in deduped)
    print(f"  by type: {dict(type_counter)}")
    print(f"  n_source_documents: {len(source_counter)}")

    print(f"\nEmbedding + indexing into ChromaDB ({EMBED_MODEL_NAME})...")
    stats = build_fact_store(deduped)
    print(
        f"  Indexed {stats['n_facts']} facts at embed_dim={stats['embed_dim']}"
    )

    print(f"\nBuilding BM25 over {len(deduped)} fact claims...")
    payload = _build_bm25(deduped)
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BM25_PATH.open("wb") as f:
        pickle.dump(payload, f)
    print(f"  BM25 -> {BM25_PATH}")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(
        json.dumps(
            {
                "embed_model": EMBED_MODEL_NAME,
                "n_xbrl": len(xbrl),
                "n_prose": len(prose),
                "n_combined_input": len(combined),
                "n_after_dedupe": len(deduped),
                "n_dedupe_collisions": n_collisions,
                "by_type": dict(type_counter),
                "n_source_documents": len(source_counter),
                "embed_dim": stats["embed_dim"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Build log -> {LOG_PATH}")


if __name__ == "__main__":
    main()

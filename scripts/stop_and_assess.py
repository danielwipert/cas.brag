"""Phase 1 → Phase 2 stop-and-assess: run the 10 manual queries from the
build plan against both the Chunk Store and the Fact Store and capture
the top-K hits for each retrieval path (vector + BM25).

Writes a JSON log to data/logs/stop_and_assess.json plus a short stdout
summary per query for eyeballing.

Run from the repo root::

    python -m scripts.stop_and_assess
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

# Windows defaults to cp1252 for stdout; the corpus contains smart quotes
# and em-dashes that crash a charmap encoder.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import chromadb
from sentence_transformers import SentenceTransformer

from ingestion.chunker.embed_and_index import (
    bm25_search as chunk_bm25_search,
    vector_search as chunk_vector_search,
)
from ingestion.fact_store.embed_and_index import (
    COLLECTION_NAME as FACT_COLLECTION,
    DEFAULT_CHROMA_PATH as FACT_CHROMA_PATH,
    EMBED_MODEL_NAME,
)


FACT_BM25_PATH = Path("data/fact_store/bm25.pkl")
OUT_PATH = Path("data/logs/stop_and_assess.json")
K = 5

QUERIES = [
    ("Netflix Q3 2024 operating income", "XBRL financial_metric"),
    ("Netflix advertising tier launch announcement", "strategic_claim + chunks"),
    ("Netflix forward guidance for Q1 2024 paid memberships", "forward_guidance from Q4 2023 letter"),
    ("Netflix content amortization policy", "accounting_policy + footnotes"),
    ("Netflix risk factors related to subscriber retention", "risk_disclosure + Item 1A"),
    ("free cash flow drivers in 2022", "causal_explanation"),
    ("Netflix's stance on advertising in 2018", "strategic_claim from 2018 docs"),
    ("Netflix paid net additions Q4 2023", "operational_metric"),
    ("Greg Peters live sports", "transcript chunks with asserter=Greg Peters"),
    ("Netflix net income FY2023", "financial_metric for FY2023"),
]

_TOKEN_RE = re.compile(r"\b[a-z0-9][a-z0-9'-]*\b")

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def fact_vector_search(query: str, k: int) -> list[dict]:
    """Top-k cosine hits in the facts collection. Returns serializable
    dicts so the result can be JSON-logged."""
    model = _get_model()
    qemb = model.encode([query], normalize_embeddings=True)
    qvec = [list(map(float, qemb[0]))]
    client = chromadb.PersistentClient(path=str(FACT_CHROMA_PATH))
    coll = client.get_collection(name=FACT_COLLECTION)
    res = coll.query(query_embeddings=qvec, n_results=k)
    out = []
    for cid, d, doc, meta in zip(
        res["ids"][0], res["distances"][0],
        res["documents"][0], res["metadatas"][0],
    ):
        out.append({
            "fact_id": cid,
            "distance": round(float(d), 4),
            "claim": doc,
            "fact_type": (meta or {}).get("fact_type"),
            "source_document": (meta or {}).get("source_document"),
            "period": (meta or {}).get("period"),
        })
    return out


def fact_bm25_search(query: str, k: int) -> list[dict]:
    with FACT_BM25_PATH.open("rb") as f:
        payload = pickle.load(f)
    bm25 = payload["bm25"]
    fact_ids = payload["fact_ids"]
    fact_claims = payload["fact_claims"]
    scores = bm25.get_scores(_tokenize(query))
    pairs = sorted(zip(fact_ids, scores), key=lambda p: -p[1])[:k]
    return [
        {
            "fact_id": fid,
            "score": round(float(s), 4),
            "claim": fact_claims.get(fid, ""),
        }
        for fid, s in pairs
        if s > 0
    ]


def chunk_vec(query: str, k: int) -> list[dict]:
    hits = chunk_vector_search(query, k=k)
    return [
        {"chunk_id": cid, "distance": round(float(d), 4),
         "excerpt": txt[:200]}
        for cid, d, txt in hits
    ]


def chunk_bm25(query: str, k: int) -> list[dict]:
    hits = chunk_bm25_search(query, k=k)
    return [
        {"chunk_id": cid, "score": round(float(s), 4),
         "excerpt": txt[:200]}
        for cid, s, txt in hits
    ]


def _print_chunk(label: str, hits: list[dict]) -> None:
    print(f"  {label}:")
    for h in hits:
        score_key = "distance" if "distance" in h else "score"
        print(f"    [{h[score_key]:>7.4f}] {h['chunk_id']}")
        print(f"             {h['excerpt'][:140].strip()[:140]}")


def _print_fact(label: str, hits: list[dict]) -> None:
    print(f"  {label}:")
    for h in hits:
        score_key = "distance" if "distance" in h else "score"
        extra = ""
        if "fact_type" in h:
            extra = f" [{h['fact_type']}/{h.get('period') or '—'}/{h['source_document']}]"
        print(f"    [{h[score_key]:>7.4f}] {h['fact_id']}{extra}")
        print(f"             {h['claim'][:140]}")


def main() -> None:
    results = []
    for q, expected in QUERIES:
        print("=" * 80)
        print(f"QUERY: {q}")
        print(f"EXPECT: {expected}")
        print()
        rec = {
            "query": q,
            "expected": expected,
            "chunk_vector": chunk_vec(q, K),
            "chunk_bm25": chunk_bm25(q, K),
            "fact_vector": fact_vector_search(q, K),
            "fact_bm25": fact_bm25_search(q, K),
        }
        _print_chunk("Chunk vector", rec["chunk_vector"])
        _print_chunk("Chunk BM25",   rec["chunk_bm25"])
        _print_fact("Fact vector",  rec["fact_vector"])
        _print_fact("Fact BM25",    rec["fact_bm25"])
        print()
        results.append(rec)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps({"k": K, "queries": results}, indent=2),
        encoding="utf-8",
    )
    print(f"Log -> {OUT_PATH}")


if __name__ == "__main__":
    main()

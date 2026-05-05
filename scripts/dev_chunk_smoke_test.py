"""Block 3 smoke test: 5 Netflix-shaped queries × (vector, BM25).

For each query, fetches the top-5 vector hits and the top-5 BM25 hits and
writes everything (chunk_id, score/distance, 200-char snippet) to
``data/logs/block3_smoke.json`` for manual inspection.

Run from the repo root after ``build_dev_chunk_store.py``::

    python -m scripts.dev_chunk_smoke_test

Pass criteria (per build plan Block 3): top-3 of each channel are clearly
relevant on every query; vector and BM25 should not return identical
ranking — hybrid retrieval is only useful if the channels disagree
constructively.
"""
from __future__ import annotations

import json
from pathlib import Path

from ingestion.chunker.embed_and_index import bm25_search, vector_search


QUERIES: tuple[str, ...] = (
    "advertising tier strategy",
    "password sharing crackdown impact",
    "free cash flow drivers",
    "content amortization policy",
    "Q1 2024 paid net additions guidance",
)

LOG_PATH = Path("data/logs/block3_smoke.json")
SNIPPET_CHARS = 240


def _snippet(text: str) -> str:
    s = " ".join(text.split())
    return s[:SNIPPET_CHARS] + ("…" if len(s) > SNIPPET_CHARS else "")


def main() -> None:
    results: dict[str, dict] = {}
    for q in QUERIES:
        v = vector_search(q, k=5)
        b = bm25_search(q, k=5)
        results[q] = {
            "vector": [
                {
                    "rank": i + 1,
                    "chunk_id": cid,
                    "distance": dist,
                    "snippet": _snippet(text),
                }
                for i, (cid, dist, text) in enumerate(v)
            ],
            "bm25": [
                {
                    "rank": i + 1,
                    "chunk_id": cid,
                    "score": score,
                    "snippet": _snippet(text),
                }
                for i, (cid, score, text) in enumerate(b)
            ],
        }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for q, r in results.items():
        v_top = [h["chunk_id"] for h in r["vector"][:3]]
        b_top = [h["chunk_id"] for h in r["bm25"][:3]]
        overlap = len(set(v_top) & set(b_top))
        print(f"\n=== {q} ===")
        print(f"vector top-3: {v_top}")
        print(f"bm25  top-3:  {b_top}")
        print(f"overlap: {overlap}/3 (lower is better — channels should differ)")
    print(f"\nFull results -> {LOG_PATH}")


if __name__ == "__main__":
    main()

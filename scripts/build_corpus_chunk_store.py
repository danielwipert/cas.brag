"""Build the chunk store on the full corpus (Block 6c, stage 2).

Reads ``data/corpus/<document_id>/*.txt`` (output of
``build_corpus_sections.py``), applies the section-aware chunker, embeds
with BGE-small, persists Chroma + BM25 to ``data/chunk_store/``, and
writes a build log to ``data/logs/corpus_chunk_build.json``.

This script overwrites the chunk store produced by
``build_dev_chunk_store.py`` (3-doc dev subset) with the full corpus
(~121 docs). The Chroma collection is dropped and rebuilt; the BM25
pickle is replaced.

Run from repo root::

    python -m scripts.build_corpus_chunk_store
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ingestion.chunker.embed_and_index import (
    EMBED_MODEL_NAME,
    build_chunk_store,
)
from ingestion.chunker.section_aware import (
    chunk_document,
    iter_dev_subset_documents,
)


CORPUS_ROOT = Path("data/corpus")
LOG_PATH = Path("data/logs/corpus_chunk_build.json")


def main() -> None:
    docs = iter_dev_subset_documents(CORPUS_ROOT)
    if not docs:
        raise SystemExit(
            f"No documents found under {CORPUS_ROOT}. "
            "Run scripts/build_corpus_sections.py first."
        )

    all_chunks = []
    per_doc_counts: dict[str, int] = {}
    per_section_counts: dict[str, int] = defaultdict(int)
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        per_doc_counts[doc.document_id] = len(chunks)
        for ch in chunks:
            per_section_counts[f"{doc.document_id}/{ch.section}"] += 1

    print(f"Loaded {len(docs)} documents, produced {len(all_chunks)} chunks")
    by_form: dict[str, int] = defaultdict(int)
    for doc_id, n in per_doc_counts.items():
        if "-10k-" in doc_id:
            by_form["10-K"] += n
        elif "-10q-" in doc_id:
            by_form["10-Q"] += n
        elif doc_id.endswith("-letter"):
            by_form["8-K letter"] += n
        elif doc_id.endswith("-transcript"):
            by_form["transcript"] += n
        else:
            by_form["other"] += n
    print("  by form (chunks):")
    for form, n in sorted(by_form.items()):
        print(f"    {form:<15} {n}")

    stats = build_chunk_store(all_chunks)
    print(f"\nIndexed {stats['n_chunks']} chunks at embed_dim={stats['embed_dim']}")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(
        json.dumps(
            {
                "embed_model": EMBED_MODEL_NAME,
                "total_chunks": len(all_chunks),
                "embed_dim": stats["embed_dim"],
                "n_documents": len(docs),
                "per_document_chunks": per_doc_counts,
                "by_form_chunks": dict(by_form),
                "per_section_chunks": dict(per_section_counts),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Build log -> {LOG_PATH}")


if __name__ == "__main__":
    main()

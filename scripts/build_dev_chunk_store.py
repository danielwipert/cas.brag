"""Build the chunk store on the dev subset (build plan Block 3 runner).

Reads ``data/dev_subset/<document_id>/*.txt``, applies the section-aware
chunker, embeds with BGE-small, persists Chroma + BM25 to
``data/chunk_store/``, and writes a build log to
``data/logs/dev_chunk_build.json``.

Run from the repo root::

    python -m scripts.build_dev_chunk_store
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


DEV_SUBSET_ROOT = Path("data/dev_subset")
LOG_PATH = Path("data/logs/dev_chunk_build.json")


def main() -> None:
    docs = iter_dev_subset_documents(DEV_SUBSET_ROOT)
    if not docs:
        raise SystemExit(
            f"No documents found under {DEV_SUBSET_ROOT}. "
            "Did Block 2 finish writing the dev subset?"
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
    for doc_id, n in per_doc_counts.items():
        print(f"  {doc_id}: {n} chunks")
    for k, n in sorted(per_section_counts.items()):
        print(f"    - {k}: {n}")

    stats = build_chunk_store(all_chunks)
    print(f"Indexed {stats['n_chunks']} chunks at embed_dim={stats['embed_dim']}")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(
        json.dumps(
            {
                "embed_model": EMBED_MODEL_NAME,
                "total_chunks": len(all_chunks),
                "embed_dim": stats["embed_dim"],
                "per_document": per_doc_counts,
                "per_section": dict(per_section_counts),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Build log -> {LOG_PATH}")


if __name__ == "__main__":
    main()

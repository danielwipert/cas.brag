"""Block 6 Stage I4 — corpus-wide fact store validation.

Re-checks every persisted fact (XBRL + prose) against the source-of-truth
section text under ``data/corpus/{document_id}/*.txt``. Failures here should
be near-zero — the extractor already dropped anchor mismatches inline. This
pass is the safety net.

Checks:
  1. Verbatim anchor presence: anchor must appear in concatenated section
     text for its source_document. Exact substring first; whitespace-
     collapsed fallback (matches the extractor's policy).
  2. XBRL canonical-concept coverage: for each 10-K and 10-Q, which
     canonical concepts produced ≥1 aggregate fact? Per-filing matrix +
     concept-level coverage rates.
  3. Prose extraction coverage: every document should produce ≥1 prose
     fact. Zero-prose-fact documents are flagged.
  4. Period parsing: every fact.period must parse via ``schemas/period.py``.

Writes:
  - data/logs/corpus_validation.json — full report
  - prints a short summary to stdout

Run from the repo root::

    python -m scripts.validate_corpus_fact_store
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from ingestion.prose.extract import load_facts_jsonl
from ingestion.xbrl.concept_filter import all_canonical_concepts
from schemas.period import parse_period
from schemas.records import FactRecord


XBRL_JSONL = Path("data/fact_store/xbrl_facts.jsonl")
PROSE_JSONL = Path("data/fact_store/prose_facts.jsonl")
CORPUS_ROOT = Path("data/corpus")
REPORT_PATH = Path("data/logs/corpus_validation.json")

_WS_RE = re.compile(r"\s+")

# Smart-quote → ASCII normalization for prose/transcript anchors. PDF text
# extraction commonly yields curly quotes that don't match the LLM's input
# (or vice versa).
_QUOTE_MAP = str.maketrans({
    "‘": "'", "’": "'",  # single curly quotes
    "“": '"', "”": '"',  # double curly quotes
    "–": "-", "—": "-",  # en/em dashes
})


def _normalize_text(s: str) -> str:
    return s.translate(_QUOTE_MAP)


def _load_doc_text(document_id: str) -> str | None:
    """Concatenate every section .txt file for a document. Returns None
    if the corpus folder is missing."""
    doc_dir = CORPUS_ROOT / document_id
    if not doc_dir.is_dir():
        return None
    parts: list[str] = []
    for path in sorted(doc_dir.glob("*.txt")):
        try:
            parts.append(path.read_text(encoding="utf-8"))
        except OSError:
            continue
    return "\n".join(parts)


def _anchor_variants(anchor: str) -> list[str]:
    """Yield substring forms to try, ordered cheap-first. Handles XBRL
    rendered-display variants ($-prefix, parenthesized negatives split
    across table cells) and prose smart-quote variants."""
    variants: list[str] = [anchor]
    stripped = anchor.lstrip("$").strip()
    if stripped != anchor:
        variants.append(stripped)
    # XBRL parens for negatives: edgartools sometimes splits "(48,330)" so
    # only one of the parens reaches the extracted text. Also try the bare
    # digit string with both parens stripped.
    if stripped.startswith("(") and stripped.endswith(")"):
        bare = stripped[1:-1].strip()
        variants.append(stripped)
        if bare:
            variants.append(bare)
    # Smart-quote / dash normalization
    normed = _normalize_text(anchor)
    if normed != anchor:
        variants.append(normed)
    return variants


def _check_anchor(anchor: str, doc_text: str, doc_text_ws: str,
                  doc_text_normed: str, doc_text_ws_normed: str) -> bool:
    """True iff anchor appears in doc_text (exact or whitespace-collapsed,
    with optional $-strip / quote normalization)."""
    if not anchor:
        return False
    for v in _anchor_variants(anchor):
        if v in doc_text:
            return True
        if v in doc_text_normed:
            return True
        norm = _WS_RE.sub(" ", v).strip()
        if norm:
            if norm in doc_text_ws:
                return True
            if norm in doc_text_ws_normed:
                return True
    return False


def _is_filing(document_id: str) -> bool:
    return document_id.startswith("nflx-10k-") or document_id.startswith("nflx-10q-")


def _is_letter_or_transcript(document_id: str) -> bool:
    return ("-letter" in document_id) or ("-transcript" in document_id)


def main() -> None:
    if not XBRL_JSONL.exists():
        raise SystemExit(f"Missing {XBRL_JSONL}. Run Block 6d first.")
    if not PROSE_JSONL.exists():
        raise SystemExit(f"Missing {PROSE_JSONL}. Run Block 6e first.")

    print("Loading facts...")
    xbrl = load_facts_jsonl(XBRL_JSONL)
    prose = load_facts_jsonl(PROSE_JSONL)
    all_facts: list[FactRecord] = list(xbrl) + list(prose)
    print(f"  {len(xbrl)} XBRL + {len(prose)} prose = {len(all_facts)} total")

    # ------------------------------------------------------------------
    # 1. Anchor validation. Cache doc text per document_id.
    # ------------------------------------------------------------------
    print("\n[1/4] Verbatim anchor recheck...")
    doc_text_cache: dict[str, tuple[str, str, str, str] | None] = {}
    missing_docs: set[str] = set()
    anchor_failures: list[dict] = []
    anchor_fail_counter: Counter[str] = Counter()
    for f in all_facts:
        cached = doc_text_cache.get(f.source_document)
        if cached is None and f.source_document not in doc_text_cache:
            text = _load_doc_text(f.source_document)
            if text is None:
                doc_text_cache[f.source_document] = None
                missing_docs.add(f.source_document)
            else:
                text_normed = _normalize_text(text)
                doc_text_cache[f.source_document] = (
                    text,
                    _WS_RE.sub(" ", text),
                    text_normed,
                    _WS_RE.sub(" ", text_normed),
                )
        cached = doc_text_cache.get(f.source_document)
        if cached is None:
            anchor_fail_counter["doc_text_missing"] += 1
            continue
        doc_text, doc_text_ws, doc_text_normed, doc_text_ws_normed = cached
        if not _check_anchor(
            f.verbatim_anchor, doc_text, doc_text_ws,
            doc_text_normed, doc_text_ws_normed,
        ):
            anchor_fail_counter[f.source_document] += 1
            if len(anchor_failures) < 30:
                anchor_failures.append(
                    {
                        "fact_id": f.fact_id,
                        "source_document": f.source_document,
                        "source_section": f.source_section,
                        "anchor_excerpt": f.verbatim_anchor[:200],
                    }
                )
    n_anchor_fail = sum(anchor_fail_counter.values())
    print(
        f"  Anchor failures: {n_anchor_fail} / {len(all_facts)} "
        f"({100.0 * n_anchor_fail / len(all_facts):.3f}%)"
    )
    if missing_docs:
        print(f"  WARNING: {len(missing_docs)} docs missing in data/corpus/: "
              f"{sorted(missing_docs)[:5]}...")

    # ------------------------------------------------------------------
    # 2. XBRL canonical-concept coverage per 10-K / 10-Q.
    # ------------------------------------------------------------------
    print("\n[2/4] XBRL canonical-concept coverage...")
    canonical = list(all_canonical_concepts())
    # The fact_id format is F-XBRL-{doc_id}-{concept-without-prefix-colon}-{period}
    # The concept_tag field on FactRecord carries the exact concept string.
    filings_with_xbrl: set[str] = set()
    concept_by_filing: dict[str, set[str]] = defaultdict(set)
    for f in xbrl:
        filings_with_xbrl.add(f.source_document)
        if f.concept_tag:
            concept_by_filing[f.source_document].add(f.concept_tag)

    # Expected filings: every 10-K and 10-Q in the chunked corpus.
    expected_filings = sorted(
        d.name for d in CORPUS_ROOT.iterdir() if d.is_dir() and _is_filing(d.name)
    )
    filings_missing_xbrl_entirely = [
        d for d in expected_filings if d not in filings_with_xbrl
    ]

    concept_coverage: dict[str, dict] = {}
    for concept in canonical:
        present_in = [d for d in expected_filings if concept in concept_by_filing[d]]
        concept_coverage[concept] = {
            "n_filings_present": len(present_in),
            "n_filings_total": len(expected_filings),
            "coverage_pct": (
                round(100.0 * len(present_in) / len(expected_filings), 1)
                if expected_filings else 0.0
            ),
            "missing_from": [d for d in expected_filings if d not in present_in],
        }
    filings_below_80pct_concepts = {
        c: cov for c, cov in concept_coverage.items() if cov["coverage_pct"] < 80.0
    }
    print(
        f"  {len(filings_with_xbrl)}/{len(expected_filings)} filings produced "
        f"XBRL facts; {len(filings_below_80pct_concepts)} canonical concepts "
        f"below 80% coverage."
    )

    # ------------------------------------------------------------------
    # 3. Prose extraction coverage — every doc should have ≥1 prose fact.
    # ------------------------------------------------------------------
    print("\n[3/4] Prose extraction coverage...")
    prose_count_per_doc: Counter[str] = Counter()
    for f in prose:
        prose_count_per_doc[f.source_document] += 1
    all_corpus_docs = sorted(d.name for d in CORPUS_ROOT.iterdir() if d.is_dir())
    zero_prose_docs = [d for d in all_corpus_docs if prose_count_per_doc[d] == 0]
    print(
        f"  Docs with prose facts: {len(all_corpus_docs) - len(zero_prose_docs)}/"
        f"{len(all_corpus_docs)}; zero-prose docs: {len(zero_prose_docs)}"
    )
    if zero_prose_docs:
        print(f"  Zero-prose docs: {zero_prose_docs}")

    # ------------------------------------------------------------------
    # 4. Period parsing — every non-null fact.period must parse.
    # ------------------------------------------------------------------
    print("\n[4/4] Period parsing...")
    period_failures: list[dict] = []
    n_period_present = 0
    n_period_parse_fail = 0
    for f in all_facts:
        if not f.period:
            continue
        n_period_present += 1
        try:
            parse_period(f.period)
        except (ValueError, TypeError) as exc:
            n_period_parse_fail += 1
            if len(period_failures) < 30:
                period_failures.append(
                    {
                        "fact_id": f.fact_id,
                        "period": f.period,
                        "error": str(exc),
                    }
                )
    print(
        f"  Period strings: {n_period_present} present, "
        f"{n_period_parse_fail} parse failures."
    )

    # ------------------------------------------------------------------
    # Write report.
    # ------------------------------------------------------------------
    report = {
        "n_facts_total": len(all_facts),
        "n_xbrl": len(xbrl),
        "n_prose": len(prose),
        "anchor_validation": {
            "n_failures": n_anchor_fail,
            "failure_rate": round(n_anchor_fail / len(all_facts), 6),
            "failures_by_document": dict(anchor_fail_counter),
            "sample_failures": anchor_failures,
            "missing_corpus_docs": sorted(missing_docs),
        },
        "xbrl_coverage": {
            "n_filings_with_any_xbrl": len(filings_with_xbrl),
            "n_filings_expected": len(expected_filings),
            "filings_missing_xbrl_entirely": filings_missing_xbrl_entirely,
            "by_concept": concept_coverage,
            "concepts_below_80pct": sorted(filings_below_80pct_concepts.keys()),
        },
        "prose_coverage": {
            "n_docs_total": len(all_corpus_docs),
            "n_docs_with_prose": len(all_corpus_docs) - len(zero_prose_docs),
            "zero_prose_docs": zero_prose_docs,
            "min_prose_per_doc": (
                min(prose_count_per_doc.values()) if prose_count_per_doc else 0
            ),
            "max_prose_per_doc": (
                max(prose_count_per_doc.values()) if prose_count_per_doc else 0
            ),
        },
        "period_parsing": {
            "n_period_present": n_period_present,
            "n_parse_failures": n_period_parse_fail,
            "sample_failures": period_failures,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport -> {REPORT_PATH}")


if __name__ == "__main__":
    main()

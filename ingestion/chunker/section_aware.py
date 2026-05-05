"""Section-aware chunker for the BRAG dev subset (build plan Block 3).

Produces ChunkRecords from the on-disk dev subset under data/dev_subset.
- 500-word chunks, 50-word overlap, sentence-boundary aware
- Never crosses section boundaries
- Skips known financial-statement tables and cover/boilerplate sections
- Splits transcripts into ``prepared_remarks`` / ``qa`` at the
  "Question and Answer" handoff (build plan Block 3 Q3, choice B)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ingestion.normalize import normalize_text
from schemas.records import ChunkRecord


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_WORDS = 500
DEFAULT_OVERLAP_WORDS = 50

# Sections we never chunk: financial-statement tables (XBRL territory in Block
# 4) and pure boilerplate. The dev subset on disk is already filtered, but the
# skip set lets full-corpus runs in Block 6 be safe.
_SKIP_SECTIONS: frozenset[str] = frozenset(
    {
        "consolidated_balance_sheets",
        "consolidated_statements_of_operations",
        "consolidated_statements_of_cash_flows",
        "consolidated_statements_of_comprehensive_income",
        "consolidated_statements_of_stockholders_equity",
        "cover_page",
        "signatures",
        "exhibits_index",
    }
)


# ---------------------------------------------------------------------------
# Sentence splitter (regex-based; no nltk/spacy dependency)
# ---------------------------------------------------------------------------

# Common abbreviations whose trailing period must NOT split a sentence.
_ABBREV = (
    "Inc",
    "Co",
    "Corp",
    "Ltd",
    "Mr",
    "Mrs",
    "Ms",
    "Dr",
    "Jr",
    "Sr",
    "St",
    "vs",
    "etc",
    "i.e",
    "e.g",
    "U.S",
    "U.K",
    "p.m",
    "a.m",
    "No",
    "Cf",
    "Ave",
)
_ABBREV_PLACEHOLDER = "\x01"  # private-use sentinel never appears in our text

_SENT_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")


def _protect_abbreviations(text: str) -> str:
    for a in _ABBREV:
        text = re.sub(
            rf"\b{re.escape(a)}\.(?=\s)", f"{a}{_ABBREV_PLACEHOLDER}", text
        )
    return text


def _restore_abbreviations(text: str) -> str:
    return text.replace(_ABBREV_PLACEHOLDER, ".")


def _split_sentences(text: str) -> list[str]:
    if not text.strip():
        return []
    protected = _protect_abbreviations(text)
    parts = _SENT_END_RE.split(protected)
    sentences: list[str] = []
    for p in parts:
        # Sub-split on hard newlines that separate paragraph-like turns
        # (transcripts have speaker labels followed by a turn body across
        # multiple lines — keep paragraphs glued, but split on blank lines).
        for chunk in re.split(r"\n{2,}", p):
            chunk = _restore_abbreviations(chunk).strip()
            if chunk:
                sentences.append(chunk)
    return sentences


# ---------------------------------------------------------------------------
# Transcript-specific cleanup and handoff split
# ---------------------------------------------------------------------------

# Lines that are pure S&P Global / q4cdn boilerplate footers repeating across
# pages of a Netflix earnings transcript. These survive PDF extraction and
# would otherwise leak into chunks.
_TRANSCRIPT_FOOTER_RES: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^NETFLIX,\s*INC\.\s+FQ\d.*EARNINGS\s+CALL.*$", re.MULTILINE | re.IGNORECASE
    ),
    re.compile(
        r"^Copyright\s*(\(c\)|\xa9)?\s*\d{4}\s+S&P\s+Global.*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(r"^spglobal\.com/marketintelligence\s*\d*\s*$", re.MULTILINE),
)

_PRESENTATION_RE = re.compile(r"^Presentation\s*$", re.MULTILINE)
_QA_RE = re.compile(r"^Question\s+and\s+Answer\s*$", re.MULTILINE | re.IGNORECASE)


def clean_transcript_boilerplate(text: str) -> str:
    """Strip S&P Global / q4cdn page-footer boilerplate that survived PDF
    extraction. Idempotent."""
    for r in _TRANSCRIPT_FOOTER_RES:
        text = r.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_transcript(text: str) -> dict[str, str]:
    """Split a Netflix earnings transcript into prepared remarks and Q&A.

    Returns a dict with keys ``prepared_remarks`` and ``qa``. If the handoff
    cannot be located, returns the entire (cleaned) text as
    ``prepared_remarks`` and an empty ``qa`` — callers should treat an empty
    ``qa`` as a signal that the splitter failed and may want to log it.
    """
    cleaned = clean_transcript_boilerplate(text)
    pres = _PRESENTATION_RE.search(cleaned)
    qa = _QA_RE.search(cleaned)
    if not pres or not qa or qa.start() <= pres.start():
        return {"prepared_remarks": cleaned, "qa": ""}
    remarks = cleaned[pres.end() : qa.start()].strip()
    qa_body = cleaned[qa.end() :].strip()
    return {"prepared_remarks": remarks, "qa": qa_body}


# ---------------------------------------------------------------------------
# Core chunker
# ---------------------------------------------------------------------------


def _word_count(s: str) -> int:
    return len(s.split())


def chunk_section(
    text: str,
    section_name: str,
    document_id: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_WORDS,
    overlap: int = DEFAULT_OVERLAP_WORDS,
) -> list[ChunkRecord]:
    """Split a single section into sentence-aligned chunks.

    Returns ``[]`` for sections in the skip set. Never crosses the section
    boundary — callers must invoke this once per section.
    """
    if section_name in _SKIP_SECTIONS:
        return []
    text = (text or "").strip()
    if not text:
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[ChunkRecord] = []
    cur: list[str] = []
    cur_words = 0
    idx = 0

    def emit() -> None:
        nonlocal idx
        if not cur:
            return
        body = " ".join(cur).strip()
        chunks.append(
            ChunkRecord(
                chunk_id=f"{document_id}__{section_name}__chunk_{idx}",
                text=body,
                source_document=document_id,
                section=section_name,
                position_index=idx,
                word_count=_word_count(body),
            )
        )
        idx += 1

    for sent in sentences:
        sw = _word_count(sent)
        # If adding this sentence overflows the target AND we already have
        # content, emit the current chunk and start the next with an overlap
        # tail drawn from the trailing sentences of the just-emitted chunk.
        if cur_words + sw > chunk_size and cur:
            prior = cur[:]
            emit()
            tail: list[str] = []
            tail_words = 0
            for s in reversed(prior):
                w = _word_count(s)
                if tail and tail_words + w > overlap:
                    break
                tail.insert(0, s)
                tail_words += w
            cur = tail
            cur_words = tail_words
        cur.append(sent)
        cur_words += sw

    emit()
    return chunks


# ---------------------------------------------------------------------------
# Document-level orchestration
# ---------------------------------------------------------------------------


@dataclass
class DevDocument:
    """A document loaded from the on-disk dev subset, after section split.

    ``sections`` is a list of (section_name, raw_text) pairs. The chunker
    treats each pair independently and never spans across them.
    """

    document_id: str
    sections: list[tuple[str, str]] = field(default_factory=list)


def chunk_document(document: DevDocument) -> list[ChunkRecord]:
    out: list[ChunkRecord] = []
    for section_name, text in document.sections:
        out.extend(chunk_section(text, section_name, document.document_id))
    return out


def _load_document(doc_dir: Path) -> DevDocument:
    """Build a DevDocument from a single dev_subset/<document_id>/ directory.

    Each .txt file becomes a section whose name is the file stem. The
    transcript file is special-cased: it is split into ``prepared_remarks``
    and ``qa`` at the "Question and Answer" handoff, and the cover-page /
    table-of-contents preamble is dropped via the boilerplate cleaner.
    """
    document_id = doc_dir.name
    sections: list[tuple[str, str]] = []
    for txt in sorted(doc_dir.glob("*.txt")):
        raw = normalize_text(txt.read_text(encoding="utf-8"))
        if txt.stem == "transcript":
            parts = split_transcript(raw)
            if parts["prepared_remarks"]:
                sections.append(("prepared_remarks", parts["prepared_remarks"]))
            if parts["qa"]:
                sections.append(("qa", parts["qa"]))
        else:
            sections.append((txt.stem, raw))
    return DevDocument(document_id=document_id, sections=sections)


def iter_dev_subset_documents(root: Path) -> list[DevDocument]:
    """Load every document under ``root`` (default: data/dev_subset/)."""
    root = Path(root)
    docs: list[DevDocument] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        docs.append(_load_document(sub))
    return docs

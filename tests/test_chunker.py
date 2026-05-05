"""Tests for the section-aware chunker (build plan Block 3)."""
from __future__ import annotations

from ingestion.chunker.section_aware import (
    DEFAULT_CHUNK_WORDS,
    DEFAULT_OVERLAP_WORDS,
    DevDocument,
    _split_sentences,
    chunk_document,
    chunk_section,
    clean_transcript_boilerplate,
    split_transcript,
)


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------


def test_split_sentences_basic():
    sents = _split_sentences("This is one. This is two! And three? Yes.")
    assert sents == ["This is one.", "This is two!", "And three?", "Yes."]


def test_split_sentences_protects_abbreviations():
    text = "Netflix, Inc. reported revenue. Mr. Sarandos commented."
    sents = _split_sentences(text)
    assert sents[0].startswith("Netflix, Inc.")
    assert sents[1].startswith("Mr. Sarandos")
    assert len(sents) == 2


# ---------------------------------------------------------------------------
# chunk_section
# ---------------------------------------------------------------------------


def test_chunk_id_format_and_metadata():
    text = "Hello world. " * 30
    chunks = chunk_section(text, "item_2", "nflx-10q-2024-q3")
    assert chunks[0].chunk_id == "nflx-10q-2024-q3__item_2__chunk_0"
    assert chunks[0].source_document == "nflx-10q-2024-q3"
    assert chunks[0].section == "item_2"
    assert chunks[0].position_index == 0


def test_chunk_size_bounded_near_target():
    # 1500 words, sentence boundaries every 5 words. Sentences must start
    # with a capital letter for the regex splitter to fire.
    sent = "Alpha beta gamma delta epsilon. "
    text = sent * 300  # 1500 words
    chunks = chunk_section(text, "letter_body", "nflx-test")
    assert len(chunks) >= 3
    for c in chunks:
        # Allow modest slack — sentence boundaries can push slightly over,
        # and the overlap tail is included in the next chunk's word count.
        assert c.word_count <= DEFAULT_CHUNK_WORDS + 10, (
            f"{c.chunk_id} = {c.word_count} words"
        )


def test_chunk_overlap_carries_prior_tail():
    # Each sentence is unique so we can tell which chunk it lives in.
    # Capitalize first token so the sentence splitter actually splits.
    sentences = [f"Sentence_{i} word_a word_b word_c word_d." for i in range(200)]
    text = " ".join(sentences)  # 200 sentences × 5 words = 1000 words
    chunks = chunk_section(text, "letter_body", "nflx-test")
    assert len(chunks) >= 2
    # The last sentence of chunk 0 should reappear inside the head of chunk 1
    # (overlap tail of ~50 words ≈ 10 sentences at 5 words/sentence).
    chunk0_last_sent_token = chunks[0].text.strip().split(". ")[-1].split()[0]
    chunk1_head_tokens = chunks[1].text.split()[:50]
    assert chunk0_last_sent_token in chunk1_head_tokens, (
        f"expected {chunk0_last_sent_token!r} to appear in chunk 1 head; "
        f"got {chunk1_head_tokens[:5]!r}"
    )


def test_skip_section_returns_empty():
    out = chunk_section("anything here.", "consolidated_balance_sheets", "nflx-test")
    assert out == []


def test_empty_section_returns_empty():
    assert chunk_section("", "item_2", "nflx-test") == []
    assert chunk_section("   \n\n  ", "item_2", "nflx-test") == []


def test_position_index_is_dense_per_section():
    text = "alpha beta gamma delta epsilon. " * 200
    chunks = chunk_section(text, "letter_body", "nflx-test")
    for i, c in enumerate(chunks):
        assert c.position_index == i


# ---------------------------------------------------------------------------
# chunk_document — never crosses section boundaries
# ---------------------------------------------------------------------------


def test_chunk_document_never_crosses_sections():
    doc = DevDocument(
        document_id="nflx-test",
        sections=[
            ("item_2", "alpha. " * 200),  # 200 words
            ("item_3", "beta. " * 200),
        ],
    )
    chunks = chunk_document(doc)
    item_2 = [c for c in chunks if c.section == "item_2"]
    item_3 = [c for c in chunks if c.section == "item_3"]
    for c in item_2:
        assert "beta" not in c.text
    for c in item_3:
        assert "alpha" not in c.text


# ---------------------------------------------------------------------------
# Transcript split + boilerplate strip
# ---------------------------------------------------------------------------


def test_split_transcript_extracts_remarks_and_qa():
    text = (
        "Some cover material that should be dropped\n"
        "Table of Contents stuff\n"
        "Presentation\n\n"
        "Spencer Wang:\nGood afternoon and welcome.\n\n"
        "Question and Answer\n\n"
        "Greg Peters:\nYes as we noted in the letter.\n"
    )
    parts = split_transcript(text)
    assert "Spencer Wang" in parts["prepared_remarks"]
    assert "Good afternoon" in parts["prepared_remarks"]
    assert "cover material" not in parts["prepared_remarks"]
    assert "Greg Peters" in parts["qa"]
    assert "Spencer Wang" not in parts["qa"]


def test_split_transcript_handles_missing_handoff():
    parts = split_transcript("blob with no markers at all")
    assert parts["qa"] == ""
    assert parts["prepared_remarks"] == "blob with no markers at all"


def test_clean_transcript_boilerplate_strips_sp_footers():
    text = (
        "Spencer Wang:\nSome content.\n"
        "NETFLIX, INC. FQ1 2024 EARNINGS CALL APR 18, 2024\n"
        "Copyright \xa9 2024 S&P Global Market Intelligence, a division of S&P Global Inc. All Rights reserved.\n"
        "spglobal.com/marketintelligence 5\n"
        "Greg Peters:\nMore content.\n"
    )
    cleaned = clean_transcript_boilerplate(text)
    assert "S&P Global" not in cleaned
    assert "spglobal" not in cleaned
    assert "EARNINGS CALL" not in cleaned
    assert "Spencer Wang" in cleaned
    assert "Greg Peters" in cleaned

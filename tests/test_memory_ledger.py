"""Block 9a unit tests."""
from __future__ import annotations

import pytest

from pipeline.memory_ledger import (
    Ledger,
    ZERO_PROGRESS_DELTA,
    ZERO_PROGRESS_LIMIT,
)


def test_excluded_for_slot_empty_initially() -> None:
    L = Ledger("run-1")
    assert L.excluded_for_slot("S1") == set()


def test_add_retrieval_accumulates_ids_per_slot() -> None:
    L = Ledger("run-1")
    L.add_retrieval("S1", 1, ["A", "B", "C"])
    L.add_retrieval("S1", 2, ["D", "E"])
    assert L.excluded_for_slot("S1") == {"A", "B", "C", "D", "E"}
    # Different slot is independent.
    assert L.excluded_for_slot("S2") == set()


def test_add_retrieval_dedupes_within_slot() -> None:
    L = Ledger("run-1")
    L.add_retrieval("S1", 1, ["A", "B"])
    L.add_retrieval("S1", 2, ["B", "C"])
    record = L.to_record()
    assert record.retrieved_ids["S1"] == ["A", "B", "C"]


def test_coverage_history_per_slot() -> None:
    L = Ledger("run-1")
    L.add_coverage("S1", 1, 0.5)
    L.add_coverage("S2", 1, 0.7)
    L.add_coverage("S1", 2, 0.9)
    s1 = L.coverage_history_for("S1")
    assert len(s1) == 2
    assert s1[0].iteration == 1 and s1[0].coverage_score == 0.5
    assert s1[1].iteration == 2 and s1[1].coverage_score == 0.9


def test_gap_history_per_slot() -> None:
    L = Ledger("run-1")
    L.add_gap("S1", 1, "missing FY2020 figure")
    L.add_gap("S2", 1, "no period match")
    L.add_gap("S1", 2, "still missing")
    s1 = L.gap_history_for("S1")
    assert [g.gap_description for g in s1] == [
        "missing FY2020 figure", "still missing"
    ]


# ---------------------------------------------------------------------------
# Zero-progress detection
# ---------------------------------------------------------------------------


def test_zero_progress_not_triggered_below_minimum_iterations() -> None:
    L = Ledger("run-1")
    L.add_coverage("S1", 1, 0.5)
    L.add_coverage("S1", 2, 0.51)
    # Only one delta computed — fewer than LIMIT+1 entries means safe.
    assert L.should_exhaust_early("S1") is False


def test_zero_progress_triggers_after_two_stalled_iterations() -> None:
    L = Ledger("run-1")
    L.add_coverage("S1", 1, 0.50)
    L.add_coverage("S1", 2, 0.51)  # delta=0.01 < 0.03 (zero)
    L.add_coverage("S1", 3, 0.515)  # delta=0.005 < 0.03 (zero)
    # Two consecutive zero-progress deltas hit the limit.
    assert L.should_exhaust_early("S1") is True


def test_zero_progress_reset_by_real_progress() -> None:
    L = Ledger("run-1")
    L.add_coverage("S1", 1, 0.50)
    L.add_coverage("S1", 2, 0.51)  # zero
    L.add_coverage("S1", 3, 0.80)  # real progress; reset
    L.add_coverage("S1", 4, 0.81)  # zero, but only 1 in the run
    assert L.should_exhaust_early("S1") is False


def test_zero_progress_threshold_is_strict() -> None:
    # delta exactly at threshold: not below, so counts as progress.
    L = Ledger("run-1")
    L.add_coverage("S1", 1, 0.50)
    L.add_coverage("S1", 2, 0.53)  # delta = 0.03 (== threshold)
    L.add_coverage("S1", 3, 0.56)  # delta = 0.03
    assert L.should_exhaust_early("S1") is False


def test_zero_progress_independent_per_slot() -> None:
    L = Ledger("run-1")
    # S1 stalls; S2 makes progress.
    L.add_coverage("S1", 1, 0.50)
    L.add_coverage("S1", 2, 0.51)
    L.add_coverage("S1", 3, 0.515)
    L.add_coverage("S2", 1, 0.10)
    L.add_coverage("S2", 2, 0.50)
    L.add_coverage("S2", 3, 0.90)
    assert L.should_exhaust_early("S1") is True
    assert L.should_exhaust_early("S2") is False


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def test_to_record_roundtrips() -> None:
    L = Ledger("run-1")
    L.add_retrieval("S1", 1, ["A", "B"])
    L.add_coverage("S1", 1, 0.5)
    L.add_gap("S1", 1, "missing X")
    L.add_supported("S1", ["A"])
    L.mark_query_exhausted("revenue Q3 2024")

    rec = L.to_record()
    assert rec.run_id == "run-1"
    assert rec.retrieved_ids == {"S1": ["A", "B"]}
    assert rec.supported_candidates == {"S1": ["A"]}
    assert rec.exhausted_queries == ["revenue Q3 2024"]
    assert len(rec.coverage_history) == 1
    assert len(rec.gap_history) == 1
    # Refutation fields exist on the record but stay empty in Block 9.
    assert rec.refutation_hypotheses_tested == []
    assert rec.refutation_loop_history == []


def test_constants_match_spec() -> None:
    # Build plan §Block 9 prescribes these constants.
    assert ZERO_PROGRESS_DELTA == 0.03
    assert ZERO_PROGRESS_LIMIT == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

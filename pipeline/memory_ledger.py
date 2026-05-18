"""Block 9a: Retrieval Memory Ledger (extended in Block 11 for refutation).

The Ledger tracks per-slot state across the Verifier loop: which IDs
have been retrieved (so the Retriever can exclude them on the next
iteration), the coverage score history, and the gap descriptions
emitted by the Verifier. It also detects "zero progress" — two
consecutive iterations where coverage barely moves — to trigger early
exhaustion before max_iter is reached.

Block 11 added refutation state: every hypothesis the Refutation Agent
generates is recorded via ``add_refutation_hypothesis`` (so subsequent
loop iterations don't re-test identical hypotheses), and every
refutation-driven loop re-entry is recorded via
``add_refutation_loop`` along with the strongly-refuted hypothesis
that triggered it.

The class wraps the Pydantic ``MemoryLedger`` record from
``schemas/records.py``. ``to_record()`` produces an immutable
snapshot for the ExecutionTrace.
"""
from __future__ import annotations

from schemas.records import (
    CoverageHistoryEntry,
    GapHistoryEntry,
    MemoryLedger,
    RefutationHypothesis,
    RefutationLoopRecord,
)


# Zero-progress threshold and consecutive-count limit per build plan §Block 9.
ZERO_PROGRESS_DELTA = 0.03
ZERO_PROGRESS_LIMIT = 2


class Ledger:
    """Mutable ledger used during a single pipeline run."""

    def __init__(self, run_id: str, *, session_scope: str = "session") -> None:
        self.run_id = run_id
        self.session_scope = session_scope
        # Per-slot retrieved IDs accumulated across iterations.
        self._retrieved: dict[str, list[str]] = {}
        # Per-slot supported IDs accumulated across iterations.
        self._supported: dict[str, list[str]] = {}
        # Coverage and gap history are flat lists ordered by addition.
        self._coverage: list[CoverageHistoryEntry] = []
        self._gaps: list[GapHistoryEntry] = []
        self._exhausted_queries: list[str] = []
        # Refutation activity (Block 11). Hypotheses are accumulated
        # across Refutation Agent invocations on this run; the loop
        # records track every refutation-driven loop re-entry.
        self._refutation_hypotheses: list[RefutationHypothesis] = []
        self._refutation_loops: list[RefutationLoopRecord] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_retrieval(
        self,
        slot_id: str,
        iteration: int,
        candidate_ids: list[str],
    ) -> None:
        """Record the candidate IDs returned for this slot at this
        iteration. Dedupes against any IDs already recorded for the
        slot — the Retriever should have excluded them, but we
        defend against accidental double-adds."""
        bucket = self._retrieved.setdefault(slot_id, [])
        existing = set(bucket)
        for cid in candidate_ids:
            if cid not in existing:
                bucket.append(cid)
                existing.add(cid)

    def add_coverage(
        self,
        slot_id: str,
        iteration: int,
        coverage_score: float,
    ) -> None:
        self._coverage.append(
            CoverageHistoryEntry(
                slot_id=slot_id,
                iteration=iteration,
                coverage_score=coverage_score,
            )
        )

    def add_gap(
        self,
        slot_id: str,
        iteration: int,
        gap_description: str,
    ) -> None:
        self._gaps.append(
            GapHistoryEntry(
                slot_id=slot_id,
                iteration=iteration,
                gap_description=gap_description,
            )
        )

    def add_supported(self, slot_id: str, candidate_ids: list[str]) -> None:
        """Track which IDs the Verifier marked as supported. Used at
        synthesis time and surfaced via ``to_record()``."""
        bucket = self._supported.setdefault(slot_id, [])
        existing = set(bucket)
        for cid in candidate_ids:
            if cid not in existing:
                bucket.append(cid)
                existing.add(cid)

    def mark_query_exhausted(self, query: str) -> None:
        if query not in self._exhausted_queries:
            self._exhausted_queries.append(query)

    def add_refutation_hypothesis(self, h: RefutationHypothesis) -> None:
        """Record a hypothesis the Refutation Agent generated. Used by
        the agent to dedupe across loop iterations — a hypothesis whose
        ``hypothesis_text`` matches one already tested should not be
        re-issued (prevents the agent from going in circles)."""
        existing_texts = {ex.hypothesis_text for ex in self._refutation_hypotheses}
        if h.hypothesis_text not in existing_texts:
            self._refutation_hypotheses.append(h)

    def add_refutation_loop(self, record: RefutationLoopRecord) -> None:
        """Record a refutation-driven loop re-entry."""
        self._refutation_loops.append(record)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def excluded_for_slot(self, slot_id: str) -> set[str]:
        """IDs the Retriever must exclude on the next iteration for
        this slot. Equals the union of all previously-retrieved IDs."""
        return set(self._retrieved.get(slot_id, ()))

    def coverage_history_for(self, slot_id: str) -> list[CoverageHistoryEntry]:
        return [e for e in self._coverage if e.slot_id == slot_id]

    def gap_history_for(self, slot_id: str) -> list[GapHistoryEntry]:
        return [e for e in self._gaps if e.slot_id == slot_id]

    def refutation_hypothesis_texts(self) -> set[str]:
        """Already-tested hypothesis texts — the Refutation Agent uses
        this on loop iterations to avoid regenerating duplicates."""
        return {h.hypothesis_text for h in self._refutation_hypotheses}

    def refutation_loop_count(self) -> int:
        return len(self._refutation_loops)

    def should_exhaust_early(self, slot_id: str) -> bool:
        """True iff the slot has hit ``ZERO_PROGRESS_LIMIT`` consecutive
        iterations where coverage moved less than ``ZERO_PROGRESS_DELTA``.

        Counted from the most-recent iteration backward. The first
        coverage entry can never be zero-progress (no prior to compare
        against), so a slot with fewer than ``LIMIT+1`` entries is
        always safe."""
        history = self.coverage_history_for(slot_id)
        if len(history) <= ZERO_PROGRESS_LIMIT:
            return False
        # Look at the last LIMIT+1 entries: we need LIMIT deltas to be
        # below threshold. Walk from newest backward.
        consec = 0
        for i in range(len(history) - 1, 0, -1):
            delta = abs(history[i].coverage_score - history[i - 1].coverage_score)
            if delta < ZERO_PROGRESS_DELTA:
                consec += 1
                if consec >= ZERO_PROGRESS_LIMIT:
                    return True
            else:
                break
        return False

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def to_record(self) -> MemoryLedger:
        return MemoryLedger(
            run_id=self.run_id,
            retrieved_ids={k: list(v) for k, v in self._retrieved.items()},
            exhausted_queries=list(self._exhausted_queries),
            coverage_history=list(self._coverage),
            gap_history=list(self._gaps),
            refutation_hypotheses_tested=list(self._refutation_hypotheses),
            refutation_loop_history=list(self._refutation_loops),
            supported_candidates={
                k: list(v) for k, v in self._supported.items()
            },
            session_scope=self.session_scope,
        )

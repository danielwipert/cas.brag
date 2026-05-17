"""Block 8: Reciprocal Rank Fusion.

Standard RRF formula (Cormack et al. 2009): for each candidate, sum
1/(k + rank_i) across every input ranking it appears in. Candidates
absent from a ranking contribute 0 to the sum.

k=60 is the canonical default; spec §3.4 calls for it explicitly.
Cross-channel deduplication is implicit: the score sum collapses
duplicate candidate_ids into one row.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


RRF_K = 60


@dataclass(frozen=True)
class FusedCandidate:
    candidate_id: str
    rrf_score: float


def rrf_fuse(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = RRF_K,
) -> list[FusedCandidate]:
    """Fuse one or more ranked candidate-id lists. Returns the unique
    candidates sorted by descending RRF score. Input order within each
    ranking IS the rank (position 0 = rank 1)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    fused = [FusedCandidate(cid, score) for cid, score in scores.items()]
    fused.sort(key=lambda f: f.rrf_score, reverse=True)
    return fused

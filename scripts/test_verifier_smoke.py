"""Block 9b smoke test: feed three representative slots through the
Retriever, then through the Verifier, and print the resulting
VerifierOutput. Confirms:

  - Deterministic pre-filters reject obvious mismatches.
  - The LLM returns a schema-valid VerifierOutput.
  - The merged rejected list contains the pre-filter rejections.

Run from repo root::

    python -m scripts.test_verifier_smoke
"""
from __future__ import annotations

import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from agents.retriever.retriever import retrieve
from agents.verifier import VERIFIER_MODEL, verify
from schemas.enums import ComplexityTier, EvidenceType, TargetLayer
from schemas.records import EvidenceSlot


# Three hand-picked slots covering the key check paths.
_SLOTS: list[tuple[str, ComplexityTier, EvidenceSlot]] = [
    (
        "specific_metric with period_filter — should hit numerical exactness",
        ComplexityTier.simple,
        EvidenceSlot(
            slot_id="S1",
            sub_question="What was Netflix's net income for Q1 2024?",
            evidence_type=EvidenceType.specific_metric,
            target_layer=TargetLayer.fact_store,
            period_filter="2024Q1",
            key_terms=[
                "Netflix net income Q1 2024",
                "Q1 2024 net income",
                "Netflix Q1 2024 earnings",
            ],
            coverage_threshold=0.80,
        ),
    ),
    (
        "accounting_policy, no period filter — pure LLM judgment",
        ComplexityTier.simple,
        EvidenceSlot(
            slot_id="S1",
            sub_question="What is Netflix's stated content amortization policy?",
            evidence_type=EvidenceType.accounting_policy,
            target_layer=TargetLayer.both,
            period_filter=None,
            key_terms=[
                "Netflix content amortization policy",
                "amortization of content assets",
                "content cost accounting treatment",
            ],
            coverage_threshold=0.80,
        ),
    ),
    (
        "strategic_position over chunks — broad slot, no period",
        ComplexityTier.complex,
        EvidenceSlot(
            slot_id="S1",
            sub_question="What was Netflix's initial stance on advertising before 2022?",
            evidence_type=EvidenceType.strategic_position,
            target_layer=TargetLayer.chunk_store,
            period_filter=None,
            key_terms=[
                "Netflix no ads policy",
                "ad-free experience",
                "Netflix stance against advertising",
            ],
            coverage_threshold=0.80,
        ),
    ),
]


def main() -> None:
    print(f"Verifier model: {VERIFIER_MODEL}\n")
    for desc, tier, slot in _SLOTS:
        print("=" * 78)
        print(f"DESCRIPTION: {desc}")
        print(f"  tier={tier.value}  slot={slot.slot_id}  "
              f"evidence_type={slot.evidence_type.value}")
        print(f"  sub_q: {slot.sub_question}")
        print(f"  period_filter: {slot.period_filter}")
        print()

        retrieval = retrieve(slot, complexity_tier=tier)
        print(f"Retriever: {len(retrieval.candidates)} candidates")

        t0 = time.time()
        verifier_out = verify(slot, retrieval)
        elapsed = round(time.time() - t0, 2)

        print(f"\nVerifier (took {elapsed}s):")
        print(f"  verdict:        {verifier_out.verdict.value}")
        print(f"  coverage_score: {verifier_out.coverage_score}")
        print(f"  gap:            {verifier_out.gap_description}")
        print(f"  supported:      {verifier_out.supported_candidates[:3]}"
              f"  (+{max(0, len(verifier_out.supported_candidates) - 3)} more)"
              if len(verifier_out.supported_candidates) > 3
              else f"  supported:      {verifier_out.supported_candidates}")
        print(f"  rejected:       {len(verifier_out.rejected_candidates)} "
              f"({verifier_out.rejected_candidates[:2]}...)")
        if verifier_out.contradiction_details:
            print(f"  contradictions: {verifier_out.contradiction_details}")
        print()


if __name__ == "__main__":
    main()

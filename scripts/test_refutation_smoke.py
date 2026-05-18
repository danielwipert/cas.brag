"""Block 11a smoke test: run the Refutation Agent in isolation on the
showcase "no plans to add ads" verified fact and confirm the pipeline
end-to-end (hypothesis generation → probe retrieval → classifier →
RefutationReport) finds the 2023 ad-tier announcement as strong refutation.

This test is the integration-level counterpart to the prompt-only
harness in ``tests/refutation_prompt_harness.py``. It exercises the
real corpus retrieval and classifier, not just the hypothesis prompt.

Run from repo root::

    python -m scripts.test_refutation_smoke

Requires the fact_store and chroma indices to be populated.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from agents.refutation.agent import (
    CLASSIFIER_MODEL,
    REFUTATION_MODEL,
    run_refutation,
)
from pipeline.memory_ledger import Ledger
from schemas.enums import ComplexityTier, FactType
from schemas.records import FactRecord


def _showcase_fact() -> FactRecord:
    """The hand-crafted 2018 'no plans for ads' strategic_claim — the
    canonical refutation showcase from spec §9.2 query #12."""
    return FactRecord(
        fact_id="synthetic::nflx-q3-2018-no-ads",
        claim="Netflix stated it has no plans to add advertising to its service.",
        asserter="Netflix, Inc.",
        source_document="nflx-q3-2018-letter",
        source_section="Member experience",
        verbatim_anchor="We have no plans to introduce advertising on Netflix.",
        fact_type=FactType.strategic_claim,
        period=None,
        value=None,
        unit=None,
        concept_tag=None,
        assertion_date=date(2018, 10, 16),
        confidence=0.92,
    )


def main() -> None:
    print(f"Refutation model:  {REFUTATION_MODEL}")
    print(f"Classifier model:  {CLASSIFIER_MODEL}")
    print()

    fact = _showcase_fact()
    print(f"Targeted claim: {fact.claim!r}")
    print(f"  fact_id:        {fact.fact_id}")
    print(f"  assertion_date: {fact.assertion_date.isoformat()}")
    print()

    ledger = Ledger("smoke-r1")
    t0 = time.time()
    result = run_refutation(
        run_id="smoke-r1",
        query="Did Netflix have plans to introduce advertising in 2018?",
        complexity_tier=ComplexityTier.simple,
        verified_facts=[fact],
        ledger=ledger,
        iteration=1,
        max_loop_iterations=2,
    )
    elapsed = round(time.time() - t0, 2)
    print(f"Agent elapsed: {elapsed}s")

    report = result.report
    print()
    print(f"Overall verdict:   {report.overall_verdict.value}")
    print(f"Triggered loop:    {report.triggered_loop_reentry}")
    print(f"Hypotheses:        {len(report.hypotheses)}")

    for h in report.hypotheses:
        print(f"\n  h_id={h.hypothesis_id} strategy={h.strategy.value}")
        print(f"    text:    {h.hypothesis_text}")
        print(f"    verdict: {h.refutation_verdict.value}")
        print(f"    evidence_ids: {h.evidence_ids}")
        if h.evidence_ids:
            for eid in h.evidence_ids:
                refuting = result.refuting_facts.get(h.hypothesis_id, [])
                match = next((r for r in refuting if r.fact_id == eid), None)
                if match:
                    print(
                        f"      [{eid}] {match.assertion_date.isoformat()} "
                        f"{match.asserter}: {match.claim[:150]}"
                    )

    # Persist the report for inspection.
    out_dir = Path("data/logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "block11a_refutation_smoke.json"
    out_path.write_text(
        json.dumps(
            {
                "elapsed_s": elapsed,
                "report": report.model_dump(mode="json"),
                "refuting_facts": {
                    hid: [rec.model_dump(mode="json") for rec in recs]
                    for hid, recs in result.refuting_facts.items()
                },
                "ledger": ledger.to_record().model_dump(mode="json"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nReport -> {out_path}")

    # Hard expectation: the showcase MUST produce strong refutation
    # for the later_reversal strategy. If the classifier said anything
    # else, exit non-zero so the smoke test surfaces the regression.
    if not result.strongly_refuted:
        print("\nFAIL: showcase did not produce a strong refutation.")
        sys.exit(1)
    print("\nPASS: strong refutation surfaced on the ads showcase.")


if __name__ == "__main__":
    main()

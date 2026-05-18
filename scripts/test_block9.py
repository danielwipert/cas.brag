"""Block 9c end-to-end test: run the 5 multi-hop Netflix queries from
the build plan through the full pipeline and observe the Verifier loop
in action.

Per-query, prints:

* Decomposition plan summary
* For each slot, per-iteration retrieval size + Verifier verdict
* Whether RETRY fired with a reformulated sub_question
* Whether zero-progress triggered early exhaustion
* Whether period_filter rejected mismatched candidates
* Final degradation level

Logs the full ExecutionTrace per query to data/logs/block9_test.json
for review.

Run from repo root::

    python -m scripts.test_block9
    python -m scripts.test_block9 --only 0   # just the first query
"""
from __future__ import annotations

import argparse
import json
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from pathlib import Path

from pipeline.orchestrator import run_pipeline


OUT_PATH = Path("data/logs/block9_test.json")


_QUERIES: list[tuple[str, str]] = [
    (
        "Compare Netflix's operating margin from FY2019 to FY2023",
        "multi-period numerical",
    ),
    (
        "What did Netflix say about advertising in Q4 2023?",
        "period-filtered strategic",
    ),
    (
        "What was Netflix's free cash flow in 2022 and what drove it?",
        "numerical + causal",
    ),
    (
        "Netflix's password sharing policy timeline",
        "temporal_evolution, multi-period",
    ),
    (
        "Netflix accounting policy for content amortization",
        "accounting_policy, footnote-heavy",
    ),
]


def _summarize(trace) -> str:
    lines = []
    lines.append(
        f"  level={trace.degradation_level.name} "
        f"cause={trace.degradation_cause.value} "
        f"slots={len(trace.final_slot_states)} "
        f"total_iters={trace.total_iterations} "
        f"elapsed={trace.elapsed_seconds}s"
    )
    for fs in trace.final_slot_states:
        lines.append(
            f"    [{fs.slot_id}] {fs.terminal_verdict.value:<14} "
            f"coverage={fs.final_coverage:.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=int, default=None,
                        help="Run only the N-th query (0-indexed).")
    args = parser.parse_args()

    queries = _QUERIES if args.only is None else [_QUERIES[args.only]]
    results: list[dict] = []

    for i, (query, descr) in enumerate(queries):
        print("=" * 78)
        print(f"QUERY {i}: {query}")
        print(f"NOTE:    {descr}")
        print()

        t0 = time.time()
        trace = run_pipeline(query, verbose=True)
        elapsed = round(time.time() - t0, 2)

        print()
        print(_summarize(trace))
        print(f"  wall: {elapsed}s")
        print()

        results.append(trace.model_dump(mode="json"))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"Log -> {OUT_PATH}")


if __name__ == "__main__":
    main()

# BRAG — Boosted Retrieval-Augmented Generation

A cybernetic RAG system over ten years of Netflix's public financial reporting (May 2016 – May 2026, ~120 documents). Showcase build by [Daniel Wipert](https://github.com/danielwipert).

The distinguishing commitment, versus a textbook RAG: **every Normal-coverage answer is gated by two LLM agents whose job is to disagree with the retrieved evidence** — a Verifier that checks constructive coverage, and a Refutation Agent that runs an adversarial counterfactual probe. If either fails, the answer either degrades (Partial / Clarification Request / Hard Halt) or discloses the disagreement in the final output.

The corpus is structured to make this behavior legible. Forward guidance vs. realized actuals, restated values across quarters, strategic reversals ("we have no plans to add ads" → ads launched), and dropped product lines all appear in Netflix's filings and transcripts. A RAG system that just retrieves the highest-similarity chunk will confidently report whichever version it happened to pull. BRAG won't.

---

## Showcase site

The full case study — Normal RAG vs BRAG, an editorial walk through each pipeline stage, a side-by-side interactive comparison on a real query, and the six trace gallery — lives at [`docs/index.html`](docs/index.html). Once GitHub Pages is enabled for this repo (`Settings → Pages → Source: main / docs`), it's served at `https://danielwipert.github.io/cas.brag/`.

To preview locally:

```bash
python -m http.server 8765 -d docs
# open http://127.0.0.1:8765/
```

## Live demo

Six curated queries run end-to-end through the full Validate → Plan → Verify → Refute → Generate → Govern pipeline. The outputs below are the artifacts of an actual run (6/6 queries pass acceptance, 23/23 checks).

Open the per-query trace HTML to see retrieved evidence, Verifier output, Refutation Agent hypotheses + verdicts, the generated answer, governance violations, and the full execution timeline.

| ID  | Query                                                       | Outcome              | Refutation               | Trace |
|-----|-------------------------------------------------------------|----------------------|--------------------------|-------|
| D1  | What was Netflix's revenue for Q2 2023?                     | Normal               | answer_strengthened      | [HTML](docs/demo/D1.html) |
| D2  | Did Netflix ever say it had no plans to add ads?            | Normal (temporal)    | refutation_to_loop       | [HTML](docs/demo/D2.html) |
| D3  | Has Netflix's stance on advertising changed?                | Partial              | refutation_to_loop       | [HTML](docs/demo/D3.html) |
| D4  | Has Netflix's password sharing policy been consistent?      | Partial              | bypassed (slot exhaust.) | [HTML](docs/demo/D4.html) |
| D5  | Why did Netflix's free cash flow turn positive in 2022?     | Clarification Req.   | bypassed                 | [HTML](docs/demo/D5.html) |
| D6  | What's Disney's streaming subscriber count?                 | Hard Halt (OOS)      | bypassed                 | [HTML](docs/demo/D6.html) |

The [`docs/demo/index.html`](docs/demo/index.html) page links to all six.

> GitHub renders `.html` as source. To view the demos rendered, either use the hosted showcase site above, clone the repo and open the files locally, or paste a file URL into [htmlpreview.github.io](https://htmlpreview.github.io/).

---

## Architecture

```
                                  Query
                                    |
                                    v
                          +---------------------+
                          |  Input Validation   |  determines OOS / Hard Halt early
                          +---------------------+
                                    |
                                    v
                          +---------------------+
                          |       Planner       |  Llama 3.3 70B
                          | decomposes query    |  emits DecompositionPlan
                          | into evidence slots |  with period_filter + evidence_type
                          +---------------------+
                                    |
                                    v
                          +---------------------+
                          |     Retriever       |  hybrid: BGE-small dense + BM25
                          | (Fact + Chunk store)|  channel-aware pre-filter
                          +---------------------+
                                    |
                                    v
                          +---------------------+
                          |      Verifier       |  Qwen2.5 72B
                          | constructive cover- |  numerical exactness,
                          | age + period checks |  rubric-graded
                          +---------------------+
                                    |
                                    v
                          +---------------------+
                          |  Refutation Agent   |  Mistral Large 2411
                          | adversarial counter |  7 strategies: restated_value,
                          | factual probe       |  revised_value, guidance_vs_actual,
                          |                     |  later_reversal, alternative_cause,
                          |                     |  materialization, policy_change
                          +---------------------+
                                    |
                                    v
                          +---------------------+
                          |     Generator       |  DeepSeek-Chat
                          | composes answer +   |  with assertion-date attribution
                          | structured discl.   |
                          +---------------------+
                                    |
                                    v
                          +---------------------+
                          | Output Governance   |  schema, citations, OOS, refusal
                          +---------------------+
                                    |
                                    v
                                  Answer
                          (+ MemoryLedger trace)
```

Every retrieval, verifier verdict, refutation hypothesis, and degradation decision is logged to a `MemoryLedger` and rendered into the trace HTML.

---

## Fact Store: dual-path ingestion

A core design choice: financial values come from XBRL, not from the LLM.

- **XBRL path** — `edgartools` + `lxml` parse on-disk XBRL filings. Aggregate + UCAN/EMEA/LATAM/APAC geographic segments. `verbatim_anchor` is the formatted display value matching rendered HTML, so citations land on the exact figure a human would see.
- **Prose path** — LLM extractor (DeepSeek V3) over chunked filings + transcripts. Six fact types: `operational_metric`, `forward_guidance`, `strategic_claim`, `causal_explanation`, `risk_disclosure`, `accounting_policy`. `financial_metric` is XBRL-only by construction.

Both paths produce `FactRecord` schemas with the same shape and land in the same Fact Store. The Refutation Agent draws from both.

---

## Repo layout

```
schemas/         Pydantic records + enums (FactRecord, DecompositionPlan, MemoryLedger, ...)
ingestion/
  edgar/         SEC EDGAR filing acquisition (edgartools)
  transcripts/   q4cdn earnings-call transcript fetch
  chunker/       section-aware chunking (BGE-small token budget)
  xbrl/          XBRL fact ingestion
  prose/         LLM prose fact extraction
  fact_store/    combined fact store (XBRL + prose)
agents/
  planner.py         Llama 3.3 70B — query decomposition
  retriever/         hybrid dense + BM25, channel-aware
  verifier.py        Qwen2.5 72B — constructive coverage gate
  refutation/        Mistral Large 2411 — adversarial probe
  generator/         DeepSeek-Chat — answer composition
  llm_client.py      OpenRouter wrapper, retry-on-transient
pipeline/
  orchestrator.py    Validate → Plan → Verify → Refute → Generate → Govern
  memory_ledger.py   per-run audit trail
  degradation.py     Normal / Partial / CR / Hard Halt decisions
  governance.py      output governance
  trace_renderer.py  text/markdown trace
  html_renderer.py   self-contained per-query HTML
scripts/         build + smoke-test runners (one per block)
tests/           pytest unit + integration tests
docs/demo/       checked-in artifacts from the Phase 3 demo run
```

---

## Running it locally

Building the corpus runs many LLM calls (prose extraction is the expensive step) and downloads SEC filings. The Phase 3 demo can replay against the on-disk Fact + Chunk stores once they exist.

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Fill in OPENROUTER_API_KEY and EDGAR_USER_AGENT.

# 3. Build the dev subset (3 documents — fast, for iteration)
python -m scripts.pull_dev_subset
python -m scripts.build_dev_chunk_store
python -m scripts.build_dev_xbrl_facts
python -m scripts.build_dev_prose_facts --workers 4
python -m scripts.build_dev_combined_fact_store

# 4. (Optional) Build the full corpus (~120 documents)
python -m scripts.pull_sec_filings
python -m scripts.download_discovered_transcripts
python -m scripts.build_corpus_sections
python -m scripts.build_corpus_chunk_store
python -m scripts.build_corpus_xbrl_facts
python -m scripts.build_corpus_prose_facts --workers 4
python -m scripts.build_corpus_combined_fact_store

# 5. Run the demo
python -m scripts.test_phase3_demo
# Outputs land in data/logs/phase3_demo/ (D1..D6.{html,json,txt} + index.html).

# Subset / skip
python -m scripts.test_phase3_demo --only D1,D2
python -m scripts.test_phase3_demo --skip D5
```

Tests:

```bash
pytest                    # unit + integration
python -m scripts.test_phase4_adversarial   # spec §9.2 15-query adversarial slate
```

---

## Provider + models

LLM calls go through [OpenRouter](https://openrouter.ai). Model choices reflect the role:

| Role               | Model                  | Why                                                 |
|--------------------|------------------------|-----------------------------------------------------|
| Planner            | Llama 3.3 70B          | reliable JSON schema adherence on decomposition     |
| Prose extractor    | DeepSeek V3            | cheap + accurate on structured extraction           |
| Verifier           | Qwen2.5 72B            | strong on numerical + period reasoning              |
| Refutation Agent   | Mistral Large 2411     | counterfactual reasoning, willing to disagree       |
| Generator          | DeepSeek-Chat          | clean prose composition with citation discipline    |

Embeddings: local `BAAI/bge-small-en-v1.5` (384-dim, 512-token context). Lexical: `rank_bm25`. Vector store: ChromaDB on disk.

---

## Status + limitations

- **Corpus coverage gap:** 8 pre-2019 10-Q filings are missing MD&A and Quantitative-Risk items (`edgartools`' legacy parser doesn't reach Part I items on old `TenQ`). 10-K side partially rescued. Queries that depend on pre-2019 narrative may degrade to Partial.
- **Adversarial calibration is not exhaustive.** The full Phase 4 adversarial slate (`scripts/test_phase4_adversarial.py`) does not yet pass 15/15. The 6 curated demo queries do.
- **Generator can flake.** OpenRouter / DeepSeek-Chat occasionally drops mid-stream; the demo retries once.
- **Cost.** A full demo run is ~6 minutes of wall-clock and a handful of cents on OpenRouter.

This is a portfolio showcase, not a production system. The architectural pieces are real and audit-traceable; the calibration is honest about its failure modes.

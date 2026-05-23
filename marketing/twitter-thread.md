# Twitter / X thread — Six principles for trustworthy agentic AI

7-tweet thread. Each tweet under 280 characters. Copy and paste each one into the thread composer in order.

The closing tweet links to the principles index, where readers can drill into any individual principle (each principle has its own page with its own OG card).

---

## Tweet 1 (intro / hook)

```
RAG is the weakest link in production AI.

Most retrieval systems don't know when they're wrong — they pull the most-similar passage and trust it.

I built BRAG to fix this. Six design rules came out of it. Useful for anyone shipping agentic AI.

Thread ↓
```

(279 chars)

---

## Tweet 2 (Principle 01)

```
1/ Type your evidence.

Embeddings give you "similar." Type tags give you "relevant."

Tag chunks at ingest (forward-looking, historical, policy, opinion). Filter retrieval by type BEFORE similarity scoring. Otherwise your AI answers from the wrong kind of evidence.
```

(269 chars)

---

## Tweet 3 (Principle 02)

```
2/ Numbers come through a separate channel.

The LLM should never write a number it didn't receive as a typed value.

If you have structured data, use it directly. If you have prose with numbers, pre-extract them deterministically. Stop laundering guesses through paraphrase.
```

(280 chars — at limit)

---

## Tweet 4 (Principle 03)

```
3/ Add a refusal step.

A clean "I can't answer that" is the highest-trust action your AI can take.

Default agentic systems happily try anything (cosine similarity always returns something). Build an explicit "out of scope" check before retrieval runs. Track refusal rate as a metric.
```

(279 chars)

---

## Tweet 5 (Principle 04)

```
4/ Run a skeptic against your own answer.

A second LLM whose only job is to refute your first LLM's answer catches what no eval set can.

Costs ~2x per query. Worth it for any app where being wrong is expensive.

The most distinctive design choice in BRAG.
```

(260 chars)

---

## Tweet 6 (Principle 05)

```
5/ Make every decision logged.

Every retrieval, verdict, hypothesis, degradation, and tool call → one structured ledger you can replay.

Not optional in healthcare/finance/legal/gov. Invaluable everywhere else.

Build the replay UI early, even if it's ugly. You'll use it constantly.
```

(277 chars)

---

## Tweet 7 (Principle 06 + close)

```
6/ Degrade visibly.

Three failure modes: silent, confident-wrong, visibly degraded.

First two lose trust. Third keeps it.

Build explicit outcome states. Surface them in the UI, not the logs.

All six principles, with examples + cost scenarios:
https://danielwipert.github.io/cas.brag/#builders
```

(280 chars — at limit)

---

## Optional follow-up tweet (if engagement warrants)

```
For anyone wanting the "watch it work" version: BRAG is a RAG system I built over 10 years of Netflix's public financial filings. Same query, normal RAG vs BRAG, side by side:

https://danielwipert.github.io/cas.brag/#proof
```

(228 chars)

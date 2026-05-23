# LinkedIn posts — six principles

One post per principle. Copy whichever, paste into LinkedIn, post.

Each ends with a link to the per-principle mini-page (which has its own OG card so the link preview is tailored to that principle).

Suggested cadence: one post per week for six weeks.

---

## Post 1 — Type your evidence

> Most RAG systems treat all retrieved chunks the same. They shouldn't.
>
> A forward-looking statement and a historical fact about the same number look almost identical to an embedding model. So do an opinion and a policy. So do a draft and a final version.
>
> Your retriever returns whichever scored higher. Your LLM treats them all the same way. Your answer is grounded in the wrong type of evidence — confidently, with a citation.
>
> The fix isn't more compute. It's a small taxonomy. Tag evidence at ingest (5–10 categories is usually enough). Have your planner decide what kind your question needs. Filter retrieval by type *before* similarity scoring.
>
> Embeddings give you "similar." Type tags give you "relevant."
>
> This is principle 01 of six rules I pulled out of BRAG, a RAG system I built over 10 years of Netflix's public filings. Full principle here →
>
> https://danielwipert.github.io/cas.brag/p/typed-evidence.html
>
> #AgenticAI #RAG #AIEngineering

---

## Post 2 — Numbers come through a separate channel

> Every time an LLM reads a number out of prose, you have to trust it didn't paraphrase or invent it.
>
> Here's the failure mode that ships:
>
> Your AI assistant tells your team "Q3 revenue was $8.5 billion." That number came from a passage that said "approximately $8.5 billion in adjusted revenue, excluding one-time items." The "approximately" disappeared. The "adjusted" disappeared. The "excluding" disappeared. The citation links back correctly. Nobody catches it until an investor calls.
>
> The fix: the LLM should never write a number it didn't receive as a typed value.
>
> Parse structured data deterministically. Store numbers with provenance. Cite them by reference. If you have databases / APIs / CSVs, use them directly — don't paraphrase them through an LLM. If you have prose with numbers in it, pre-extract those numbers with a parser and store them as typed values.
>
> BRAG uses XBRL (the structured filings the SEC requires) for every financial number. The LLM never reads numbers out of prose. The category of hallucination is eliminated by construction, not by hoping.
>
> Principle 02 of six →
>
> https://danielwipert.github.io/cas.brag/p/separate-channel.html
>
> #AgenticAI #RAG #LLM

---

## Post 3 — Add a refusal step

> The single highest-trust action your AI can take is "I can't answer that."
>
> It's also the hardest one to ship.
>
> A retrieval system always returns *something*. Cosine similarity will rank the top 8 documents even when none of them are relevant. The LLM downstream will dutifully write an answer from those 8 documents. The answer will sound confident. The user will trust it.
>
> Ask your support bot about a competitor's product. It'll retrieve the 8 most similar passages from your own knowledge base, and confidently summarize your own product's features as if they belong to the competitor. The user gets misled. You get a ticket later.
>
> Build a refusal step. Define the scope of your system explicitly. Use a small fast model (or a deterministic check) as a first pass before expensive retrieval. Build an explicit "I can't answer that" outcome state. Surface it in your UI.
>
> Track refusal rate as a metric. If it's zero, your system is lying.
>
> Principle 03 of six →
>
> https://danielwipert.github.io/cas.brag/p/refusal-step.html
>
> #AgenticAI #AIEngineering #ProductManagement

---

## Post 4 — Run a skeptic against your own answer

> You can build all the evals in the world. They'll catch the failure modes you anticipated. They won't catch the failure mode that ships next quarter.
>
> The strongest defense I've found: a second LLM whose only job is to refute your first LLM's answer.
>
> It reads the draft, picks the most attackable claim, generates a counter-hypothesis, and searches the corpus for evidence that *contradicts* the answer. If it finds something, the answer changes. If it can't, the answer stands with higher confidence.
>
> Cost: roughly 2x your per-query LLM spend. Real. Worth it for any application where being wrong is expensive.
>
> Example from BRAG: a query asked "Did Netflix ever say it had no plans to add ads?" Normal RAG retrieved a 2017 transcript ("not having advertising is an important strategic differentiator") and answered "no." Wrong.
>
> BRAG's Refutation Agent generated the hypothesis "Netflix's 2022 letter announced an ad-supported tier, reversing the earlier position." Found the evidence. Triggered a re-retrieval. The final answer included BOTH the original stance AND the reversal.
>
> Principle 04 of six. This is BRAG's most distinctive design choice →
>
> https://danielwipert.github.io/cas.brag/p/run-a-skeptic.html
>
> #AgenticAI #RAG #LLM

---

## Post 5 — Make every decision logged

> When something goes wrong with your AI — and it will — the worst position is "we don't know why it said that."
>
> A customer reports a bad answer. You can't reproduce it because the system is stochastic. You can't tell them why it answered the way it did. They escalate. Compliance asks for the audit log. There isn't one.
>
> Build the ledger from day one. Retrofitting one later is painful.
>
> Every decision your agent makes goes in a single structured log — not log lines, a structured format. Capture *the evidence the model saw*, not just the model's output. Token usage, retrieved IDs, intermediate prompts, every verdict, every tool call.
>
> Build a replay UI early, even if it's ugly. You'll use it constantly.
>
> Make the ledger user-visible when the user asks. Trust comes from being able to show your work.
>
> Not optional in healthcare, finance, legal, or government. Invaluable everywhere else.
>
> Principle 05 of six →
>
> https://danielwipert.github.io/cas.brag/p/logged-decisions.html
>
> #AgenticAI #AIEngineering #Observability

---

## Post 6 — Degrade visibly

> There are three failure modes for an AI system:
>
> 1. Silent — it does the wrong thing without telling you.
> 2. Confident-wrong — it tells you the wrong thing with conviction.
> 3. Visibly degraded — it tells you exactly what it couldn't do.
>
> The first two are how AI loses trust. The third is how AI keeps trust.
>
> The worst failure mode is the confident wrong answer. The second worst is silent failure. Visible degradation is the third option — and almost always the right one.
>
> Build explicit outcome states. (Three to five is usually enough.) Surface them in the UI, not just in the logs. Train your users to read them — a small "I'm only ~70% confident in this" tag is more useful than a confident-looking long answer.
>
> BRAG ships four outcomes for every query: Normal, Partial, Clarification Request, Hard Halt. The answer is annotated with which one it is. Every disclosed gap, contradiction, and refutation is surfaced in the response itself.
>
> Track the distribution over time. A system that's always "Normal" is probably hiding partial failures.
>
> Principle 06 of six — and the most important one for trust →
>
> https://danielwipert.github.io/cas.brag/p/degrade-visibly.html
>
> #AgenticAI #ProductManagement #AIEngineering

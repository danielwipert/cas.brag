# Slide deck outline — Six principles for trustworthy agentic AI

A 12-slide talk you can give in 20-25 minutes (or 10 minutes if you skip the deep-dive on the showpiece). Paste these into Keynote / Google Slides / PowerPoint and use the speaker notes below each slide.

The design system from the showcase translates well to slides — use Fraunces for titles, Inter for body, JetBrains Mono for any code/data. Warm dark background, FT-red accent for emphasis. (See `docs/assets/css/tokens.css` for hex values.)

---

## Slide 1 — Title

**Title:** RAG is the weakest link in production AI.
**Subtitle:** Six principles for builders, from BRAG.
**Footer:** Daniel Wipert · cas.brag

**Speaker notes:**
Open with the punch. "RAG is the weakest link" gets nods from engineers and product people alike — they've all had the moment where their AI confidently quoted a stale or wrong source. Pause. Then introduce: "I built a system to fix this. Six design rules came out of it. They're useful even if you don't use my system."

---

## Slide 2 — The problem (one slide, four bullets)

**Title:** Normal RAG breaks in four ways.

**Bullets:**
- **Stale facts** — restated values stay in the corpus alongside the new version
- **Contradictions inside the corpus** — two passages disagree; the higher-scoring one wins
- **Reversed positions** — "we said X" then "we said not-X"; both look on-topic
- **Hallucinated numbers** — the LLM paraphrases a figure and calls it a citation

**Speaker notes:**
Don't dwell. The audience knows RAG has problems. This slide is calibration — "I see the same things you see." 30 seconds.

---

## Slide 3 — The rule

**Title:** Never let the LLM be the only thing between the question and the answer.

(One sentence, very large type. No bullets.)

**Speaker notes:**
This is the design philosophy. Say it slowly. "Every other principle is a consequence of this one rule." Audience pause.

---

## Slide 4 — The system (one diagram)

**Title:** Two skeptics, gating every answer.

**Diagram:** Query → Validator → Planner → Retriever → **Verifier** → **Refutation Agent** → Generator → Governor → Answer

(Verifier and Refutation Agent in red. Everything else in muted text. Arrows are simple lines.)

**Speaker notes:**
This is what BRAG does. The two red boxes are the distinctive bit — two LLMs whose only job is to disagree with the draft answer. Mention: "Verifier checks coverage; Refutation Agent runs an adversarial counterfactual probe." Don't go deeper unless asked.

---

## Slide 5 — Proof (the live demo or screenshot)

**Title:** Same question. Same corpus. Two systems.

**Visual:** Side-by-side screenshot of the proof section from cas.brag/#proof, OR live demo if the venue has wifi.

**Speaker notes:**
This is the heart of the talk. Walk through D2: "Did Netflix ever say it had no plans to add ads?"

Normal RAG: retrieves a 2017 transcript that says ads aren't part of the strategy. Answers "no." Confidently wrong.

BRAG: same question, full pipeline. The Refutation Agent generates the hypothesis "Netflix's 2022 letter announced an ad-supported tier, reversing the position." Searches for evidence. Finds it. Triggers re-retrieval. Final answer includes BOTH stances with assertion dates.

3-4 minutes. This is the moment that lands.

---

## Slides 6–11 — One slide per principle

Use the same layout for all six:

**Title:** [Principle number] [Principle name]
**Subtitle/Lede:** [The one-liner]
**Body:** [The cost-of-not-doing-it scenario, set as a quote]
**Footer:** [How to apply it, 2-3 bullets]

---

### Slide 6 — Principle 01

**Title:** Type your evidence.
**Lede:** Embeddings give you "similar." Type tags give you "relevant."
**Cost:** *"Your AI is asked about GDPR. The retriever returns a 2019 blog post titled 'GDPR is mostly common sense.' Both look relevant. Your AI answers from the opinion."*
**Apply:**
- Tag chunks at ingest with a small taxonomy
- Have your planner decide what kind of evidence each question needs
- Filter by type *before* similarity scoring

---

### Slide 7 — Principle 02

**Title:** Numbers come through a separate channel.
**Lede:** The LLM never writes a number it didn't receive as a typed value.
**Cost:** *"Your AI tells your team 'Q3 revenue was $8.5B.' Source: 'approximately $8.5B in adjusted revenue, excluding one-time items.' Three qualifiers disappeared. The citation links correctly. An investor calls."*
**Apply:**
- Structured data? Use it directly.
- Unstructured data with numbers? Pre-extract them deterministically.
- Cite numbers by reference, not by retyping.

---

### Slide 8 — Principle 03

**Title:** Add a refusal step.
**Lede:** A clean "I can't answer that" is the highest-value behavior an agent can have.
**Cost:** *"Your support bot is asked about a competitor's product. It retrieves the 8 most-similar passages from your own knowledge base and confidently summarizes your own features as if they belong to the competitor."*
**Apply:**
- Define system scope explicitly
- Use a small fast model as a first-pass scope check
- Track refusal rate as a metric — if it's zero, your system is lying

---

### Slide 9 — Principle 04

**Title:** Run a skeptic against your own answer.
**Lede:** A second LLM tasked with refuting your first one catches what no eval can.
**Cost:** *"Your AI gives a well-cited answer based on a 2017 doc. The policy was superseded in 2022. No eval flagged it because no eval anticipated the phrasing."*
**Apply:**
- Different model family for the skeptic, if possible
- Sharp prompt: "Find a reason to believe this is wrong"
- Loop back to retrieval when refutation succeeds
- Cost: ~2x per query. Worth it.

---

### Slide 10 — Principle 05

**Title:** Make every decision logged.
**Lede:** When someone asks "why did the system do that?" you should be able to show them.
**Cost:** *"Customer reports a bad answer. You can't reproduce it (stochastic). Compliance asks for the audit log. There isn't one. It becomes a company-wide policy review."*
**Apply:**
- One structured ledger, not log lines
- Capture the evidence the model saw, not just the output
- Build a replay UI early, even if it's ugly

---

### Slide 11 — Principle 06

**Title:** Degrade visibly.
**Lede:** The worst failure is confident-wrong. The second worst is silent. Visible degradation is the third option.
**Cost:** *"Your AI is asked a question requiring two facts. It only finds one. Rather than say 'I found A but not B,' it answers from A alone. The user assumes both were considered. They act on incomplete info."*
**Apply:**
- Explicit outcome states (3–5)
- Surface them in the UI, not just the logs
- Track distribution over time — always "Normal" probably means hiding failures

---

## Slide 12 — Close

**Title:** None of this requires BRAG.
**Body:** *"Take what's useful."*

**Three links:**
- Full showcase: cas.brag.com (or the GitHub Pages URL)
- All six principles: cas.brag/#builders
- Source: github.com/danielwipert/cas.brag

**Speaker notes:**
End on the takeaway. The principles are the artifact, not the system. Most of the audience will never use BRAG. They'll use one or two of these principles in their own systems, and that's the win. Open for questions.

---

## Timing guide

- **20-min version (default):** spend ~2 min per principle slide + 4 min on the proof + 2 min intro/close = ~20 min
- **10-min version:** skip slides 6–11, walk through the proof in detail, end on slide 12 with "and here are the principles" pointing at the page
- **45-min version:** add a Q&A break after the proof, then go deeper on slides 4 and 5; demo the system live if you have wifi

## Style notes

- The talk is conversational. The slides carry the structure; you carry the tone.
- The cost-of-not-doing-it quotes (italicized on each principle slide) work well when you read them slowly. They're the part of each principle the audience remembers.
- Don't apologize for opinions. "I think you should…" is weaker than "Do this." The audience is here for a point of view.

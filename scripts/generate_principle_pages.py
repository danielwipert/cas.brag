"""
Generate per-principle mini-pages under docs/p/.

Each principle has hand-tuned content (why_matters, cost_scenario, how_brag,
apply_intro, apply_steps, optional apply_outro). The template wraps it in
the editorial mini-page chrome (masthead, hero, body sections, prev/next nav,
CTA, footer) and writes one HTML file per principle plus an index.

Idempotent: regenerating overwrites the files cleanly.

Usage:
    python -m scripts.generate_principle_pages
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "p"


PRINCIPLES = [
    {
        "slug": "typed-evidence",
        "number": "01",
        "title": "Type your evidence.",
        "lede": "Embeddings give you &ldquo;similar.&rdquo; Type tags give you &ldquo;relevant.&rdquo;",
        "og_title": "Type your evidence &mdash; BRAG Principle 01",
        "og_description": "When a RAG system retrieves 'the most similar passage,' it has no idea what kind of passage that is. Here's why typing evidence is BRAG's first rule for trustworthy agentic AI.",
        "why_matters": """
            <p>When a RAG system retrieves &ldquo;the most similar passage,&rdquo; it has no idea what
            <em>kind</em> of passage that is. A forward&#8209;looking statement and a historical fact
            about the same number look almost identical to an embedding model. So do an opinion
            and a policy. The retriever returns whichever scored higher; the LLM treats them all
            the same way; your answer ends up grounded in the wrong type of evidence.</p>
        """,
        "cost_scenario": """
            <p>You ask &ldquo;What are our compliance obligations under GDPR?&rdquo; The retriever returns
            a chunk that&rsquo;s an <em>internal opinion piece</em> from a 2019 blog post titled
            &ldquo;GDPR compliance is mostly common sense.&rdquo; Both look relevant. The opinion scores
            higher because it&rsquo;s more conversational. Your AI answers based on the opinion.</p>
        """,
        "how_brag": """
            <p>BRAG&rsquo;s Planner emits a <span class="mono">DecompositionPlan</span> &mdash; a list of
            evidence slots, each tagged with an <span class="mono">evidence_type</span> like
            <span class="mono">strategic_position</span>, <span class="mono">forward_guidance</span>,
            <span class="mono">operational_metric</span>, or <span class="mono">accounting_policy</span>.
            The retriever filters candidates by type before scoring. The Verifier grades coverage
            against the right rubric for that type.</p>
        """,
        "apply_intro": "You don&rsquo;t need BRAG&rsquo;s specific taxonomy. The move is just: stop treating all chunks as equivalent.",
        "apply_steps": [
            "Build a taxonomy of evidence types relevant to your domain (5&ndash;10 categories is usually enough).",
            "Tag documents and chunks at ingest time with one or more types.",
            "Have your planner decide what kind of evidence each question needs.",
            "Filter retrieval candidates by type before similarity scoring.",
        ],
        "apply_outro": "",
    },
    {
        "slug": "separate-channel",
        "number": "02",
        "title": "Numbers come through a separate channel.",
        "lede": "The LLM never writes a number it didn&rsquo;t receive as a typed value.",
        "og_title": "Numbers come through a separate channel &mdash; BRAG Principle 02",
        "og_description": "Every time an LLM reads a number out of prose, you have to trust it didn't paraphrase or invent it. Here's how BRAG eliminates a whole category of hallucination.",
        "why_matters": """
            <p>Every time an LLM reads a number out of prose, you have to trust that it didn&rsquo;t
            paraphrase, round, or invent it. There&rsquo;s no good way to verify after the fact. The
            model can launder a guess as a citation because the citation points to a passage that
            <em>contains</em> the number, not to the number itself.</p>
        """,
        "cost_scenario": """
            <p>Your AI assistant tells your sales team &ldquo;Q3 revenue was&hairsp;$8.5&nbsp;billion.&rdquo;
            That number came from a passage that said &ldquo;Approximately&hairsp;$8.5&nbsp;billion in
            adjusted revenue, excluding one&#8209;time items.&rdquo; The &ldquo;approximately&rdquo; disappeared.
            The &ldquo;adjusted&rdquo; disappeared. The &ldquo;excluding one&#8209;time items&rdquo; disappeared. The
            citation links back correctly. Nobody catches it until an investor calls.</p>
        """,
        "how_brag": """
            <p>Financial numbers come from XBRL &mdash; the structured data the SEC requires every
            public company to file. The ingestion pipeline parses XBRL deterministically, stores
            each value as a typed <span class="mono">FactRecord</span> with its source ID, period,
            and exact display string. The LLM never reads numbers out of prose; it cites them by
            ID and the renderer displays them verbatim.</p>
        """,
        "apply_intro": "The pattern generalises to any structured data your domain has.",
        "apply_steps": [
            "Have structured data? (Databases, APIs, CSVs.) Use it directly. Never paraphrase it through an LLM.",
            "Have unstructured data with numbers in it? Pre&#8209;extract those numbers with a deterministic parser (regex, OCR + validation, structured extraction) and store them as typed values with provenance.",
            "When the LLM cites a number, cite it by reference, not by retyping it.",
        ],
        "apply_outro": "The rule: <strong>the LLM never writes a number it didn&rsquo;t receive as a typed value.</strong>",
    },
    {
        "slug": "refusal-step",
        "number": "03",
        "title": "Add a refusal step.",
        "lede": "A clean &ldquo;I don&rsquo;t know&rdquo; is the highest&#8209;value behaviour an agent can have &mdash; and the hardest to ship.",
        "og_title": "Add a refusal step &mdash; BRAG Principle 03",
        "og_description": "A retrieval system always returns something. Here's why a refusal step is one of BRAG's six rules for trustworthy agentic AI.",
        "why_matters": """
            <p>A retrieval system always returns <em>something</em>. Cosine similarity will rank
            the top 8 documents even when none of them are relevant. An LLM downstream will
            dutifully write an answer from those 8 documents. The answer will sound confident.
            The user will trust it. This is how an AI confidently answers questions it shouldn&rsquo;t
            have taken.</p>
        """,
        "cost_scenario": """
            <p>Your support bot is asked about a competitor&rsquo;s product. It retrieves the 8 most
            similar passages from your own knowledge base (none of which are actually about the
            competitor) and confidently summarises your own product&rsquo;s features as if they belong
            to the competitor. The user is misled. You get a ticket later complaining that the
            &ldquo;competitor product&rdquo; didn&rsquo;t work as you described.</p>
        """,
        "how_brag": """
            <p>The first stage of BRAG&rsquo;s pipeline is Input Validation. Before any retrieval runs,
            the system decides whether the query is in scope. If you ask about Disney&rsquo;s
            subscribers, BRAG Hard Halts at stage 01 with
            <span class="mono">cause: input_failure, reason: out_of_scope</span>. Retrieval never
            runs. Total elapsed time: <span class="mono">0.0s</span>.</p>
        """,
        "apply_intro": "Refusal looks different in every system, but the discipline is the same.",
        "apply_steps": [
            "Define the scope of your system explicitly: what topics, what time period, what entities.",
            "Use a small fast model (or even a deterministic check) as a &ldquo;first pass&rdquo; before expensive retrieval and generation.",
            "Build an explicit &ldquo;I can&rsquo;t answer that&rdquo; outcome state and surface it in your UI.",
            "Track refusal rate as a metric. If it&rsquo;s zero, your system is lying.",
        ],
        "apply_outro": "A clean &ldquo;I can&rsquo;t answer that&rdquo; is the single highest&#8209;trust action an AI assistant can take.",
    },
    {
        "slug": "run-a-skeptic",
        "number": "04",
        "title": "Run a skeptic against your own answer.",
        "lede": "A second LLM tasked with refuting your first one catches what no eval set can.",
        "og_title": "Run a skeptic against your own answer &mdash; BRAG Principle 04",
        "og_description": "Evals catch the failure modes you anticipated. A second-pass adversarial checker catches the ones you didn't. Here's BRAG's most distinctive design choice.",
        "why_matters": """
            <p>You can build all the evaluation suites in the world. They&rsquo;ll catch the failure
            modes you anticipated. They won&rsquo;t catch the failure mode that ships next quarter. A
            second&#8209;pass adversarial checker &mdash; another LLM whose only job is to attack your
            first LLM&rsquo;s answer &mdash; catches things eval sets miss because it sees the same
            context the answerer saw, with fresh eyes and an explicit mandate to disagree.</p>
        """,
        "cost_scenario": """
            <p>Your AI gives a confident, well&#8209;cited answer based on a 2017 policy document. The
            policy was superseded in 2022 &mdash; but the 2017 doc was the most similar to the query
            (more explicit, more verbose, more readable). No eval flagged it because no eval
            anticipated the specific phrasing. Your customer reads the answer and acts on
            superseded policy.</p>
        """,
        "how_brag": """
            <p>After the Verifier passes the answer, BRAG&rsquo;s Refutation Agent (a different LLM)
            reads the draft, picks the most attackable claim, generates a counter&#8209;hypothesis,
            and searches the corpus for evidence that <em>contradicts</em> it. It runs seven
            strategies &mdash; <span class="mono">restated_value</span>,
            <span class="mono">later_reversal</span>, <span class="mono">guidance_vs_actual</span>,
            <span class="mono">alternative_cause</span>, <span class="mono">materialization</span>,
            <span class="mono">policy_change</span>, <span class="mono">revised_value</span> &mdash;
            each tuned to a specific way the answer could be wrong.</p>

            <p>In the demo D2 query &ldquo;Did Netflix ever say it had no plans to add ads?&rdquo;, the
            Refutation Agent generated the hypothesis &ldquo;Netflix&rsquo;s Q4&nbsp;2022 shareholder letter
            announced an ad&#8209;supported tier, reversing its earlier no&#8209;ads position,&rdquo; searched
            for evidence, found it, and triggered a re&#8209;retrieval loop. The final answer
            includes both the original position <em>and</em> the reversal.</p>
        """,
        "apply_intro": "The pattern is robust. The cost is real.",
        "apply_steps": [
            "Pick a strong second LLM (different model family from your generator, if possible).",
            "Give it a sharp, single&#8209;purpose prompt: &ldquo;Read this answer and the supporting evidence. Find a reason to believe the answer is wrong.&rdquo;",
            "Categorise the kinds of refutation it should try &mdash; the categories depend on your domain.",
            "Loop back to retrieval when refutation succeeds.",
        ],
        "apply_outro": "Cost: roughly 2&times; your per&#8209;query LLM spend. Worth it for any application where being wrong is expensive.",
    },
    {
        "slug": "logged-decisions",
        "number": "05",
        "title": "Make every decision logged.",
        "lede": "When someone asks &ldquo;why did the system do that?&rdquo; you should be able to show them, step by step.",
        "og_title": "Make every decision logged &mdash; BRAG Principle 05",
        "og_description": "When something goes wrong with your AI, the worst position is 'we don't know why it said that.' Here's why an audit trail is non-negotiable.",
        "why_matters": """
            <p>When something goes wrong with your AI &mdash; and it will &mdash; the worst possible
            position is &ldquo;we don&rsquo;t know why it said that.&rdquo; An audit trail that captures every
            retrieval, every model verdict, every degradation decision, and every tool call lets
            you replay any user&rsquo;s session and explain it. It&rsquo;s not optional in healthcare,
            finance, legal, or government. It&rsquo;s invaluable everywhere else.</p>
        """,
        "cost_scenario": """
            <p>A customer reports that your AI gave them wrong information. You can&rsquo;t reproduce
            the issue because the system is stochastic. You can&rsquo;t tell them why it answered the
            way it did. They escalate to compliance. Compliance asks for the audit log. There
            isn&rsquo;t one. The incident becomes a company&#8209;wide policy review.</p>
        """,
        "how_brag": """
            <p>Every BRAG run produces a <span class="mono">MemoryLedger</span> &mdash; a complete
            record of every retrieval pass (candidate IDs, scores, filters used), every Verifier
            verdict (with rubric scores and contradictions), every Refutation Agent hypothesis
            (with strategy, target, evidence, verdict), every coverage progression, and every
            degradation decision. The ledger is rendered into per&#8209;query trace HTML &mdash; the same
            data the system used. You can replay any query and show your work step by step.</p>
        """,
        "apply_intro": "Build the ledger from day one. Retrofitting one later is painful.",
        "apply_steps": [
            "Decide upfront: every decision your agent makes goes in a ledger.",
            "Use a single structured format (JSON, protobuf, whatever) &mdash; not log lines.",
            "Capture <em>the evidence the model saw</em>, not just the model&rsquo;s output. Token usage, retrieved IDs, intermediate prompts.",
            "Build a replay UI early, even if it&rsquo;s ugly. You&rsquo;ll use it constantly.",
            "Make the ledger user&#8209;visible when the user asks. Trust comes from being able to show your work.",
        ],
        "apply_outro": "",
    },
    {
        "slug": "degrade-visibly",
        "number": "06",
        "title": "Degrade visibly.",
        "lede": "The worst failure mode is a confident wrong answer; the second worst is a system that fails silently.",
        "og_title": "Degrade visibly &mdash; BRAG Principle 06",
        "og_description": "Build explicit outcome states and surface them in the response. Here's how visible degradation keeps trust where silent failure loses it.",
        "why_matters": """
            <p>There are three failure modes for an AI system: silent (it does the wrong thing
            without telling you), confident&#8209;wrong (it tells you the wrong thing with conviction),
            and visibly degraded (it tells you exactly what it couldn&rsquo;t do). The first two are
            how AI loses trust. The third is how AI keeps trust.</p>
        """,
        "cost_scenario": """
            <p>Your AI is asked a complex question that requires synthesising two facts. It only
            finds one of them. Rather than say &ldquo;I found A but not B, here&rsquo;s A only,&rdquo; it
            confidently answers based on A alone. The user assumes both A and B were considered.
            They make a decision based on incomplete information.</p>
        """,
        "how_brag": """
            <p>BRAG has four explicit outcome states, each with its own degradation handling:
            <strong>Normal</strong> (full answer, all evidence covered),
            <strong>Partial</strong> (answer is incomplete; the disclosed gap explains what&rsquo;s
            missing), <strong>Clarification Request</strong> (the question is ambiguous; the system
            asks for more), and <strong>Hard Halt</strong> (the question is out of scope; the
            system refuses). Every answer is annotated with its degradation level. Every disclosed
            gap, contradiction, and refutation is surfaced in the response itself. The reader sees
            not just the answer, but the <em>quality</em> of the answer.</p>
        """,
        "apply_intro": "Outcome states don&rsquo;t need to be the same as BRAG&rsquo;s. They just need to be explicit.",
        "apply_steps": [
            "Define your outcome states explicitly. (3&ndash;5 is usually enough.)",
            "Surface them in the UI, not just in the logs.",
            "Train your users to read them. A small &ldquo;I&rsquo;m only ~70% confident in this&rdquo; tag is more useful than a confident&#8209;looking long answer.",
            "Track the distribution of outcomes over time. A system that&rsquo;s always &ldquo;Normal&rdquo; is probably hiding partial failures.",
        ],
        "apply_outro": "",
    },
]


def render_page(p, prev_p, next_p):
    apply_steps_html = "\n          ".join(f"<li>{s}</li>" for s in p["apply_steps"])
    apply_outro_html = (
        f'\n        <p>{p["apply_outro"]}</p>'
        if p.get("apply_outro")
        else ""
    )

    prev_link = (
        f'<a href="{prev_p["slug"]}.html"><span class="label">&larr; Previous</span>'
        f'<span class="title">{prev_p["number"]} &middot; {prev_p["title"].rstrip(".")}</span></a>'
        if prev_p
        else '<span class="placeholder"></span>'
    )
    next_link = (
        f'<a href="{next_p["slug"]}.html" class="next"><span class="label">Next &rarr;</span>'
        f'<span class="title">{next_p["number"]} &middot; {next_p["title"].rstrip(".")}</span></a>'
        if next_p
        else '<span class="placeholder"></span>'
    )

    return dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta name="theme-color" content="#13110D">
        <title>{p["og_title"].replace("&mdash;", "—")} &middot; BRAG</title>
        <meta name="description" content="{p["og_description"]}">
        <meta property="og:title" content="{p["og_title"]}">
        <meta property="og:description" content="{p["og_description"]}">
        <meta property="og:type" content="article">
        <link rel="preload" href="../assets/fonts/fraunces-var-latin.woff2" as="font" type="font/woff2" crossorigin>
        <link rel="preload" href="../assets/fonts/inter-var-latin.woff2" as="font" type="font/woff2" crossorigin>
        <link rel="preload" href="../assets/fonts/jetbrains-mono-var-latin.woff2" as="font" type="font/woff2" crossorigin>
        <link rel="stylesheet" href="../assets/css/tokens.css">
        <link rel="stylesheet" href="../assets/css/mini.css">
        </head>
        <body>

        <main class="mini">

          <div class="mini-masthead">
            <a class="mini-back" href="../"><span class="arrow">&larr;</span>BRAG</a>
            <span class="mini-chrome">Principle {p["number"]} of 06</span>
          </div>

          <header class="mini-hero">
            <span class="eyebrow">Principle {p["number"]}</span>
            <div class="number">{p["number"]}</div>
            <h1>{p["title"]}</h1>
            <p class="lede">{p["lede"]}</p>
          </header>

          <section class="mini-section">
            <h2>Why it matters</h2>
            {p["why_matters"].strip()}
          </section>

          <section class="mini-section">
            <h2>The cost of not doing it</h2>
            <div class="mini-callout">
              <span class="label">Scenario</span>
              {p["cost_scenario"].strip()}
            </div>
          </section>

          <section class="mini-section">
            <h2>How BRAG does it</h2>
            {p["how_brag"].strip()}
          </section>

          <section class="mini-section">
            <h2>Apply it in your own system</h2>
            <p>{p["apply_intro"]}</p>
            <ol>
              {apply_steps_html}
            </ol>{apply_outro_html}
          </section>

          <nav class="mini-nav" aria-label="Other principles">
            {prev_link}
            {next_link}
          </nav>

          <section class="mini-cta">
            <p>This is one of <strong>six principles</strong> abstracted from <strong>BRAG</strong>
            &mdash; a retrieval&#8209;augmented generation system with two LLM skeptics gating every
            confident answer. Tested on ten years of Netflix&rsquo;s public financial reporting.</p>
            <div class="mini-cta-actions">
              <a href="../#builders">See all six principles &rarr;</a>
              <a href="../">Read the full showcase &rarr;</a>
              <a href="../#proof">Watch the side&#8209;by&#8209;side proof &rarr;</a>
            </div>
          </section>

          <div class="mini-footer">
            <span>BRAG &middot; Boosted RAG</span>
            <span>by Daniel Wipert</span>
          </div>

        </main>

        </body>
        </html>
    """)


def render_index():
    rows = []
    for p in PRINCIPLES:
        gloss = p["lede"]
        rows.append(
            f'<li><span class="num">{p["number"]}</span>'
            f'<div><a href="{p["slug"]}.html">{p["title"].rstrip(".")}</a>'
            f'<span class="gloss">{gloss}</span></div></li>'
        )
    rows_html = "\n            ".join(rows)
    return dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta name="theme-color" content="#13110D">
        <title>Six principles for trustworthy agentic AI &middot; BRAG</title>
        <meta name="description" content="Six design principles abstracted from BRAG — for builders shipping agentic AI in production.">
        <meta property="og:title" content="Six principles for trustworthy agentic AI">
        <meta property="og:description" content="Design rules abstracted from BRAG for builders shipping agentic AI. Take what's useful — none of this requires BRAG.">
        <meta property="og:type" content="article">
        <link rel="preload" href="../assets/fonts/fraunces-var-latin.woff2" as="font" type="font/woff2" crossorigin>
        <link rel="preload" href="../assets/fonts/inter-var-latin.woff2" as="font" type="font/woff2" crossorigin>
        <link rel="preload" href="../assets/fonts/jetbrains-mono-var-latin.woff2" as="font" type="font/woff2" crossorigin>
        <link rel="stylesheet" href="../assets/css/tokens.css">
        <link rel="stylesheet" href="../assets/css/mini.css">
        </head>
        <body>

        <main class="mini">

          <div class="mini-masthead">
            <a class="mini-back" href="../"><span class="arrow">&larr;</span>BRAG</a>
            <span class="mini-chrome">Principles &middot; index</span>
          </div>

          <header class="mini-hero">
            <span class="eyebrow">For builders</span>
            <h1>Six principles for trustworthy agentic AI.</h1>
            <p class="lede">Design rules abstracted from BRAG, the retrieval&#8209;augmented generation system with two LLM skeptics gating every answer. Take what&rsquo;s useful &mdash; none of this requires BRAG.</p>
          </header>

          <ol class="mini-index">
            {rows_html}
          </ol>

          <section class="mini-cta">
            <div class="mini-cta-actions">
              <a href="../">Read the full BRAG showcase &rarr;</a>
              <a href="../#proof">Watch the side&#8209;by&#8209;side proof &rarr;</a>
            </div>
          </section>

          <div class="mini-footer">
            <span>BRAG &middot; Boosted RAG</span>
            <span>by Daniel Wipert</span>
          </div>

        </main>

        </body>
        </html>
    """)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(PRINCIPLES):
        prev_p = PRINCIPLES[i - 1] if i > 0 else None
        next_p = PRINCIPLES[i + 1] if i + 1 < len(PRINCIPLES) else None
        html = render_page(p, prev_p, next_p)
        path = OUT_DIR / f"{p['slug']}.html"
        path.write_text(html, encoding="utf-8")
        print(f"  ok {path.name}  ({len(html):,} bytes)")
    idx_path = OUT_DIR / "index.html"
    idx_path.write_text(render_index(), encoding="utf-8")
    print(f"  ok {idx_path.name}  ({len(render_index()):,} bytes)")


if __name__ == "__main__":
    main()

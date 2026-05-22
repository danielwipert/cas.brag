"""
Restyle docs/demo/*.html in place so they share the landing-page design system.

Replaces the renderer's inline dark <style> block with links to the shared
tokens.css and trace.css under docs/assets/css/. Also injects a "back to
landing" link at the top of each trace page.

Idempotent: running it twice is a no-op (the marker comment is checked first).

Usage:
    python -m scripts.restyle_demo_traces
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "docs" / "demo"

MARKER = "<!-- BRAG editorial restyle -->"

STYLE_BLOCK_RE = re.compile(r"<style>.*?</style>", re.DOTALL)
HEAD_CLOSE_RE = re.compile(r"</head>", re.IGNORECASE)
BODY_OPEN_RE = re.compile(r"<body[^>]*>", re.IGNORECASE)
CONTAINER_OPEN_RE = re.compile(r'<div\s+class="container"\s*>', re.IGNORECASE)
H1_BRAG_RE = re.compile(r"<h1[^>]*>\s*BRAG Execution Trace\s*</h1>", re.IGNORECASE)

NEW_HEAD_LINKS = (
    f"  {MARKER}\n"
    '  <link rel="stylesheet" href="../assets/css/tokens.css">\n'
    '  <link rel="stylesheet" href="../assets/css/trace.css">\n'
)

BACK_LINK = '<a class="trace-back" href="../index.html">&larr; Back to landing</a>\n\n'


def restyle_trace(html: str, *, is_index: bool) -> str:
    if MARKER in html:
        return html

    # 1. Strip the inline <style>...</style> block
    html = STYLE_BLOCK_RE.sub("", html, count=1)

    # 2. Inject our stylesheet links before </head>
    html = HEAD_CLOSE_RE.sub(NEW_HEAD_LINKS + "</head>", html, count=1)

    if is_index:
        # demo/index.html — different chrome
        html = CONTAINER_OPEN_RE.sub(
            '<div class="container demo-index">\n' + BACK_LINK,
            html,
            count=1,
        )
    else:
        # per-query trace — inject back link after the container opens
        html = CONTAINER_OPEN_RE.sub(
            '<div class="container">\n' + BACK_LINK,
            html,
            count=1,
        )
        # Replace generic h1 with a more editorial label
        html = H1_BRAG_RE.sub(
            "<h1>Execution Trace</h1>",
            html,
            count=1,
        )

    return html


def main() -> None:
    paths = sorted(DEMO_DIR.glob("*.html"))
    if not paths:
        print(f"No html files found under {DEMO_DIR}")
        return

    for p in paths:
        original = p.read_text(encoding="utf-8")
        is_index = p.name == "index.html"
        updated = restyle_trace(original, is_index=is_index)
        if updated == original:
            print(f"  -  {p.name}  (already styled, skipped)")
            continue
        p.write_text(updated, encoding="utf-8")
        sz_before = len(original)
        sz_after = len(updated)
        print(f"  ok {p.name}  ({sz_before:,} -> {sz_after:,} bytes)")


if __name__ == "__main__":
    main()

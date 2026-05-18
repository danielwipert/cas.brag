"""Block 7a: Input Validation gate (deterministic — no LLM).

Spec §3.2. The gate is the constitutional boundary of the pipeline. It
runs six checks in fail-fast order:

  1. UTF-8 encoding — round-trippable bytes.
  2. Length — <= 2,000 tokens (word-count * 1.3 heuristic; no tokenizer dep).
  3. Prompt injection — regex blocklist of known attack patterns.
  4. Language — English-only v1 (ASCII letter ratio + stopword presence).
  5. Scope — soft warning for out-of-window years and competitor names.
  6. Complexity tier — rule-based simple / standard / complex.

Steps 1-4 reject. Step 5 warns. Step 6 classifies. All results are
captured on a single ``ValidationResult`` so the caller can branch on
``passed`` and surface warnings without re-running anything.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from schemas.enums import ComplexityTier


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    rejection_reason: str | None
    warnings: tuple[str, ...]
    complexity_tier: ComplexityTier
    normalized_query: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Conservative token estimate. Real tokenizers (cl100k_base) average ~1.3
# tokens per English word; a word_count * 1.3 cap rejects roughly the same
# queries as a 2,000-token cap without pulling in tiktoken.
_MAX_TOKENS = 2000
_TOKEN_PER_WORD = 1.3

# Corpus temporal window (May 2016 – May 2026). Years outside this range
# in a query trigger a scope warning.
_CORPUS_FIRST_YEAR = 2016
_CORPUS_LAST_YEAR = 2026

# Common English function words. Presence of >=3 distinct hits in a
# non-trivial query is a strong signal the input is English without
# requiring a language-detection library.
_ENGLISH_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "to", "for", "and", "or",
    "is", "was", "are", "were", "be", "been", "has", "have", "had",
    "did", "do", "does", "what", "when", "where", "why", "how", "which",
    "this", "that", "these", "those", "with", "by", "from", "at",
    "as", "but", "if", "than", "then",
    # Pronouns and possessives that show up in legitimate Netflix queries
    # ("did Netflix meet its guidance", "how does it compare", "their").
    "it", "its", "their", "they", "them", "we", "us", "our",
    "all", "more", "most", "any", "some",
})

# Prompt-injection patterns. Kept conservative — false positives reject
# the query. Match the clearly-malicious shapes, not every mention of
# "instructions". Patterns are case-insensitive and anchored on word
# boundaries so they don't fire on legitimate substrings.
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bignore\s+(?:the\s+|all\s+|your\s+|any\s+)?(?:previous|prior|above|preceding)\s+(?:instructions?|prompts?|messages?|rules?)",
        r"\bdisregard\s+(?:the\s+|all\s+|your\s+|any\s+)?(?:previous|prior|above|preceding)\s+(?:instructions?|prompts?|messages?|rules?)",
        r"\breveal\s+(?:your\s+|the\s+)?(?:system\s+)?(?:prompt|instructions?)",
        r"\bshow\s+(?:me\s+)?(?:your\s+|the\s+)?(?:system\s+)?(?:prompt|instructions?)",
        r"\bwhat\s+(?:are|is)\s+your\s+(?:system\s+)?(?:prompt|instructions?)",
        r"\byou\s+are\s+now\s+",
        r"\bfrom\s+now\s+on\s*,?\s+you\s+",
        r"\bforget\s+(?:everything|all|your\s+(?:previous|prior))",
        r"\bact\s+as\s+(?:a\s+|an\s+)?(?:different|new|other)",
        r"\bpretend\s+(?:you\s+are|to\s+be)\s+",
        r"<\s*\|?\s*(?:system|im_start|im_end|endoftext)\s*\|?\s*>",
    ]
)

# Competitor names trigger a scope warning (proceed, don't reject).
# Matches common rebrands too.
_COMPETITORS_RE = re.compile(
    r"\b(?:Disney(?:\+)?|Hulu|HBO(?:\s*Max)?|Max\b|Amazon\s+Prime|Prime\s+Video|"
    r"Apple\s+TV(?:\+)?|Paramount(?:\+)?|Peacock|YouTube\s+TV|Spotify|Roku|"
    r"Tubi|Pluto\s*TV|Comcast|AT&T)\b",
    re.IGNORECASE,
)

# Period extraction. Each regex returns a (year, optional_quarter) tuple.
# Used both for scope warnings and complexity classification.
_PERIOD_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bQ([1-4])\s*(20\d{2})\b", re.IGNORECASE),          # "Q3 2024"
    re.compile(r"\b(20\d{2})\s*Q([1-4])\b", re.IGNORECASE),          # "2024Q3"
    re.compile(r"\bFY\s*(20\d{2})\b", re.IGNORECASE),                # "FY2023"
    re.compile(r"\b(20\d{2})\b"),                                    # "2018"
)

# Cues that bump complexity. Detected as whole-word case-insensitive.
_EVOLUTION_CUES = re.compile(
    r"\b(?:evolve|evolved|evolves|evolution|trajectory|over\s+time|"
    r"shift(?:ed|ing)?|transition(?:ed|ing|s)?|"
    r"change[ds]?\s+over|trend(?:ed|ing|s)?)\b",
    re.IGNORECASE,
)
_COMPARISON_CUES = re.compile(
    r"\b(?:compare[ds]?|comparison|versus|vs\.?|v\.|between\s+\S+\s+and\s+\S+|"
    r"did\s+\w+\s+meet|did\s+\w+\s+hit)\b",
    re.IGNORECASE,
)
_NUMERICAL_CUES = re.compile(
    r"\b(?:what\s+(?:was|were|is|are)|how\s+much|how\s+many|"
    r"revenue|income|margin|cash\s+flow|earnings|EPS|memberships?|"
    r"subscribers?|net\s+adds?|operating\s+income|free\s+cash\s+flow)\b",
    re.IGNORECASE,
)
_NARRATIVE_CUES = re.compile(
    r"\b(?:why\s+(?:did|does|was|is)|how\s+did|stance|strategy|"
    r"position|drove|drivers?|explain|impact|effect|reason|rationale|"
    r"narrative|story)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize(query: str) -> str:
    """NFC unicode-normalize and collapse internal whitespace."""
    nfc = unicodedata.normalize("NFC", query)
    return re.sub(r"\s+", " ", nfc).strip()


def _estimate_tokens(text: str) -> int:
    """Word count * 1.3 — conservative approximation of cl100k_base."""
    return int(len(text.split()) * _TOKEN_PER_WORD)


def _looks_english(text: str) -> bool:
    """Heuristic: query is plausibly English if (a) at least 80% of its
    letter characters are ASCII a-z and (b) for queries of >=6 words,
    it contains at least 2 distinct common English stopwords. Shorter
    queries pass on the ASCII check alone since they may legitimately
    contain few or no stopwords (e.g. "Netflix Q3 2024 revenue")."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    ascii_letters = sum(1 for c in letters if "a" <= c.lower() <= "z")
    if ascii_letters / len(letters) < 0.80:
        return False
    # Use >=2-char words only, so contractions like "what's" don't inflate
    # the word count via the orphan "s" fragment. Short noun-phrase queries
    # (e.g., "Netflix accounting policy for content amortization") may have
    # only one stopword; we trust the ASCII-ratio signal on queries under
    # 8 words and only require stopwords on longer ones.
    words = [w for w in re.findall(r"\b[a-z]+\b", text.lower()) if len(w) >= 2]
    if len(words) < 8:
        return True
    distinct_stopwords = {w for w in words if w in _ENGLISH_STOPWORDS}
    return len(distinct_stopwords) >= 2


def _detect_injection(text: str) -> re.Pattern[str] | None:
    for pat in _INJECTION_PATTERNS:
        if pat.search(text):
            return pat
    return None


def _extract_years(text: str) -> set[int]:
    """All 4-digit year mentions in the text (any pattern)."""
    years: set[int] = set()
    for m in re.finditer(r"\b(20\d{2})\b", text):
        years.add(int(m.group(1)))
    return years


def _count_distinct_periods(text: str) -> int:
    """Count distinct (year, quarter|FY|year-only) tokens. A standalone
    year and a Q-qualified version of the same year count as one period."""
    periods: set[tuple[int, str]] = set()
    for m in _PERIOD_PATTERNS[0].finditer(text):  # "Q3 2024"
        periods.add((int(m.group(2)), f"Q{m.group(1)}"))
    for m in _PERIOD_PATTERNS[1].finditer(text):  # "2024Q3"
        periods.add((int(m.group(1)), f"Q{m.group(2)}"))
    for m in _PERIOD_PATTERNS[2].finditer(text):  # "FY2023"
        periods.add((int(m.group(1)), "FY"))
    # Bare year mentions only count if not already captured by Q/FY.
    captured_years = {y for y, _ in periods}
    for m in _PERIOD_PATTERNS[3].finditer(text):
        y = int(m.group(1))
        if y not in captured_years:
            periods.add((y, "Y"))
    return len(periods)


def _classify_tier(text: str) -> ComplexityTier:
    n_periods = _count_distinct_periods(text)
    evolution = bool(_EVOLUTION_CUES.search(text))
    comparison = bool(_COMPARISON_CUES.search(text))
    has_numerical = bool(_NUMERICAL_CUES.search(text))
    has_narrative = bool(_NARRATIVE_CUES.search(text))
    mixed = has_numerical and has_narrative

    # Complex: cross-document multi-period with narrative+numerical mixing.
    # A 3+-period query OR an evolution-cued query that spans multiple
    # periods OR mixes narrative and numerical reading qualifies.
    if n_periods >= 3:
        return ComplexityTier.complex
    if evolution and (mixed or n_periods >= 2):
        return ComplexityTier.complex
    # Standard: a 2-period comparison, or a query with comparison/evolution
    # cues but only one (or zero) anchored period.
    if n_periods >= 2 or comparison or evolution:
        return ComplexityTier.standard
    return ComplexityTier.simple


def _scope_warnings(text: str) -> list[str]:
    warnings: list[str] = []
    competitor_match = _COMPETITORS_RE.search(text)
    if competitor_match:
        warnings.append(
            f"Query references a competitor ({competitor_match.group(0)}); "
            f"BRAG's corpus is limited to Netflix's public filings."
        )
    out_of_window = sorted(
        y for y in _extract_years(text)
        if y < _CORPUS_FIRST_YEAR or y > _CORPUS_LAST_YEAR
    )
    if out_of_window:
        warnings.append(
            f"Query references year(s) outside the corpus window "
            f"({_CORPUS_FIRST_YEAR}–{_CORPUS_LAST_YEAR}): "
            f"{', '.join(str(y) for y in out_of_window)}."
        )
    return warnings


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate(query: str) -> ValidationResult:
    """Run all gate checks and return a ``ValidationResult``.

    Hard rejection sets ``passed=False`` and ``rejection_reason``;
    scope failures attach warnings but do not block. Complexity tier
    is always populated (defaults to ``simple`` on rejection paths so
    downstream code has a non-null tier)."""
    if not isinstance(query, str) or not query:
        return ValidationResult(
            passed=False,
            rejection_reason="Empty or non-string query.",
            warnings=(),
            complexity_tier=ComplexityTier.simple,
            normalized_query="",
        )

    # 1. UTF-8 round-trip. Strings reaching this function are already
    #    Python str (unicode), so the practical check is that they
    #    encode to UTF-8 without surrogate errors.
    try:
        query.encode("utf-8")
    except UnicodeEncodeError as exc:
        return ValidationResult(
            passed=False,
            rejection_reason=f"Query is not valid UTF-8: {exc}",
            warnings=(),
            complexity_tier=ComplexityTier.simple,
            normalized_query="",
        )

    normalized = _normalize(query)

    # 2. Length.
    n_tokens = _estimate_tokens(normalized)
    if n_tokens > _MAX_TOKENS:
        return ValidationResult(
            passed=False,
            rejection_reason=(
                f"Query is too long: ~{n_tokens} tokens "
                f"(limit {_MAX_TOKENS})."
            ),
            warnings=(),
            complexity_tier=ComplexityTier.simple,
            normalized_query=normalized,
        )

    # 3. Prompt injection.
    injection = _detect_injection(normalized)
    if injection is not None:
        return ValidationResult(
            passed=False,
            rejection_reason=(
                "Query matches a known prompt-injection pattern and was "
                "rejected. Rephrase the question as a direct query about "
                "Netflix's public reporting."
            ),
            warnings=(),
            complexity_tier=ComplexityTier.simple,
            normalized_query=normalized,
        )

    # 4. Language.
    if not _looks_english(normalized):
        return ValidationResult(
            passed=False,
            rejection_reason=(
                "Query does not appear to be English. BRAG v1 supports "
                "English-language queries only."
            ),
            warnings=(),
            complexity_tier=ComplexityTier.simple,
            normalized_query=normalized,
        )

    # 5. Scope warnings (do not reject).
    warnings = tuple(_scope_warnings(normalized))

    # 6. Complexity tier.
    tier = _classify_tier(normalized)

    return ValidationResult(
        passed=True,
        rejection_reason=None,
        warnings=warnings,
        complexity_tier=tier,
        normalized_query=normalized,
    )

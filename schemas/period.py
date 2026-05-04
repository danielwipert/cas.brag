"""Canonical period strings for BRAG.

Spec §3.3 / Block 4 note: periods must round-trip through a single canonical
format so Verifier numerical-exactness and period-integrity checks can compare
them as strings without ambiguity.

Supported forms:
    quarter           "2024Q3"
    fiscal_year       "FY2023"
    instant           "2024-09-30"
    fy_guidance       "FY2025-guidance"
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

_QUARTER_RE = re.compile(r"^(\d{4})Q([1-4])$")
_FY_RE = re.compile(r"^FY(\d{4})$")
_FY_GUIDANCE_RE = re.compile(r"^FY(\d{4})-guidance$")
_INSTANT_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


@dataclass(frozen=True)
class Period:
    kind: str  # "quarter" | "fiscal_year" | "instant" | "fy_guidance"
    year: int
    quarter: int | None = None
    instant: date | None = None

    def __str__(self) -> str:
        return format_period(self)


def parse_period(s: str) -> Period:
    if not isinstance(s, str) or not s:
        raise ValueError(f"period must be a non-empty string, got {s!r}")

    m = _QUARTER_RE.match(s)
    if m:
        return Period(kind="quarter", year=int(m.group(1)), quarter=int(m.group(2)))

    m = _FY_RE.match(s)
    if m:
        return Period(kind="fiscal_year", year=int(m.group(1)))

    m = _FY_GUIDANCE_RE.match(s)
    if m:
        return Period(kind="fy_guidance", year=int(m.group(1)))

    m = _INSTANT_RE.match(s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return Period(kind="instant", year=y, instant=date(y, mo, d))
        except ValueError as e:
            raise ValueError(f"invalid instant date in period {s!r}: {e}") from e

    raise ValueError(
        f"unrecognized period format: {s!r} "
        "(expected YYYYQN, FYYYYY, FYYYYY-guidance, or YYYY-MM-DD)"
    )


def format_period(p: Period) -> str:
    if p.kind == "quarter":
        if p.quarter is None:
            raise ValueError("quarter period missing quarter component")
        return f"{p.year}Q{p.quarter}"
    if p.kind == "fiscal_year":
        return f"FY{p.year}"
    if p.kind == "fy_guidance":
        return f"FY{p.year}-guidance"
    if p.kind == "instant":
        if p.instant is None:
            raise ValueError("instant period missing instant component")
        return p.instant.isoformat()
    raise ValueError(f"unknown period kind: {p.kind!r}")


def is_valid_period(s: str) -> bool:
    try:
        parse_period(s)
    except ValueError:
        return False
    return True

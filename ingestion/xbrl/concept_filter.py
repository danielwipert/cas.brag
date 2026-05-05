"""Canonical XBRL retention set + label / statement-section / dimension maps.

Block 4 design decisions (2026-05-04):
- Retention set is hardcoded (~30 concepts) and matches the build plan's list.
  Adding a concept means updating three dicts in this file: ``_LABELS``,
  ``_STATEMENT_SECTIONS``, and (only if the concept has dimensional cuts to
  retain) the dimension policy below.
- Geographic dimensional facts are kept for the four Netflix-published
  regions (UCAN / EMEA / LATAM / APAC) on ``srt:StatementGeographicalAxis``.
  All other dimension axes are dropped.
- ``srt:ProductOrServiceAxis = nflx:StreamingMember`` is treated as a
  no-op disambiguator (post-DVD, streaming = total). A context carrying
  only Streaming is collapsed to the aggregate; a context carrying
  Streaming + a region is treated as the region's regional fact.

Hardcoded labels avoid the XBRL label-linkbase fetch (which would require
network + the schema/.xsd file). The labels match the standard us-gaap
linkbase conventions.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Retention set + human labels
# ---------------------------------------------------------------------------


_LABELS: dict[str, str] = {
    # --- Income statement -------------------------------------------------
    "us-gaap:Revenues": "Revenues",
    "us-gaap:CostOfRevenue": "Cost of revenues",
    "us-gaap:GrossProfit": "Gross profit",
    "us-gaap:OperatingExpenses": "Operating expenses",
    "us-gaap:OperatingIncomeLoss": "Operating income (loss)",
    "us-gaap:NetIncomeLoss": "Net income (loss)",
    "us-gaap:EarningsPerShareBasic": "Earnings per share, basic",
    "us-gaap:EarningsPerShareDiluted": "Earnings per share, diluted",
    "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic": "Weighted-average shares outstanding, basic",
    "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding": "Weighted-average shares outstanding, diluted",
    "us-gaap:IncomeTaxExpenseBenefit": "Provision for income taxes",
    "us-gaap:InterestExpense": "Interest expense",
    "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": "Income before income taxes",
    "us-gaap:MarketingExpense": "Marketing expense",
    "us-gaap:GeneralAndAdministrativeExpense": "General and administrative expense",
    "us-gaap:ResearchAndDevelopmentExpense": "Technology and development expense",
    # --- Balance sheet ----------------------------------------------------
    "us-gaap:CashAndCashEquivalentsAtCarryingValue": "Cash and cash equivalents",
    "us-gaap:Assets": "Total assets",
    "us-gaap:AssetsCurrent": "Total current assets",
    "us-gaap:Liabilities": "Total liabilities",
    "us-gaap:LiabilitiesCurrent": "Total current liabilities",
    "us-gaap:StockholdersEquity": "Stockholders' equity",
    "us-gaap:LongTermDebtNoncurrent": "Long-term debt",
    "us-gaap:LongTermDebtCurrent": "Current portion of long-term debt",
    "us-gaap:AccountsPayableCurrent": "Accounts payable",
    # --- Cash flow --------------------------------------------------------
    "us-gaap:NetCashProvidedByUsedInOperatingActivities": "Net cash provided by operating activities",
    "us-gaap:NetCashProvidedByUsedInInvestingActivities": "Net cash provided by (used in) investing activities",
    "us-gaap:NetCashProvidedByUsedInFinancingActivities": "Net cash provided by (used in) financing activities",
    "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect": "Net change in cash, cash equivalents, and restricted cash",
    # --- Netflix-specific -------------------------------------------------
    "nflx:ContentAssetsNet": "Content assets, net",
    "nflx:ContentLiabilitiesCurrent": "Content liabilities, current",
    "nflx:ContentLiabilitiesNoncurrent": "Content liabilities, noncurrent",
    "nflx:AdditionstoStreamingContentAssets": "Additions to content assets",
    "nflx:CostofServicesAmortizationofStreamingContentAssets": "Amortization of content assets",
    "nflx:NumberOfPaidMemberships": "Number of paid memberships",
    "nflx:NumberOfPaidMembershipAdditionsLossesDuringPeriod": "Paid memberships net additions",
}


# Concept -> rendered statement section. Used as ``FactRecord.source_section``.
_STATEMENT_SECTIONS: dict[str, str] = {
    # Income statement
    "us-gaap:Revenues": "Condensed Consolidated Statements of Operations",
    "us-gaap:CostOfRevenue": "Condensed Consolidated Statements of Operations",
    "us-gaap:GrossProfit": "Condensed Consolidated Statements of Operations",
    "us-gaap:OperatingExpenses": "Condensed Consolidated Statements of Operations",
    "us-gaap:OperatingIncomeLoss": "Condensed Consolidated Statements of Operations",
    "us-gaap:NetIncomeLoss": "Condensed Consolidated Statements of Operations",
    "us-gaap:EarningsPerShareBasic": "Condensed Consolidated Statements of Operations",
    "us-gaap:EarningsPerShareDiluted": "Condensed Consolidated Statements of Operations",
    "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic": "Condensed Consolidated Statements of Operations",
    "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding": "Condensed Consolidated Statements of Operations",
    "us-gaap:IncomeTaxExpenseBenefit": "Condensed Consolidated Statements of Operations",
    "us-gaap:InterestExpense": "Condensed Consolidated Statements of Operations",
    "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": "Condensed Consolidated Statements of Operations",
    "us-gaap:MarketingExpense": "Condensed Consolidated Statements of Operations",
    "us-gaap:GeneralAndAdministrativeExpense": "Condensed Consolidated Statements of Operations",
    "us-gaap:ResearchAndDevelopmentExpense": "Condensed Consolidated Statements of Operations",
    # Balance sheet
    "us-gaap:CashAndCashEquivalentsAtCarryingValue": "Condensed Consolidated Balance Sheets",
    "us-gaap:Assets": "Condensed Consolidated Balance Sheets",
    "us-gaap:AssetsCurrent": "Condensed Consolidated Balance Sheets",
    "us-gaap:Liabilities": "Condensed Consolidated Balance Sheets",
    "us-gaap:LiabilitiesCurrent": "Condensed Consolidated Balance Sheets",
    "us-gaap:StockholdersEquity": "Condensed Consolidated Balance Sheets",
    "us-gaap:LongTermDebtNoncurrent": "Condensed Consolidated Balance Sheets",
    "us-gaap:LongTermDebtCurrent": "Condensed Consolidated Balance Sheets",
    "us-gaap:AccountsPayableCurrent": "Condensed Consolidated Balance Sheets",
    "nflx:ContentAssetsNet": "Condensed Consolidated Balance Sheets",
    "nflx:ContentLiabilitiesCurrent": "Condensed Consolidated Balance Sheets",
    "nflx:ContentLiabilitiesNoncurrent": "Condensed Consolidated Balance Sheets",
    # Cash flow
    "us-gaap:NetCashProvidedByUsedInOperatingActivities": "Condensed Consolidated Statements of Cash Flows",
    "us-gaap:NetCashProvidedByUsedInInvestingActivities": "Condensed Consolidated Statements of Cash Flows",
    "us-gaap:NetCashProvidedByUsedInFinancingActivities": "Condensed Consolidated Statements of Cash Flows",
    "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect": "Condensed Consolidated Statements of Cash Flows",
    "nflx:AdditionstoStreamingContentAssets": "Condensed Consolidated Statements of Cash Flows",
    "nflx:CostofServicesAmortizationofStreamingContentAssets": "Condensed Consolidated Statements of Cash Flows",
    # Netflix-specific operational metrics (segment / disclosures)
    "nflx:NumberOfPaidMemberships": "Segment Information and Operating Metrics",
    "nflx:NumberOfPaidMembershipAdditionsLossesDuringPeriod": "Segment Information and Operating Metrics",
}


# ---------------------------------------------------------------------------
# Dimension policy
# ---------------------------------------------------------------------------


# Geographic axis we accept members from.
GEOGRAPHIC_AXIS = "srt:StatementGeographicalAxis"

# Map XBRL geographic member tag -> short Netflix-published region label.
_GEOGRAPHIC_MEMBERS: dict[str, str] = {
    "nflx:UnitedStatesAndCanadaMember": "UCAN",
    "us-gaap:EMEAMember": "EMEA",
    "srt:LatinAmericaMember": "LATAM",
    "srt:AsiaPacificMember": "APAC",
}

# Product-or-service axis dimensions that are treated as no-ops because, post-
# DVD shutdown, all Netflix revenue is streaming. The (Streaming) dimension
# carries no information beyond the aggregate.
_PRODUCT_AXIS = "srt:ProductOrServiceAxis"
_NOOP_PRODUCT_MEMBERS: frozenset[str] = frozenset({"nflx:StreamingMember"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_canonical(concept: str) -> bool:
    """True iff ``concept`` is in the retention set."""
    return concept in _LABELS


def human_label(concept: str) -> str:
    """Return the human-readable label for a concept. Raises KeyError for
    non-retained concepts — callers should guard with ``is_canonical``."""
    return _LABELS[concept]


def statement_section(concept: str) -> str:
    """Return the rendered statement section name for a retained concept."""
    return _STATEMENT_SECTIONS[concept]


def classify_dimensions(dimensions: dict[str, str]) -> tuple[str, str | None]:
    """Classify a fact's dimensions for our retention policy.

    ``dimensions`` is a mapping ``{axis_tag: member_tag}`` extracted from the
    fact's context segment. Returns one of three classifications, with an
    optional region label:

        ("aggregate", None)        — keep, no segment
        ("regional", <region>)     — keep, regional fact (region is e.g. "UCAN")
        ("drop",     None)         — drop, dimensions outside our policy

    Decision rules:
        - Drop the (ProductOrServiceAxis, StreamingMember) entry as a no-op.
        - Empty dimensions after that filter -> aggregate.
        - Exactly one remaining dimension on the geographic axis with a
          whitelisted member -> regional with that label.
        - Anything else -> drop.
    """
    filtered: dict[str, str] = {}
    for axis, member in dimensions.items():
        if axis == _PRODUCT_AXIS and member in _NOOP_PRODUCT_MEMBERS:
            continue
        filtered[axis] = member

    if not filtered:
        return ("aggregate", None)

    if len(filtered) == 1:
        axis, member = next(iter(filtered.items()))
        if axis == GEOGRAPHIC_AXIS and member in _GEOGRAPHIC_MEMBERS:
            return ("regional", _GEOGRAPHIC_MEMBERS[member])

    return ("drop", None)


def all_canonical_concepts() -> tuple[str, ...]:
    """Used by tests + by smoke logging."""
    return tuple(sorted(_LABELS.keys()))

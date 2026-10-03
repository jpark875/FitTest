"""Comparable selection and valuation: what a listing should resell for, and how sure we are."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal

from rapidfuzz import fuzz

from mta.models.listing import (
    ItemCondition,
    RawListing,
    TrendAssessment,
    Valuation,
)

_CENTS = Decimal("0.01")

#: Share of a mint item's price a condition typically retains. UNKNOWN is the harshest
#: non-poor discount so a missing label costs a buy rather than creating one.
CONDITION_FACTOR: dict[ItemCondition, Decimal] = {
    ItemCondition.NEW_WITH_TAGS: Decimal("1.00"),
    ItemCondition.EXCELLENT: Decimal("0.95"),
    ItemCondition.GOOD: Decimal("0.85"),
    ItemCondition.FAIR: Decimal("0.70"),
    ItemCondition.POOR: Decimal("0.50"),
    ItemCondition.UNKNOWN: Decimal("0.65"),
}

#: Comps at which the comp median and the model estimate carry equal weight.
COMP_HALF_WEIGHT = 3

#: Title similarity (0-100) a candidate needs to count as comparable.
MIN_TITLE_SIMILARITY = 55


@dataclass(frozen=True)
class Comparable:
    """A previously seen listing used as a price reference."""

    fingerprint: str
    title: str
    price: Decimal
    currency: str
    condition: ItemCondition
    category: str
    brand: str | None


def select_comparables(
    listing: RawListing,
    assessment: TrendAssessment,
    candidates: Iterable[Comparable],
    *,
    min_similarity: float = MIN_TITLE_SIMILARITY,
) -> list[Comparable]:
    """Keep candidates in the same currency and category that read like the same item."""
    chosen = []
    for candidate in candidates:
        if candidate.fingerprint == listing.fingerprint or candidate.currency != listing.currency:
            continue
        if candidate.category != assessment.inferred_category:
            continue
        if (
            assessment.inferred_brand
            and candidate.brand
            and assessment.inferred_brand.lower() != candidate.brand.lower()
        ):
            continue
        if fuzz.token_set_ratio(listing.title.lower(), candidate.title.lower()) >= min_similarity:
            chosen.append(candidate)
    return chosen


def comp_weight(count: int) -> float:
    """How much to trust comps over the model estimate: 0 with none, approaching 1."""
    return count / (count + COMP_HALF_WEIGHT)


def _comp_estimate(comps: list[Comparable], condition: ItemCondition) -> Decimal | None:
    """Median comp price, restated at mint and re-discounted to the listing's condition."""
    if not comps:
        return None
    mint = [float(c.price / CONDITION_FACTOR[c.condition]) for c in comps]
    return (Decimal(str(statistics.median(mint))) * CONDITION_FACTOR[condition]).quantize(_CENTS)


def arbitrage_score(margin_pct: Decimal, confidence: float) -> float:
    """Rank scalar in [0, 1]: confidence times a saturating function of return on capital."""
    return round(confidence * math.tanh(max(float(margin_pct), 0.0)), 4)


def value_listing(
    listing: RawListing,
    assessment: TrendAssessment,
    comps: list[Comparable],
    *,
    platform_fee_rate: Decimal,
    shipping_cost: Decimal,
) -> Valuation:
    """Blend the comp median with the model's estimate and compute margins net of fees.

    With no estimate from either source the resale value falls back to the asking price
    and confidence is zero, so the row ranks last instead of vanishing.
    """
    model_estimate = assessment.estimated_retail_value
    comp_estimate = _comp_estimate(comps, listing.condition)
    weight = comp_weight(len(comps)) if comp_estimate is not None else 0.0

    if comp_estimate is not None and model_estimate is not None:
        blended = comp_estimate * Decimal(str(weight)) + model_estimate * Decimal(str(1 - weight))
    elif comp_estimate is not None:
        blended = comp_estimate
    elif model_estimate is not None:
        blended = model_estimate
    else:
        blended = listing.asking_price

    if comp_estimate is None and model_estimate is None:
        confidence = 0.0
    else:
        # Without comps the model's claim is uncorroborated, so it is discounted by a fifth.
        confidence = assessment.style_confidence * (0.8 + 0.2 * weight)

    median = (
        Decimal(str(statistics.median(float(c.price) for c in comps))).quantize(_CENTS)
        if comps
        else None
    )
    valuation = Valuation(
        asking_price=listing.asking_price,
        currency=listing.currency,
        estimated_resale_value=blended.quantize(_CENTS),
        comparable_count=len(comps),
        median_comp_price=median,
        platform_fee_rate=platform_fee_rate,
        shipping_cost=shipping_cost,
        valuation_confidence=round(confidence, 4),
        arbitrage_score=0.0,
    )
    score = arbitrage_score(valuation.margin_pct, valuation.valuation_confidence)
    return valuation.model_copy(update={"arbitrage_score": score})

"""Warehouse to DataFrame, kept free of Streamlit so it can be tested directly."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

import pandas as pd

from mta.models.listing import TrendTag
from mta.storage.repository import Opportunity, Repository

COLUMNS = [
    "title",
    "platform",
    "trend",
    "brand",
    "category",
    "condition",
    "price",
    "resale",
    "profit",
    "return_pct",
    "confidence",
    "score",
    "comps",
    "model",
    "reasoning",
    "url",
]


def to_frame(items: Sequence[Opportunity]) -> pd.DataFrame:
    """Flatten opportunities into one row each, money as floats for charting."""
    rows = []
    for item in items:
        listing, assessment, valuation = item.listing, item.assessment, item.valuation
        rows.append(
            {
                "title": listing.title,
                "platform": listing.platform.value,
                "trend": assessment.primary_trend.value,
                "brand": assessment.inferred_brand or listing.brand_raw or "",
                "category": assessment.inferred_category,
                "condition": listing.condition.value,
                "price": float(valuation.asking_price),
                "resale": float(valuation.estimated_resale_value),
                "profit": float(valuation.absolute_margin),
                "return_pct": float(valuation.margin_pct) * 100,
                "confidence": valuation.valuation_confidence,
                "score": valuation.arbitrage_score,
                "comps": valuation.comparable_count,
                "model": assessment.model_id,
                "reasoning": assessment.reasoning,
                "url": str(listing.url),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def load_frame(
    repository: Repository,
    *,
    min_margin_pct: Decimal,
    min_absolute_margin: Decimal,
    min_confidence: float,
    trends: Sequence[TrendTag] | None = None,
) -> pd.DataFrame:
    """Opportunities clearing every floor, best first. On-trend listings only."""
    items = [
        o
        for o in repository.opportunities(
            min_margin_pct=min_margin_pct,
            min_absolute_margin=min_absolute_margin,
            min_confidence=min_confidence,
            trends=trends,
        )
        if o.assessment.is_on_trend
    ]
    return to_frame(items)

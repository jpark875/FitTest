"""Shared test fixtures.

The suite runs offline: HTTP tests mock at the httpx transport layer, and
ingestion tests replay recorded payloads from tests/fixtures/.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest import mock

import httpx._config
import httpx._transports.default
import pytest

from mta.models.listing import (
    ItemCondition,
    Platform,
    RawListing,
    TrendAssessment,
    TrendTag,
    Valuation,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def _shared_trust_store() -> Iterator[None]:
    """Load the CA bundle once rather than once per httpx client.

    Each AsyncClient reads certifi's PEM into a fresh SSLContext, which costs
    about half a second on Windows. Invisible in production (one client per
    run), but the HTTP tests build one per test. Patched where the name is
    looked up, since the transport module binds it at import time.
    """
    cached = httpx._config.create_ssl_context()
    with mock.patch.object(httpx._transports.default, "create_ssl_context", return_value=cached):
        yield


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def raw_listing() -> RawListing:
    return RawListing(
        platform=Platform.DEPOP,
        external_id="abc123",
        url="https://example.invalid/products/abc123",
        title="Vintage Carhartt Detroit jacket",
        description="90s, blanket lined.",
        asking_price=Decimal("25.00"),
        currency="GBP",
        condition=ItemCondition.GOOD,
        brand_raw="Carhartt",
        size_raw="L",
        seller_username="northern_thrift",
        image_urls=["https://media.example.invalid/abc123/1.jpg"],
        listed_at=datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
    )


@pytest.fixture
def assessment() -> TrendAssessment:
    return TrendAssessment(
        trend_tags=[TrendTag.WORKWEAR, TrendTag.GRUNGE],
        inferred_brand="Carhartt",
        inferred_category="outerwear",
        inferred_decade="1990s",
        style_confidence=0.82,
        estimated_retail_value=Decimal("90.00"),
        reasoning="Duck canvas, blanket lining and a squared cut read as 90s workwear.",
        model_id="claude-opus-5",
        prompt_version="v1",
    )


@pytest.fixture
def valuation() -> Valuation:
    return Valuation(
        asking_price=Decimal("25.00"),
        currency="GBP",
        estimated_resale_value=Decimal("90.00"),
        comparable_count=7,
        median_comp_price=Decimal("88.00"),
        platform_fee_rate=Decimal("0.10"),
        shipping_cost=Decimal("4.50"),
        valuation_confidence=0.78,
        arbitrage_score=0.71,
    )

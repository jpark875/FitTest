"""Contract tests for the domain models.

The money arithmetic is the most expensive thing here to get wrong and the
cheapest to test, so it is covered exhaustively.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from mta.models.listing import (
    MAX_IMAGES_PER_LISTING,
    EnrichedListing,
    ItemCondition,
    MalformedListingError,
    Platform,
    RawListing,
    TrendAssessment,
    TrendTag,
    Valuation,
    parse_condition,
    parse_price,
)

THRESHOLDS = {
    "min_margin_pct": Decimal("0.30"),
    "min_confidence": 0.60,
    "min_absolute_margin": Decimal("10.00"),
}


class TestParsePrice:
    @pytest.mark.parametrize(
        ("raw", "amount", "currency"),
        [
            ("£24", Decimal("24.00"), "GBP"),
            ("24.00 GBP", Decimal("24.00"), "GBP"),
            ("$1,299", Decimal("1299.00"), "USD"),
            ("€18,50", Decimal("18.50"), "EUR"),
            ("18", Decimal("18.00"), "GBP"),
            ("£20 ono", Decimal("20.00"), "GBP"),
            ("  £1,299.99  ", Decimal("1299.99"), "GBP"),
            ("¥5000", Decimal("5000.00"), "JPY"),
        ],
    )
    def test_parses_seller_authored_prices(self, raw: str, amount: Decimal, currency: str) -> None:
        assert parse_price(raw) == (amount, currency)

    def test_iso_code_beats_the_default(self) -> None:
        # Reading only the symbol would file 34.50 EUR as 34.50 GBP.
        assert parse_price("34,50 EUR", default_currency="GBP") == (Decimal("34.50"), "EUR")

    def test_unmarked_price_takes_the_default(self) -> None:
        assert parse_price("42.00", default_currency="USD") == (Decimal("42.00"), "USD")

    @pytest.mark.parametrize("raw", ["", "   ", "FREE", "offers", "make me an offer :)"])
    def test_rejects_unpriced_listings(self, raw: str) -> None:
        # Defaulting to zero would manufacture an infinite-margin opportunity.
        with pytest.raises(MalformedListingError):
            parse_price(raw)


class TestParseCondition:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("New With Tags", ItemCondition.NEW_WITH_TAGS),
            ("Brand new", ItemCondition.NEW_WITH_TAGS),
            ("Used - like new", ItemCondition.EXCELLENT),
            ("Excellent", ItemCondition.EXCELLENT),
            ("Very good", ItemCondition.GOOD),
            ("Used - fair", ItemCondition.FAIR),
            ("Heavily distressed", ItemCondition.POOR),
        ],
    )
    def test_normalizes_platform_vocabularies(self, raw: str, expected: ItemCondition) -> None:
        assert parse_condition(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "pre-loved sparkle"])
    def test_unknown_vocabulary_degrades(self, raw: str | None) -> None:
        assert parse_condition(raw) is ItemCondition.UNKNOWN


class TestRawListing:
    def test_fingerprint_is_platform_scoped(self, raw_listing: RawListing) -> None:
        assert raw_listing.fingerprint == "depop:abc123"

    def test_currency_is_uppercased(self) -> None:
        listing = RawListing(
            platform=Platform.FIXTURE,
            external_id="1",
            url="https://example.invalid/1",
            title="t",
            asking_price=Decimal("1.00"),
            currency="gbp",
        )
        assert listing.currency == "GBP"

    def test_extra_images_are_trimmed_not_rejected(self) -> None:
        images = [f"https://media.example.invalid/{i}.jpg" for i in range(12)]
        listing = RawListing(
            platform=Platform.FIXTURE,
            external_id="1",
            url="https://example.invalid/1",
            title="t",
            asking_price=Decimal("1.00"),
            currency="GBP",
            image_urls=images,
        )
        assert len(listing.image_urls) == MAX_IMAGES_PER_LISTING

    def test_is_analyzable_requires_images_and_a_price(self) -> None:
        base = {
            "platform": Platform.FIXTURE,
            "external_id": "1",
            "url": "https://example.invalid/1",
            "title": "t",
            "currency": "GBP",
        }
        image = ["https://i.invalid/1.jpg"]

        assert RawListing(**base, asking_price=Decimal("10.00"), image_urls=image).is_analyzable
        assert not RawListing(**base, asking_price=Decimal("10.00")).is_analyzable
        assert not RawListing(**base, asking_price=Decimal("0.00"), image_urls=image).is_analyzable

    def test_is_frozen(self, raw_listing: RawListing) -> None:
        with pytest.raises(ValidationError):
            raw_listing.asking_price = Decimal("30.00")  # type: ignore[misc]

    def test_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RawListing(
                platform=Platform.FIXTURE,
                external_id="1",
                url="https://example.invalid/1",
                title="t",
                asking_price=Decimal("1.00"),
                currency="GBP",
                seller_rating=4.8,
            )

    def test_naive_datetimes_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RawListing(
                platform=Platform.FIXTURE,
                external_id="1",
                url="https://example.invalid/1",
                title="t",
                asking_price=Decimal("1.00"),
                currency="GBP",
                listed_at=datetime(2026, 7, 1, 12, 0),
            )


class TestTrendAssessment:
    def test_primary_trend_is_the_first_tag(self, assessment: TrendAssessment) -> None:
        assert assessment.primary_trend is TrendTag.WORKWEAR
        assert assessment.is_on_trend

    def test_none_tag_means_off_trend(self, assessment: TrendAssessment) -> None:
        assert not assessment.model_copy(update={"trend_tags": [TrendTag.NONE]}).is_on_trend

    def test_requires_at_least_one_tag(self, assessment: TrendAssessment) -> None:
        with pytest.raises(ValidationError):
            TrendAssessment(**{**assessment.model_dump(), "trend_tags": []})


class TestValuation:
    def test_net_proceeds_are_after_commission_and_postage(self, valuation: Valuation) -> None:
        # £90 gross, 10% commission, £4.50 postage.
        assert valuation.net_proceeds == Decimal("76.50")

    def test_absolute_margin_is_net_of_the_buy_price(self, valuation: Valuation) -> None:
        assert valuation.absolute_margin == Decimal("51.50")

    def test_gross_margin_would_overstate_this(self, valuation: Valuation) -> None:
        gross = valuation.estimated_resale_value - valuation.asking_price
        assert gross == Decimal("65.00")
        assert valuation.absolute_margin < gross

    def test_margin_pct_is_return_on_capital(self, valuation: Valuation) -> None:
        assert valuation.margin_pct == Decimal("2.0600")

    def test_margin_can_be_negative(self) -> None:
        losing = Valuation(
            asking_price=Decimal("80.00"),
            currency="GBP",
            estimated_resale_value=Decimal("85.00"),
            comparable_count=3,
            valuation_confidence=0.5,
            arbitrage_score=0.0,
        )
        assert losing.absolute_margin < 0
        assert losing.margin_pct < 0

    def test_free_items_do_not_divide_by_zero(self) -> None:
        free = Valuation(
            asking_price=Decimal("0.00"),
            currency="GBP",
            estimated_resale_value=Decimal("20.00"),
            comparable_count=1,
            valuation_confidence=0.5,
            arbitrage_score=0.0,
        )
        assert free.margin_pct == Decimal("0.00")

    def test_rejects_a_malformed_currency_code(self) -> None:
        with pytest.raises(ValidationError):
            Valuation(
                asking_price=Decimal("10.00"),
                currency="12£",
                estimated_resale_value=Decimal("20.00"),
                comparable_count=1,
                valuation_confidence=0.5,
                arbitrage_score=0.0,
            )


class TestEnrichedListing:
    def test_composes_the_three_stages(
        self, raw_listing: RawListing, assessment: TrendAssessment, valuation: Valuation
    ) -> None:
        enriched = EnrichedListing(listing=raw_listing, assessment=assessment, valuation=valuation)
        assert enriched.fingerprint == raw_listing.fingerprint

    def test_rejects_a_valuation_from_another_listing(
        self, raw_listing: RawListing, assessment: TrendAssessment, valuation: Valuation
    ) -> None:
        mismatched = valuation.model_copy(update={"asking_price": Decimal("99.00")})
        with pytest.raises(ValidationError, match="does not match"):
            EnrichedListing(listing=raw_listing, assessment=assessment, valuation=mismatched)

    def test_rejects_a_currency_mismatch(
        self, raw_listing: RawListing, assessment: TrendAssessment, valuation: Valuation
    ) -> None:
        mismatched = valuation.model_copy(update={"currency": "USD"})
        with pytest.raises(ValidationError, match="Currency mismatch"):
            EnrichedListing(listing=raw_listing, assessment=assessment, valuation=mismatched)

    def test_meets_threshold_on_a_good_opportunity(
        self, raw_listing: RawListing, assessment: TrendAssessment, valuation: Valuation
    ) -> None:
        enriched = EnrichedListing(listing=raw_listing, assessment=assessment, valuation=valuation)
        assert enriched.meets_threshold(**THRESHOLDS)

    def test_low_confidence_is_excluded(
        self, raw_listing: RawListing, assessment: TrendAssessment, valuation: Valuation
    ) -> None:
        enriched = EnrichedListing(
            listing=raw_listing,
            assessment=assessment,
            valuation=valuation.model_copy(update={"valuation_confidence": 0.10}),
        )
        assert not enriched.meets_threshold(**THRESHOLDS)

    def test_an_off_trend_listing_never_surfaces(
        self, raw_listing: RawListing, assessment: TrendAssessment, valuation: Valuation
    ) -> None:
        enriched = EnrichedListing(
            listing=raw_listing,
            assessment=assessment.model_copy(update={"trend_tags": [TrendTag.NONE]}),
            valuation=valuation,
        )
        assert not enriched.meets_threshold(**THRESHOLDS)

    def test_small_absolute_profits_are_filtered_out(
        self, raw_listing: RawListing, assessment: TrendAssessment, valuation: Valuation
    ) -> None:
        # A high percentage on a tiny buy is not worth the postage.
        cheap = raw_listing.model_copy(update={"asking_price": Decimal("2.00")})
        thin = valuation.model_copy(
            update={
                "asking_price": Decimal("2.00"),
                "estimated_resale_value": Decimal("10.00"),
                "shipping_cost": Decimal("4.50"),
            }
        )
        enriched = EnrichedListing(listing=cheap, assessment=assessment, valuation=thin)

        assert enriched.valuation.margin_pct > Decimal("0.30")
        assert not enriched.meets_threshold(**THRESHOLDS)

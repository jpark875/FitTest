"""Tests for comparable selection and valuation arithmetic."""

from __future__ import annotations

from decimal import Decimal

import pytest

from mta.models.listing import ItemCondition, RawListing, TrendAssessment
from mta.processing.valuation import (
    CONDITION_FACTOR,
    Comparable,
    arbitrage_score,
    comp_weight,
    select_comparables,
    value_listing,
)

FEE = Decimal("0.10")
POSTAGE = Decimal("4.50")


def comp(
    fingerprint: str,
    title: str = "Vintage Carhartt Detroit jacket",
    price: str = "80.00",
    condition: ItemCondition = ItemCondition.EXCELLENT,
    **kwargs: object,
) -> Comparable:
    defaults: dict[str, object] = {
        "currency": "GBP",
        "category": "outerwear",
        "brand": "Carhartt",
    }
    return Comparable(
        fingerprint=fingerprint,
        title=title,
        price=Decimal(price),
        condition=condition,
        **{**defaults, **kwargs},  # type: ignore[arg-type]
    )


class TestSelectComparables:
    def test_keeps_similar_items_in_same_currency_and_category(
        self, raw_listing: RawListing, assessment: TrendAssessment
    ) -> None:
        candidates = [
            comp("a"),
            comp("b", currency="USD"),
            comp("c", category="denim"),
            comp("d", title="Completely unrelated ballgown"),
            comp("e", brand="Dickies"),
            comp(raw_listing.fingerprint),
        ]
        chosen = select_comparables(raw_listing, assessment, candidates)
        assert [c.fingerprint for c in chosen] == ["a"]

    def test_unbranded_candidate_is_not_excluded_by_brand(
        self, raw_listing: RawListing, assessment: TrendAssessment
    ) -> None:
        chosen = select_comparables(raw_listing, assessment, [comp("a", brand=None)])
        assert len(chosen) == 1


class TestValueListing:
    def test_model_only_valuation(
        self, raw_listing: RawListing, assessment: TrendAssessment
    ) -> None:
        valuation = value_listing(
            raw_listing, assessment, [], platform_fee_rate=FEE, shipping_cost=POSTAGE
        )
        assert valuation.estimated_resale_value == Decimal("90.00")
        assert valuation.comparable_count == 0
        assert valuation.median_comp_price is None
        # 90 - 9.00 fees - 4.50 postage = 76.50 proceeds on a 25.00 buy.
        assert valuation.net_proceeds == Decimal("76.50")
        assert valuation.absolute_margin == Decimal("51.50")
        assert valuation.valuation_confidence == pytest.approx(0.82 * 0.8, abs=1e-4)

    def test_comps_pull_the_estimate_and_raise_confidence(
        self, raw_listing: RawListing, assessment: TrendAssessment
    ) -> None:
        comps = [comp(str(i), price="60.00", condition=ItemCondition.GOOD) for i in range(3)]
        with_comps = value_listing(
            raw_listing, assessment, comps, platform_fee_rate=FEE, shipping_cost=POSTAGE
        )
        without = value_listing(
            raw_listing, assessment, [], platform_fee_rate=FEE, shipping_cost=POSTAGE
        )
        # Comps restated at mint (60 / 0.85) then discounted to the listing's GOOD condition
        # land back on 60; with weight 0.5 against the model's 90 that blends to 75.
        assert with_comps.estimated_resale_value == Decimal("75.00")
        assert with_comps.median_comp_price == Decimal("60.00")
        assert with_comps.valuation_confidence > without.valuation_confidence

    def test_condition_discount_applies_to_comp_estimate(
        self, raw_listing: RawListing, assessment: TrendAssessment
    ) -> None:
        poor = raw_listing.model_copy(update={"condition": ItemCondition.POOR})
        mint_comps = [
            comp(str(i), price="100.00", condition=ItemCondition.NEW_WITH_TAGS) for i in range(3)
        ]
        bare = assessment.model_copy(update={"estimated_retail_value": None})

        valuation = value_listing(
            poor, bare, mint_comps, platform_fee_rate=FEE, shipping_cost=POSTAGE
        )

        assert valuation.estimated_resale_value == 100 * CONDITION_FACTOR[ItemCondition.POOR]

    def test_no_evidence_ranks_last_with_zero_confidence(
        self, raw_listing: RawListing, assessment: TrendAssessment
    ) -> None:
        blind = assessment.model_copy(update={"estimated_retail_value": None})
        valuation = value_listing(
            raw_listing, blind, [], platform_fee_rate=FEE, shipping_cost=POSTAGE
        )
        assert valuation.valuation_confidence == 0.0
        assert valuation.arbitrage_score == 0.0
        assert valuation.absolute_margin < 0

    def test_score_is_stored_on_the_valuation(
        self, raw_listing: RawListing, assessment: TrendAssessment
    ) -> None:
        valuation = value_listing(
            raw_listing, assessment, [], platform_fee_rate=FEE, shipping_cost=POSTAGE
        )
        assert valuation.arbitrage_score == arbitrage_score(
            valuation.margin_pct, valuation.valuation_confidence
        )
        assert 0.0 < valuation.arbitrage_score <= 1.0


def test_comp_weight_rises_with_depth() -> None:
    assert comp_weight(0) == 0.0
    assert comp_weight(3) == 0.5
    assert comp_weight(30) > 0.9


def test_score_ignores_negative_margins() -> None:
    assert arbitrage_score(Decimal("-0.4"), 0.9) == 0.0
    assert arbitrage_score(Decimal("0.5"), 1.0) > arbitrage_score(Decimal("0.5"), 0.5)

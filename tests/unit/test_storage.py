"""Tests for the SQLite warehouse."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from mta.models.listing import RawListing, TrendAssessment, TrendTag, Valuation
from mta.storage import Repository


@pytest.fixture
def repository(tmp_path: Path) -> Repository:
    repo = Repository(f"sqlite:///{tmp_path / 'nested' / 'mta.db'}")
    repo.create_schema()
    return repo


def store_all(
    repo: Repository, listing: RawListing, assessment: TrendAssessment, valuation: Valuation
) -> None:
    repo.upsert_listing(listing)
    repo.upsert_assessment(listing.fingerprint, assessment)
    repo.upsert_valuation(listing.fingerprint, valuation)


def test_round_trip_preserves_every_field(
    repository: Repository,
    raw_listing: RawListing,
    assessment: TrendAssessment,
    valuation: Valuation,
) -> None:
    store_all(repository, raw_listing, assessment, valuation)

    [found] = repository.opportunities()

    assert found.listing.model_dump(exclude={"captured_at"}) == raw_listing.model_dump(
        exclude={"captured_at"}
    )
    assert found.listing.captured_at == raw_listing.captured_at
    assert found.assessment == assessment
    assert found.valuation == valuation
    assert found.enriched.valuation.absolute_margin == valuation.absolute_margin


def test_upserts_are_idempotent_and_refresh_the_price(
    repository: Repository,
    raw_listing: RawListing,
    assessment: TrendAssessment,
    valuation: Valuation,
) -> None:
    assert repository.upsert_listing(raw_listing) is True
    assert repository.upsert_listing(raw_listing) is False
    cheaper = raw_listing.model_copy(update={"asking_price": Decimal("19.99")})
    repository.upsert_listing(cheaper)
    repository.upsert_assessment(raw_listing.fingerprint, assessment)
    repository.upsert_assessment(raw_listing.fingerprint, assessment)
    repository.upsert_valuation(raw_listing.fingerprint, valuation)
    repository.upsert_valuation(raw_listing.fingerprint, valuation)

    assert repository.count_listings() == 1
    assert len(repository.opportunities()) == 1
    assert repository.comparables("outerwear", "GBP")[0].price == Decimal("19.99")


def test_a_new_prompt_version_adds_an_assessment_row(
    repository: Repository,
    raw_listing: RawListing,
    assessment: TrendAssessment,
    valuation: Valuation,
) -> None:
    store_all(repository, raw_listing, assessment, valuation)
    newer = assessment.model_copy(
        update={"prompt_version": "v2", "assessed_at": datetime(2030, 1, 1, tzinfo=UTC)}
    )
    repository.upsert_assessment(raw_listing.fingerprint, newer)
    repository.upsert_valuation(raw_listing.fingerprint, valuation)

    [found] = repository.opportunities()
    assert found.assessment.prompt_version == "v2"


def test_valuation_needs_a_stored_listing_and_assessment(
    repository: Repository, raw_listing: RawListing, valuation: Valuation
) -> None:
    with pytest.raises(LookupError, match="not stored"):
        repository.upsert_valuation(raw_listing.fingerprint, valuation)
    repository.upsert_listing(raw_listing)
    with pytest.raises(LookupError, match="no assessment"):
        repository.upsert_valuation(raw_listing.fingerprint, valuation)


def test_opportunity_filters_and_ordering(
    repository: Repository,
    raw_listing: RawListing,
    assessment: TrendAssessment,
    valuation: Valuation,
) -> None:
    store_all(repository, raw_listing, assessment, valuation)
    second = raw_listing.model_copy(update={"external_id": "zzz999"})
    weak_assessment = assessment.model_copy(update={"trend_tags": [TrendTag.Y2K]})
    weak = valuation.model_copy(update={"arbitrage_score": 0.1, "valuation_confidence": 0.3})
    store_all(repository, second, weak_assessment, weak)

    ranked = repository.opportunities()
    assert [o.listing.external_id for o in ranked] == ["abc123", "zzz999"]

    confident = repository.opportunities(min_confidence=0.5)
    assert [o.listing.external_id for o in confident] == ["abc123"]

    y2k = repository.opportunities(trends=[TrendTag.Y2K])
    assert [o.listing.external_id for o in y2k] == ["zzz999"]

    assert repository.opportunities(min_absolute_margin=Decimal("1000")) == []
    assert len(repository.opportunities(limit=1)) == 1


def test_comparables_exclude_the_subject_and_other_currencies(
    repository: Repository, raw_listing: RawListing, assessment: TrendAssessment
) -> None:
    repository.upsert_listing(raw_listing)
    repository.upsert_assessment(raw_listing.fingerprint, assessment)
    euro = raw_listing.model_copy(update={"external_id": "eu1", "currency": "EUR"})
    repository.upsert_listing(euro)
    repository.upsert_assessment(euro.fingerprint, assessment)

    assert repository.comparables("outerwear", "GBP", exclude=raw_listing.fingerprint) == []
    assert [c.fingerprint for c in repository.comparables("outerwear", "GBP")] == [
        raw_listing.fingerprint
    ]
    assert repository.comparables("denim", "GBP") == []


def test_trend_summary(
    repository: Repository,
    raw_listing: RawListing,
    assessment: TrendAssessment,
    valuation: Valuation,
) -> None:
    store_all(repository, raw_listing, assessment, valuation)
    [(trend, count, mean)] = repository.trend_summary()
    assert (trend, count) == ("workwear", 1)
    assert mean == pytest.approx(float(valuation.margin_pct))


def test_money_survives_exactly(
    repository: Repository,
    raw_listing: RawListing,
    assessment: TrendAssessment,
    valuation: Valuation,
) -> None:
    odd = raw_listing.model_copy(update={"asking_price": Decimal("0.29")})
    odd_valuation = valuation.model_copy(update={"asking_price": Decimal("0.29")})
    store_all(repository, odd, assessment, odd_valuation)
    [found] = repository.opportunities()
    assert found.listing.asking_price == Decimal("0.29")

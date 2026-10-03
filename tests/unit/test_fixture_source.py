"""Tests for the offline fixture replay source.

These back the claim that a fresh clone runs the pipeline end to end with no
network and no credentials: they read the same recordings the default run reads.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from mta.ingestion.base import PermanentIngestionError, SearchQuery
from mta.ingestion.fixtures import FixtureSource
from mta.models.listing import MAX_IMAGES_PER_LISTING, Platform, RawListing

DEFAULT_QUERY = SearchQuery(keywords="alternative streetwear")
CARHARTT_QUERY = SearchQuery(keywords="vintage carhartt jacket")


async def collect(source: FixtureSource, query: SearchQuery) -> list[RawListing]:
    return [listing async for listing in source.fetch_listings(query)]


def by_id(listings: list[RawListing], external_id: str) -> RawListing:
    return next(listing for listing in listings if listing.external_id == external_id)


def write_fixture(directory: Path, name: str, listings: list[dict[str, object]]) -> Path:
    path = directory / name
    path.write_text(json.dumps({"listings": listings}), encoding="utf-8")
    return path


@pytest.fixture
def source(fixtures_dir: Path) -> FixtureSource:
    return FixtureSource(fixtures_dir)


class TestOfflineReplay:
    async def test_runs_end_to_end_with_no_network(self, source: FixtureSource) -> None:
        listings = await collect(source, DEFAULT_QUERY)
        assert len(listings) == 10
        assert all(listing.platform is Platform.FIXTURE for listing in listings)

    async def test_paginates_and_stops(self, source: FixtureSource) -> None:
        # Two recorded pages plus the empty third that ends pagination.
        await collect(source, DEFAULT_QUERY)
        assert source.stats.pages_fetched == 3

    async def test_reports_honest_statistics(self, source: FixtureSource) -> None:
        await collect(source, DEFAULT_QUERY)
        stats = source.stats
        assert stats.records_seen == 13
        assert stats.listings_yielded == 10
        assert stats.listings_skipped == 2
        assert stats.duplicates_suppressed == 1

    async def test_fingerprints_are_platform_scoped(self, source: FixtureSource) -> None:
        listings = await collect(source, DEFAULT_QUERY)
        assert by_id(listings, "fx-1001").fingerprint == "fixture:fx-1001"


class TestRecordMapping:
    async def test_maps_a_well_formed_record(self, source: FixtureSource) -> None:
        listing = by_id(await collect(source, DEFAULT_QUERY), "fx-1001")
        assert listing.asking_price == Decimal("48.00")
        assert listing.currency == "GBP"
        assert listing.brand_raw == "Carhartt"
        assert listing.seller_username == "northern_thrift"
        assert listing.condition.value == "good"
        assert listing.listed_at is not None

    async def test_honours_an_explicit_currency_code(self, source: FixtureSource) -> None:
        listing = by_id(await collect(source, DEFAULT_QUERY), "fx-1012")
        assert (listing.asking_price, listing.currency) == (Decimal("34.50"), "EUR")

    async def test_parses_thousands_separators(self, source: FixtureSource) -> None:
        listing = by_id(await collect(source, DEFAULT_QUERY), "fx-1011")
        assert listing.asking_price == Decimal("1299.00")

    async def test_an_unmarked_price_takes_the_default(self, fixtures_dir: Path) -> None:
        source = FixtureSource(fixtures_dir, default_currency="usd")
        listing = by_id(await collect(source, DEFAULT_QUERY), "fx-1005")
        assert (listing.asking_price, listing.currency) == (Decimal("18.00"), "USD")

    async def test_extra_images_are_trimmed_not_dropped(self, source: FixtureSource) -> None:
        listing = by_id(await collect(source, DEFAULT_QUERY), "fx-1009")
        assert len(listing.image_urls) == MAX_IMAGES_PER_LISTING

    async def test_a_listing_with_no_images_is_kept(self, source: FixtureSource) -> None:
        # Still worth storing, just not worth a vision call.
        listing = by_id(await collect(source, DEFAULT_QUERY), "fx-1010")
        assert not listing.is_analyzable


class TestRecordLevelFailures:
    async def test_an_unpriced_listing_is_skipped(self, source: FixtureSource) -> None:
        listings = await collect(source, DEFAULT_QUERY)
        assert all(listing.external_id != "fx-1007" for listing in listings)
        assert source.stats.skip_reasons.get("ListingParseError") == 2

    async def test_a_record_without_a_url_is_skipped(self, source: FixtureSource) -> None:
        listings = await collect(source, DEFAULT_QUERY)
        assert all(listing.external_id != "fx-1008" for listing in listings)

    async def test_a_record_without_an_id_is_skipped(self, tmp_path: Path) -> None:
        write_fixture(tmp_path, "default-p1.json", [{"price": "£10", "url": "https://x.invalid/1"}])
        source = FixtureSource(tmp_path)
        assert await collect(source, DEFAULT_QUERY) == []
        assert source.stats.skip_reasons == {"ListingParseError": 1}

    async def test_the_recorded_skip_rate_stays_below_the_alarm(
        self, source: FixtureSource
    ) -> None:
        # If a fixture or parser change pushes this over the threshold, the
        # suite should say so rather than a log line six weeks later.
        await collect(source, DEFAULT_QUERY)
        assert source.stats.skip_rate < 0.20


class TestFixtureResolution:
    async def test_a_query_specific_recording_wins(self, source: FixtureSource) -> None:
        listings = await collect(source, CARHARTT_QUERY)
        assert [x.external_id for x in listings] == ["fx-2001", "fx-2002", "fx-2003", "fx-2004"]

    async def test_a_recording_does_not_spill_onto_the_default_set(
        self, source: FixtureSource
    ) -> None:
        listings = await collect(source, CARHARTT_QUERY)
        assert all(listing.external_id.startswith("fx-2") for listing in listings)

    async def test_an_unrecorded_query_falls_back(self, source: FixtureSource) -> None:
        listings = await collect(source, SearchQuery(keywords="something nobody recorded"))
        assert by_id(listings, "fx-1001")


class TestStructuralFailures:
    async def test_a_missing_fixtures_directory_is_loud(self, tmp_path: Path) -> None:
        source = FixtureSource(tmp_path / "does-not-exist")
        with pytest.raises(PermanentIngestionError, match="No fixture found"):
            await collect(source, DEFAULT_QUERY)

    async def test_corrupt_json_is_loud(self, tmp_path: Path) -> None:
        (tmp_path / "default-p1.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(PermanentIngestionError, match="not valid JSON"):
            await collect(FixtureSource(tmp_path), DEFAULT_QUERY)

    async def test_a_payload_that_is_not_an_object_is_loud(self, tmp_path: Path) -> None:
        (tmp_path / "default-p1.json").write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(PermanentIngestionError, match="must be a JSON object"):
            await collect(FixtureSource(tmp_path), DEFAULT_QUERY)

    async def test_a_payload_without_a_listings_array_is_loud(self, tmp_path: Path) -> None:
        (tmp_path / "default-p1.json").write_text('{"items": []}', encoding="utf-8")
        with pytest.raises(PermanentIngestionError, match="no 'listings' array"):
            await collect(FixtureSource(tmp_path), DEFAULT_QUERY)

    async def test_an_unreadable_fixture_is_loud(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_fixture(tmp_path, "default-p1.json", [])

        def explode(*args: object, **kwargs: object) -> str:
            raise OSError("permission denied")

        monkeypatch.setattr(Path, "read_text", explode)
        with pytest.raises(PermanentIngestionError, match="Could not read fixture"):
            await collect(FixtureSource(tmp_path), DEFAULT_QUERY)

    async def test_non_mapping_entries_are_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "default-p1.json").write_text(
            json.dumps(
                {
                    "listings": [
                        "a bare string",
                        None,
                        {
                            "id": "ok-1",
                            "url": "https://fixture.invalid/products/ok-1",
                            "title": "Fine",
                            "price": "£10.00",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        listings = await collect(FixtureSource(tmp_path), DEFAULT_QUERY)
        assert [listing.external_id for listing in listings] == ["ok-1"]

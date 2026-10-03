"""Tests for the shared scraping loop.

Every adapter inherits the template method, so a bug here is a bug in every
platform at once. It is driven through a fake source whose pages are scripted
per test, including pages that raise.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from mta.ingestion.base import (
    SUSPICIOUS_SKIP_RATE,
    IngestionStats,
    ListingParseError,
    ListingSource,
    PermanentIngestionError,
    RateLimitedError,
    SearchQuery,
    TransientIngestionError,
)
from mta.models.listing import Platform, RawListing

QUERY = SearchQuery(keywords="carhartt jacket", max_pages=5)


def page(*ids: str, **flags: object) -> str:
    """Render a page payload holding one record per id."""
    return json.dumps([{"id": i, "price": "£10.00", **flags} for i in ids])


class FakeSource(ListingSource):
    """A source whose pages are scripted: payloads, or exceptions to raise."""

    platform: ClassVar[Platform] = Platform.FIXTURE

    def __init__(self, pages: list[str | Exception], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.pages = pages
        self.requested_pages: list[int] = []
        self.closed = False

    async def _fetch_page(self, query: SearchQuery, page_number: int) -> str:
        self.requested_pages.append(page_number)
        if page_number > len(self.pages):
            return "[]"
        scripted = self.pages[page_number - 1]
        if isinstance(scripted, Exception):
            raise scripted
        return scripted

    def _extract_records(self, payload: str) -> Iterable[Mapping[str, Any]]:
        return json.loads(payload)

    def _to_listing(self, record: Mapping[str, Any], query: SearchQuery) -> RawListing:
        if record.get("bad"):
            raise ListingParseError(f"record {record['id']} is unusable")
        return RawListing(
            platform=self.platform,
            external_id=str(record["id"]),
            url=f"https://fixture.invalid/products/{record['id']}",
            title=f"Item {record['id']}",
            asking_price=Decimal("10.00"),
            currency="GBP",
        )

    async def aclose(self) -> None:
        self.closed = True


class RecordingLimiter:
    def __init__(self) -> None:
        self.penalties: list[float] = []

    async def acquire(self, tokens: float = 1.0) -> float:
        return 0.0

    def penalize(self, seconds: float) -> None:
        self.penalties.append(seconds)


async def collect(source: ListingSource, query: SearchQuery = QUERY) -> list[RawListing]:
    return [listing async for listing in source.fetch_listings(query)]


class TestSearchQuery:
    def test_slug_is_filesystem_safe(self) -> None:
        assert SearchQuery(keywords="Vintage Carhartt Jacket!").slug == "vintage-carhartt-jacket"

    def test_slug_collapses_punctuation(self) -> None:
        assert SearchQuery(keywords="y2k  ///  cargo").slug == "y2k-cargo"

    def test_slug_is_bounded(self) -> None:
        assert len(SearchQuery(keywords="x " * 100).slug) <= 80

    def test_is_frozen_and_closed(self) -> None:
        with pytest.raises(ValidationError):
            SearchQuery(keywords="x", colour="red")


class TestPagination:
    async def test_yields_listings_across_pages(self) -> None:
        source = FakeSource([page("a", "b"), page("c")])
        assert [x.external_id for x in await collect(source)] == ["a", "b", "c"]
        assert source.stats.pages_fetched == 3

    async def test_stops_when_a_page_yields_nothing(self) -> None:
        source = FakeSource([page("a"), "[]", page("never-reached")])
        assert len(await collect(source)) == 1
        assert source.requested_pages == [1, 2]

    async def test_respects_the_page_ceiling(self) -> None:
        source = FakeSource([page(f"item-{i}") for i in range(50)])
        await collect(source, SearchQuery(keywords="x", max_pages=3))
        assert source.requested_pages == [1, 2, 3]

    async def test_the_next_page_hook_can_be_overridden(self) -> None:
        class SinglePageSource(FakeSource):
            def _has_next_page(self, payload: str, page: int, records_found: int) -> bool:
                return False

        assert len(await collect(SinglePageSource([page("a"), page("b")]))) == 1

    async def test_an_empty_first_page_is_flagged(self, caplog: pytest.LogCaptureFixture) -> None:
        # Nothing on page one is either an empty market or a broken extractor.
        source = FakeSource(["[]"])
        with caplog.at_level(logging.WARNING, logger="mta.ingestion.base"):
            assert await collect(source) == []
        assert "extractor is broken" in caplog.text


class TestDeduplication:
    async def test_the_same_item_on_two_pages_is_yielded_once(self) -> None:
        source = FakeSource([page("a", "b"), page("b", "c")])
        assert [x.external_id for x in await collect(source)] == ["a", "b", "c"]
        assert source.stats.duplicates_suppressed == 1

    async def test_duplicates_within_one_page_are_suppressed(self) -> None:
        assert len(await collect(FakeSource([page("a", "a")]))) == 1


class TestPerRecordErrorIsolation:
    async def test_one_bad_record_does_not_cost_the_page(self) -> None:
        payload = json.dumps(
            [
                {"id": "good-1", "price": "£10.00"},
                {"id": "bad-1", "price": "£10.00", "bad": True},
                {"id": "good-2", "price": "£10.00"},
            ]
        )
        source = FakeSource([payload])
        assert [x.external_id for x in await collect(source)] == ["good-1", "good-2"]
        assert source.stats.listings_skipped == 1

    async def test_skips_are_attributed_to_a_reason(self) -> None:
        source = FakeSource([page("bad-1", bad=True)])
        await collect(source)
        assert source.stats.skip_reasons == {"ListingParseError": 1}

    async def test_a_validation_failure_is_also_isolated(self) -> None:
        class BrokenMappingSource(FakeSource):
            def _to_listing(self, record: Mapping[str, Any], query: SearchQuery) -> RawListing:
                return RawListing(
                    platform=self.platform,
                    external_id="",  # violates min_length
                    url="https://fixture.invalid/x",
                    title="x",
                    asking_price=Decimal("1.00"),
                    currency="GBP",
                )

        source = BrokenMappingSource([page("a")])
        assert await collect(source) == []
        assert source.stats.skip_reasons == {"ValidationError": 1}

    async def test_a_high_skip_rate_is_escalated(self, caplog: pytest.LogCaptureFixture) -> None:
        records = [{"id": f"r{i}", "price": "£10.00", "bad": i % 2 == 0} for i in range(20)]
        source = FakeSource([json.dumps(records)])
        with caplog.at_level(logging.ERROR, logger="mta.ingestion.base"):
            await collect(source)
        assert source.stats.skip_rate > SUSPICIOUS_SKIP_RATE
        assert "parser has drifted" in caplog.text

    async def test_a_normal_skip_rate_is_not_escalated(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        records = [{"id": f"r{i}", "price": "£10.00", "bad": i == 0} for i in range(20)]
        with caplog.at_level(logging.ERROR, logger="mta.ingestion.base"):
            await collect(FakeSource([json.dumps(records)]))
        assert "parser has drifted" not in caplog.text


class TestStructuralFailures:
    async def test_an_unexpected_extractor_failure_becomes_permanent(self) -> None:
        class BrokenExtractorSource(FakeSource):
            def _extract_records(self, payload: str) -> Iterable[Mapping[str, Any]]:
                raise AttributeError("'NoneType' object has no attribute 'find_all'")

        with pytest.raises(PermanentIngestionError, match="markup has probably changed"):
            await collect(BrokenExtractorSource([page("a")]))

    async def test_a_permanent_error_propagates_unwrapped(self) -> None:
        class RedesignedSource(FakeSource):
            def _extract_records(self, payload: str) -> Iterable[Mapping[str, Any]]:
                raise PermanentIngestionError("container element missing entirely")

        with pytest.raises(PermanentIngestionError, match="container element missing"):
            await collect(RedesignedSource([page("a")]))

    async def test_a_fetch_level_permanent_error_is_not_swallowed(self) -> None:
        with pytest.raises(PermanentIngestionError):
            await collect(FakeSource([PermanentIngestionError("403, likely bot detection")]))


class TestDegradedRuns:
    async def test_a_transient_failure_keeps_the_partial_results(self) -> None:
        source = FakeSource([page("a", "b"), TransientIngestionError("connection reset")])
        assert [x.external_id for x in await collect(source)] == ["a", "b"]
        assert source.stats.transient_failures == 1
        assert source.requested_pages == [1, 2]

    async def test_a_rate_limit_stops_the_query_and_penalises_the_limiter(self) -> None:
        # The penalty lands on the shared limiter, so every source pointed at
        # that host slows down, not just this query.
        limiter = RecordingLimiter()
        source = FakeSource(
            [page("a"), RateLimitedError("429", retry_after=120.0)], rate_limiter=limiter
        )
        assert len(await collect(source)) == 1
        assert limiter.penalties == [120.0]
        assert source.stats.transient_failures == 1

    async def test_a_rate_limit_without_a_delay_still_stops_the_query(self) -> None:
        limiter = RecordingLimiter()
        source = FakeSource([RateLimitedError("429")], rate_limiter=limiter)
        assert await collect(source) == []
        assert limiter.penalties == []


class TestLifecycle:
    async def test_the_context_manager_closes_the_source(self) -> None:
        source = FakeSource([page("a")])
        async with source:
            await collect(source)
        assert source.closed

    async def test_the_source_is_closed_even_when_the_body_raises(self) -> None:
        source = FakeSource([page("a")])
        with pytest.raises(RuntimeError):
            async with source:
                raise RuntimeError("downstream blew up")
        assert source.closed


class TestIngestionStats:
    def test_skip_rate_of_an_empty_run_is_zero(self) -> None:
        assert IngestionStats().skip_rate == 0.0

    def test_skip_rate_is_a_fraction_of_records_seen(self) -> None:
        stats = IngestionStats(records_seen=10)
        stats.record_skip("ListingParseError")
        stats.record_skip("ValidationError")
        assert stats.skip_rate == 0.2
        assert stats.skip_reasons == {"ListingParseError": 1, "ValidationError": 1}

    def test_summary_reports_every_counter(self) -> None:
        stats = IngestionStats(
            pages_fetched=3,
            records_seen=10,
            listings_yielded=7,
            listings_skipped=2,
            duplicates_suppressed=1,
            transient_failures=1,
            seconds_throttled=12.5,
        )
        summary = stats.summary()
        assert "7 listings from 3 page(s)" in summary
        assert "skipped 2 (20.0%)" in summary
        assert "12.5s throttled" in summary

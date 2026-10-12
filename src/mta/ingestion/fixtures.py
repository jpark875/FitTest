"""Offline replay of recorded marketplace payloads."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, ClassVar

from pydantic import ValidationError

from mta.ingestion.base import (
    ListingParseError,
    ListingSource,
    PermanentIngestionError,
    SearchQuery,
)
from mta.ingestion.rate_limiter import RateLimiter
from mta.models.listing import (
    MalformedListingError,
    Platform,
    RawListing,
    parse_condition,
    parse_price,
)

logger = logging.getLogger(__name__)

#: Slug used when no fixture matches the query.
DEFAULT_FIXTURE_SLUG = "default"

#: Returned for a page with no recorded file. The base class stops paginating
#: on a page that yields no records, so this ends a short fixture set.
_EMPTY_PAYLOAD = '{"listings": []}'


class FixtureSource(ListingSource):
    """Replays recorded payloads from disk as if they came from a marketplace."""

    platform: ClassVar[Platform] = Platform.FIXTURE

    def __init__(
        self,
        fixtures_dir: Path,
        *,
        default_currency: str = "GBP",
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        """Point the source at a directory of recorded payloads."""
        super().__init__(rate_limiter=rate_limiter)
        self.fixtures_dir = fixtures_dir
        self.default_currency = default_currency.upper()

    async def _fetch_page(self, query: SearchQuery, page: int) -> str:
        """Read one recorded page off disk."""
        # Spend a token even though nothing leaves the machine, so the
        # throttled and unthrottled paths stay identical.
        self.stats.seconds_throttled += await self.rate_limiter.acquire()

        path = self._page_path(query, page)
        if path is None:
            if page == 1:
                raise PermanentIngestionError(
                    f"No fixture found for query {query.slug!r} page 1 in "
                    f"{self.fixtures_dir}. Expected {query.slug}-p1.json or "
                    f"{DEFAULT_FIXTURE_SLUG}-p1.json."
                )
            return _EMPTY_PAYLOAD

        try:
            # Off the event loop: irrelevant for a few kilobytes, but this
            # source should not model I/O differently from the ones it stands
            # in for.
            return await asyncio.to_thread(path.read_text, encoding="utf-8")
        except OSError as exc:
            raise PermanentIngestionError(f"Could not read fixture {path}: {exc}") from exc

    def _extract_records(self, payload: str) -> Iterable[Mapping[str, Any]]:
        """Pull the listing records out of a recorded payload."""
        try:
            document = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise PermanentIngestionError(f"Fixture payload is not valid JSON: {exc}") from exc

        if not isinstance(document, dict):
            raise PermanentIngestionError(
                f"Fixture payload must be a JSON object, got {type(document).__name__}."
            )

        records = document.get("listings")
        if not isinstance(records, list):
            raise PermanentIngestionError(
                "Fixture payload has no 'listings' array; the recording is malformed."
            )

        return [record for record in records if isinstance(record, Mapping)]

    def _to_listing(self, record: Mapping[str, Any], query: SearchQuery) -> RawListing:
        """Map one recorded record onto the RawListing contract."""
        del query

        external_id = str(record.get("id") or "").strip()
        if not external_id:
            raise ListingParseError("Record has no 'id'; it has no stable identity.")

        try:
            amount, currency = parse_price(
                str(record.get("price") or ""), default_currency=self.default_currency
            )
            return RawListing(
                platform=self.platform,
                external_id=external_id,
                url=record["url"],
                title=str(record.get("title") or ""),
                description=record.get("description"),
                asking_price=amount,
                currency=currency,
                condition=parse_condition(record.get("condition")),
                brand_raw=record.get("brand"),
                size_raw=record.get("size"),
                seller_username=record.get("seller"),
                image_urls=list(record.get("images") or []),
                listed_at=record.get("listed_at"),
            )
        except MalformedListingError as exc:
            raise ListingParseError(f"Listing {external_id}: {exc}") from exc
        except (ValidationError, KeyError, TypeError, ValueError) as exc:
            # Naming the record matters: with the fixture on disk, the id is
            # enough to open the exact entry that failed.
            raise ListingParseError(f"Listing {external_id} failed validation: {exc}") from exc

    def _page_path(self, query: SearchQuery, page: int) -> Path | None:
        """Resolve which file backs a given query and page."""
        slug = (
            query.slug
            if (self.fixtures_dir / f"{query.slug}-p1.json").is_file()
            else DEFAULT_FIXTURE_SLUG
        )
        candidate = self.fixtures_dir / f"{slug}-p{page}.json"
        if candidate.is_file():
            logger.debug("Replaying fixture %s", candidate.name)
            return candidate
        return None

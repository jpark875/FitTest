"""Offline replay of recorded marketplace payloads.

This is the default ingestion source, not a testing convenience. Both target
platforms prohibit automated collection, so cloning this repository and running
it must not depend on ignoring that: a fresh clone runs end to end against
payloads recorded here, with no network and no credentials.

Because FixtureSource satisfies the same contract as the live adapters,
everything downstream is exercised by the real code path. The seam is at the
source, not at a mock three layers up.

Fixtures are JSON documents shaped like a marketplace search response, named
{query-slug}-p{page}.json. A query with no recording falls back to
default-p{page}.json. Hosts use the reserved .invalid TLD, so the data is
obviously synthetic and an accidental live fetch fails rather than hitting a
real CDN.
"""

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
    """Replays recorded payloads from disk as if they came from a marketplace.

    Implements the same three methods as a live adapter with the same error
    semantics: a malformed record is skipped and counted, an unreadable payload
    raises. A fixture source that degraded differently from production would be
    a worse predictor of the live path, which is the point of having one.
    """

    platform: ClassVar[Platform] = Platform.FIXTURE

    def __init__(
        self,
        fixtures_dir: Path,
        *,
        default_currency: str = "GBP",
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        """Point the source at a directory of recorded payloads.

        Args:
            fixtures_dir: Directory holding {slug}-p{page}.json files.
            default_currency: ISO 4217 code assumed for unmarked prices. The
                recordings are UK listings, hence GBP.
            rate_limiter: Accepted for symmetry with live sources and defaulted
                to a no-op, so the composition root constructs every source
                identically.
        """
        super().__init__(rate_limiter=rate_limiter)
        self.fixtures_dir = fixtures_dir
        self.default_currency = default_currency.upper()

    async def _fetch_page(self, query: SearchQuery, page: int) -> str:
        """Read one recorded page off disk.

        Args:
            query: The search whose slug selects a fixture file.
            page: One-based page number.

        Returns:
            The file's contents, or an empty payload when this page was never
            recorded, which the base class reads as the end of the results.

        Raises:
            PermanentIngestionError: If the first page is missing, or a file
                that exists cannot be read. A missing page four is a short
                fixture set; a missing page one means the directory is
                misconfigured, and reporting that as "no listings found" would
                hide a real fault behind a plausible number.
        """
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
        """Pull the listing records out of a recorded payload.

        Args:
            payload: The JSON document read from disk.

        Returns:
            One mapping per recorded listing. Entries that are not mappings are
            dropped rather than counted as skips: a bare string in the array is
            a corrupt recording, not a seller typing something strange.

        Raises:
            PermanentIngestionError: If the payload is not JSON, is not an
                object, or has no listings array. A corrupt fixture is a
                repository fault and must be loud, or the offline suite passes
                vacuously.
        """
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
        """Map one recorded record onto the RawListing contract.

        Args:
            record: A single entry from the listings array.
            query: The originating query. Unused here, since fixtures carry
                their own currency symbols.

        Returns:
            The validated listing.

        Raises:
            ListingParseError: If this record cannot be coerced into the
                contract, so the base class can count and attribute it.
        """
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
        """Resolve which file backs a given query and page.

        The query's slug wins; default is the fallback, so any search returns
        plausible data. Nothing else about the query is honoured: fixtures are
        a recording, not a search engine.

        The fallback is decided once, on page one, and held for the query.
        Resolving per page would let a one-page recording spill onto the
        default set's page two, so a Carhartt search would return someone
        else's results halfway through.

        Args:
            query: The search being served.
            page: One-based page number.

        Returns:
            The path to read, or None if that page was never recorded.
        """
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

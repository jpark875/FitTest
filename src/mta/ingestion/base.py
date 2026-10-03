"""The contract every listing source implements, and the loop they all share.

A template method: the base class owns pagination, per-record error isolation,
duplicate suppression and statistics, so the error policy cannot drift between
platforms. Subclasses supply three methods: how to fetch a page, how to find
records in it, and how to map one record to a contract.

Failures are classified so callers branch on type rather than string-matching a
message:

- ListingParseError: one record is unusable. Skip it, count it.
- RateLimitedError: the server said slow down. Back off.
- TransientIngestionError: network hiccup or 5xx. Retry, then stop early with
  partial results.
- PermanentIngestionError: selector broke, auth failed. Raise.

The asymmetry between the last two is deliberate. A run returning 80 of 100
listings is useful. A permanent failure must be loud, because a scraper that
silently returns zero after a site redesign looks exactly like a scraper that
found nothing to buy.
"""

from __future__ import annotations

import abc
import logging
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from types import TracebackType
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mta.ingestion.rate_limiter import NoOpRateLimiter, RateLimiter
from mta.models.listing import MalformedListingError, Platform, RawListing

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """Base class for every failure originating in the ingestion layer."""


class ListingParseError(IngestionError):
    """A single record could not be turned into a RawListing.

    Scoped to one listing: the template method counts it and continues.
    """


class TransientIngestionError(IngestionError):
    """A failure that may succeed on retry: timeout, connection reset, 5xx."""


class RateLimitedError(TransientIngestionError):
    """The server explicitly asked us to slow down (429, or Retry-After).

    Honouring it is the difference between a temporary throttle and a
    permanent block.

    Attributes:
        retry_after: Seconds the server asked us to wait, if it said.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        """Record the message and the server-supplied delay.

        Args:
            message: Human-readable description for logs.
            retry_after: Parsed Retry-After value in seconds, if present.
        """
        super().__init__(message)
        self.retry_after = retry_after


class PermanentIngestionError(IngestionError):
    """A failure retrying cannot fix: markup changed, blocked, bad credentials."""


class SearchQuery(BaseModel):
    """What to look for, expressed independently of any platform's URL scheme.

    A parameter object, so adding a filter later does not change the signature
    of every source. Each adapter translates this into its own query string.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    keywords: str = Field(min_length=1, max_length=200)
    category: str | None = None
    min_price: Decimal | None = Field(default=None, ge=Decimal(0))
    max_price: Decimal | None = Field(default=None, ge=Decimal(0))
    max_pages: int = Field(default=5, ge=1, le=100)

    @property
    def slug(self) -> str:
        """A filesystem- and log-safe identifier for this query."""
        safe = "".join(c if c.isalnum() else "-" for c in self.keywords.lower())
        return "-".join(filter(None, safe.split("-")))[:80]


@dataclass
class IngestionStats:
    """Per-run counters, surfaced at the end of every pass.

    Skips are counted rather than merely logged because the ratio is the health
    signal: three in five hundred is seller noise, four hundred means the
    markup changed and the numbers are lying to you.
    """

    pages_fetched: int = 0
    records_seen: int = 0
    listings_yielded: int = 0
    listings_skipped: int = 0
    duplicates_suppressed: int = 0
    transient_failures: int = 0
    seconds_throttled: float = 0.0
    skip_reasons: dict[str, int] = field(default_factory=dict)

    def record_skip(self, reason: str) -> None:
        """Increment the skip counter and tally the reason.

        Args:
            reason: Short category, typically the exception class name.
        """
        self.listings_skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1

    @property
    def skip_rate(self) -> float:
        """Fraction of seen records that could not be parsed."""
        if self.records_seen == 0:
            return 0.0
        return self.listings_skipped / self.records_seen

    def summary(self) -> str:
        """Render a one-line summary of the run."""
        return (
            f"{self.listings_yielded} listings from {self.pages_fetched} page(s); "
            f"skipped {self.listings_skipped} ({self.skip_rate:.1%}), "
            f"suppressed {self.duplicates_suppressed} duplicate(s), "
            f"{self.transient_failures} transient failure(s), "
            f"{self.seconds_throttled:.1f}s throttled"
        )


#: Skip rate above which the run is assumed broken rather than noisy.
SUSPICIOUS_SKIP_RATE = 0.20


class ListingSource(abc.ABC):
    """Abstract base for anything that can produce RawListing objects.

    Implementations exist per marketplace and, equally importantly, for
    replaying recorded payloads offline. Because the fixture source satisfies
    the same contract, everything downstream of ingestion can be run and tested
    without a network connection.
    """

    #: Which marketplace this source represents. Set by each subclass.
    platform: ClassVar[Platform]

    def __init__(self, rate_limiter: RateLimiter | None = None) -> None:
        """Initialise the source with an optional shared throttle.

        Args:
            rate_limiter: Throttle governing outbound requests. Defaults to a
                no-op, correct for sources that perform no network I/O. Live
                sources should share one limiter across every source hitting
                the same host: two scrapers each holding their own
                ten-per-minute budget is twenty per minute to the server.
        """
        self.rate_limiter: RateLimiter = rate_limiter or NoOpRateLimiter()
        self.stats = IngestionStats()
        self._seen_fingerprints: set[str] = set()

    @abc.abstractmethod
    async def _fetch_page(self, query: SearchQuery, page: int) -> str:
        """Retrieve one page of results as an unparsed payload.

        Implementations must acquire from the rate limiter before any network
        call, and must translate transport failures into this module's
        exceptions: an escaping httpx error defeats the error-policy design.

        Args:
            query: What to search for.
            page: One-based page number.

        Returns:
            The raw response body, unparsed.

        Raises:
            TransientIngestionError: Timeout, connection failure, or 5xx.
            RateLimitedError: The server returned 429 or asked us to wait.
            PermanentIngestionError: 4xx other than 429, or a blocked request.
        """

    @abc.abstractmethod
    def _extract_records(self, payload: str) -> Iterable[Mapping[str, Any]]:
        """Locate the individual listing records inside a page payload.

        Returns loose mappings rather than finished models on purpose: this
        method locates data, _to_listing validates it. Keeping them apart means
        a contract change does not touch selector code.

        Args:
            payload: The body returned by _fetch_page.

        Returns:
            One mapping per listing found. An empty iterable is a valid result.

        Raises:
            PermanentIngestionError: If the payload's structure is
                unrecognisable. That is a site redesign, not an empty result.
        """

    @abc.abstractmethod
    def _to_listing(self, record: Mapping[str, Any], query: SearchQuery) -> RawListing:
        """Map one extracted record onto the RawListing contract.

        Args:
            record: A single mapping from _extract_records.
            query: The originating query, for provenance and defaults.

        Returns:
            The validated listing.

        Raises:
            ListingParseError: If this record is unusable. Raise rather than
                returning None; the caller counts it and moves on.
        """

    def _has_next_page(self, payload: str, page: int, records_found: int) -> bool:  # noqa: ARG002
        """Decide whether to request another page.

        The default (stop when a page yields nothing) suits most marketplaces.
        Override where a platform exposes a total count or a "next" link.

        Args:
            payload: The current page's raw body.
            page: The one-based page number just processed.
            records_found: How many records were extracted from it.

        Returns:
            True if another page should be requested.
        """
        return records_found > 0

    async def aclose(self) -> None:  # noqa: B027 - a no-op default is correct here
        """Release any resources held by the source.

        Network-backed sources override this to close their HTTP client.
        """

    async def __aenter__(self) -> Self:
        """Enter the async context manager, returning the source itself."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the source on exit, whether or not an error occurred."""
        await self.aclose()

    async def fetch_listings(self, query: SearchQuery) -> AsyncIterator[RawListing]:
        """Yield every listing matching the query, across pages.

        Yielding rather than returning a list is deliberate: enrichment is far
        more expensive than scraping, so downstream stages can start on the
        first listing while page three is in flight, and a caller can stop
        early without paying to materialise results it will not use.

        Args:
            query: What to search for, including a page ceiling.

        Yields:
            Validated listings, deduplicated by fingerprint within this run.

        Raises:
            PermanentIngestionError: On a failure retrying cannot fix. Results
                already yielded remain valid.
        """
        for page in range(1, query.max_pages + 1):
            try:
                payload = await self._fetch_page(query, page)
            except RateLimitedError as exc:
                # The server has told us its actual limit. Believe it over ours.
                self.stats.transient_failures += 1
                if exc.retry_after:
                    self.rate_limiter.penalize(exc.retry_after)
                logger.warning(
                    "%s rate-limited on page %d; stopping this query early. %s",
                    self.platform.value,
                    page,
                    exc,
                )
                break
            except TransientIngestionError as exc:
                # Already retried by the HTTP layer. Keep what we have.
                self.stats.transient_failures += 1
                logger.warning(
                    "%s transient failure on page %d after retries; returning partial results. %s",
                    self.platform.value,
                    page,
                    exc,
                )
                break

            self.stats.pages_fetched += 1

            try:
                records = list(self._extract_records(payload))
            except PermanentIngestionError:
                raise
            except Exception as exc:  # noqa: BLE001 - unknown parser failures are structural
                # Selector code throwing means the markup is not what the
                # parser was written against. Promote it rather than letting
                # the run report a cheerful zero listings.
                raise PermanentIngestionError(
                    f"{self.platform.value} page {page} could not be parsed; "
                    f"the site markup has probably changed."
                ) from exc

            if not records and page == 1:
                # Nothing on page one is suspicious; nothing on page four is
                # just the end of the results.
                logger.warning(
                    "%s returned no records on the first page for %r. Either the "
                    "query genuinely matches nothing, or the extractor is broken.",
                    self.platform.value,
                    query.keywords,
                )

            for record in records:
                self.stats.records_seen += 1
                try:
                    listing = self._to_listing(record, query)
                except (
                    ListingParseError,
                    MalformedListingError,
                    ValidationError,
                    KeyError,
                    ValueError,
                    TypeError,
                ) as exc:
                    # Sellers type whatever they like into free-text fields,
                    # and a marketplace page is not a schema.
                    self.stats.record_skip(type(exc).__name__)
                    logger.debug(
                        "Skipped a %s record on page %d: %s", self.platform.value, page, exc
                    )
                    continue

                if listing.fingerprint in self._seen_fingerprints:
                    # Marketplaces re-order results between requests, so one
                    # item legitimately appears on two pages of a crawl.
                    self.stats.duplicates_suppressed += 1
                    continue

                self._seen_fingerprints.add(listing.fingerprint)
                self.stats.listings_yielded += 1
                yield listing

            if not self._has_next_page(payload, page, len(records)):
                break

        if self.stats.skip_rate > SUSPICIOUS_SKIP_RATE and self.stats.records_seen > 10:
            logger.error(
                "%s skipped %.0f%% of records (%s), above the %.0f%% threshold. "
                "This usually means a parser has drifted from the markup.",
                self.platform.value,
                self.stats.skip_rate * 100,
                self.stats.skip_reasons,
                SUSPICIOUS_SKIP_RATE * 100,
            )

        logger.info("%s ingestion complete: %s", self.platform.value, self.stats.summary())

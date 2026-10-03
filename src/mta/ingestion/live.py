"""Live marketplace adapters built on schema.org structured data.

Listing pages that publish JSON-LD (`Product` or `ItemList`) are parsed from that rather than
from CSS selectors, which break on every redesign. The URL patterns below are defaults and
have not been verified against the live sites; both platforms prohibit automated
collection, so only point these at services you are authorised to collect from.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup
from pydantic import ValidationError

from mta.ingestion.base import (
    ListingParseError,
    ListingSource,
    PermanentIngestionError,
    SearchQuery,
)
from mta.ingestion.http import ThrottledHttpClient
from mta.models.listing import (
    MalformedListingError,
    Platform,
    RawListing,
    parse_condition,
    parse_price,
)

logger = logging.getLogger(__name__)

_EMPTY_PAGE = '<html><script type="application/ld+json">{"@type": "ItemList"}</script></html>'

_SCHEMA_CONDITIONS = {
    "newcondition": "new",
    "refurbishedcondition": "used - like new",
    "usedcondition": "used",
    "damagedcondition": "poor",
}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _walk(node: Any) -> Iterable[Mapping[str, Any]]:
    """Yield every Product inside a JSON-LD document, through @graph and ItemList wrappers."""
    for item in _as_list(node):
        if not isinstance(item, Mapping):
            continue
        kinds = {str(k).rsplit("/", 1)[-1] for k in _as_list(item.get("@type"))}
        if "Product" in kinds:
            yield item
        yield from _walk(item.get("@graph"))
        for element in _as_list(item.get("itemListElement")):
            inner = element.get("item", element) if isinstance(element, Mapping) else None
            yield from _walk(inner)


def _name(value: Any) -> str | None:
    if isinstance(value, Mapping):
        value = value.get("name")
    return str(value).strip() if value else None


def _image_urls(value: Any) -> list[str]:
    urls = []
    for image in _as_list(value):
        url = image.get("url") if isinstance(image, Mapping) else image
        if isinstance(url, str) and url:
            urls.append(url)
    return urls


class StructuredDataSource(ListingSource):
    """Shared behaviour for adapters that read JSON-LD product data."""

    base_url: ClassVar[str]
    search_path: ClassVar[str]
    query_param: ClassVar[str] = "q"
    page_param: ClassVar[str] = "page"

    def __init__(
        self,
        client: ThrottledHttpClient,
        *,
        default_currency: str = "GBP",
        respect_robots: bool = True,
    ) -> None:
        """Share the client's throttle so every request, robots.txt included, is paced."""
        super().__init__(rate_limiter=client.rate_limiter)
        self.client = client
        self.default_currency = default_currency.upper()
        self.respect_robots = respect_robots
        self._robots: RobotFileParser | None = None

    async def aclose(self) -> None:
        """Close the shared HTTP client."""
        await self.client.aclose()

    async def _check_robots(self, url: str) -> None:
        if not self.respect_robots:
            return
        if self._robots is None:
            parser = RobotFileParser()
            try:
                body = await self.client.get_text(urljoin(self.base_url, "/robots.txt"))
                parser.parse(body.splitlines())
            except PermanentIngestionError:
                # No readable robots.txt means nothing is disallowed.
                parser.parse([])
            self._robots = parser
        if not self._robots.can_fetch(self.client.user_agent, url):
            raise PermanentIngestionError(f"robots.txt at {self.base_url} disallows {url}")

    async def _fetch_page(self, query: SearchQuery, page: int) -> str:
        url = urljoin(self.base_url, self.search_path)
        await self._check_robots(url)
        params: dict[str, Any] = {self.query_param: query.keywords, self.page_param: page}
        try:
            return await self.client.get_text(
                url, params=params, archive_key=f"{self.platform.value}-{query.slug}-p{page}"
            )
        except PermanentIngestionError as exc:
            # Requesting a page past the end is a 404 on most sites.
            if page > 1 and str(exc).startswith("404"):
                return _EMPTY_PAGE
            raise

    def _extract_records(self, payload: str) -> Iterable[Mapping[str, Any]]:
        soup = BeautifulSoup(payload, "lxml")
        scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
        if not scripts:
            raise PermanentIngestionError(
                f"{self.platform.value} page has no JSON-LD; the page structure has changed."
            )
        records: list[Mapping[str, Any]] = []
        for script in scripts:
            try:
                document = json.loads(script.string or "")
            except json.JSONDecodeError:
                logger.debug("Ignoring unparsable JSON-LD block")
                continue
            records.extend(_walk(document))
        return records

    def _to_listing(self, record: Mapping[str, Any], query: SearchQuery) -> RawListing:
        del query
        offer = next((o for o in _as_list(record.get("offers")) if isinstance(o, Mapping)), {})
        url = record.get("url") or offer.get("url")
        external_id = str(record.get("sku") or record.get("productID") or "").strip()
        if not external_id and url:
            external_id = str(url).rstrip("/").rsplit("/", 1)[-1]
        if not external_id:
            raise ListingParseError("Product has no sku, productID or url.")

        price = offer.get("price")
        if price is None:
            raise ListingParseError(f"Listing {external_id}: no price in offer.")
        currency = offer.get("priceCurrency")
        condition = str(offer.get("itemCondition") or "").rsplit("/", 1)[-1].lower()

        try:
            amount, code = parse_price(
                f"{price} {currency}" if currency else str(price),
                default_currency=self.default_currency,
            )
            return RawListing(
                platform=self.platform,
                external_id=external_id,
                url=urljoin(self.base_url, str(url)) if url else f"{self.base_url}/{external_id}",
                title=str(record.get("name") or ""),
                description=record.get("description"),
                asking_price=amount,
                currency=code,
                condition=parse_condition(_SCHEMA_CONDITIONS.get(condition, condition)),
                brand_raw=_name(record.get("brand")),
                size_raw=_name(record.get("size")),
                seller_username=_name(offer.get("seller")),
                image_urls=_image_urls(record.get("image")),
                listed_at=record.get("datePublished"),
            )
        except MalformedListingError as exc:
            raise ListingParseError(f"Listing {external_id}: {exc}") from exc
        except (ValidationError, TypeError, ValueError) as exc:
            raise ListingParseError(f"Listing {external_id} failed validation: {exc}") from exc


class DepopSource(StructuredDataSource):
    """Depop search results."""

    platform: ClassVar[Platform] = Platform.DEPOP
    base_url = "https://www.depop.com"
    search_path = "/search/"


class ThredUpSource(StructuredDataSource):
    """ThredUp search results."""

    platform: ClassVar[Platform] = Platform.THREDUP
    base_url = "https://www.thredup.com"
    search_path = "/products"
    query_param = "search_text"

    def __init__(self, client: ThrottledHttpClient, **kwargs: Any) -> None:
        """Default to USD, which ThredUp prices in."""
        kwargs.setdefault("default_currency", "USD")
        super().__init__(client, **kwargs)

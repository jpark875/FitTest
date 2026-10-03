"""Tests for the structured-data marketplace adapters."""

from __future__ import annotations

import json
from typing import Any

import pytest
import respx

from mta.ingestion.base import PermanentIngestionError, SearchQuery
from mta.ingestion.http import ThrottledHttpClient
from mta.ingestion.live import DepopSource, ThredUpSource
from mta.models.listing import ItemCondition, Platform

QUERY = SearchQuery(keywords="carhartt jacket", max_pages=3)
SEARCH = "https://www.depop.com/search/"


def product(sku: str, **overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "@type": "Product",
        "sku": sku,
        "name": f"Carhartt jacket {sku}",
        "url": f"/products/{sku}/",
        "description": "Blanket lined.",
        "brand": {"@type": "Brand", "name": "Carhartt"},
        "size": "L",
        "image": ["https://img.invalid/1.jpg", {"url": "https://img.invalid/2.jpg"}],
        "datePublished": "2026-07-01T12:00:00Z",
        "offers": {
            "@type": "Offer",
            "price": "48.00",
            "priceCurrency": "GBP",
            "itemCondition": "https://schema.org/UsedCondition",
            "seller": {"name": "thrifter"},
        },
    }
    item.update(overrides)
    return item


def page(*items: dict[str, Any], wrapper: str = "list") -> str:
    if wrapper == "list":
        document: Any = {
            "@type": "ItemList",
            "itemListElement": [{"@type": "ListItem", "item": item} for item in items],
        }
    else:
        document = {"@graph": list(items)}
    return f'<html><script type="application/ld+json">{json.dumps(document)}</script></html>'


def make_source(cls: type = DepopSource, **kwargs: Any) -> Any:
    client = ThrottledHttpClient(user_agent="mta-test/1.0", max_retries=0)
    return cls(client, **kwargs)


async def collect(source: Any, query: SearchQuery = QUERY) -> list[Any]:
    async with source:
        return [item async for item in source.fetch_listings(query)]


@pytest.fixture
def mock() -> Any:
    with respx.mock(assert_all_called=False) as router:
        router.get("https://www.depop.com/robots.txt").respond(404)
        yield router


class TestParsing:
    async def test_maps_a_product_onto_the_contract(self, mock: Any) -> None:
        mock.get(SEARCH, params={"q": "carhartt jacket", "page": 1}).respond(
            text=page(product("a1"))
        )
        mock.get(SEARCH, params={"page": 2}).respond(text=page())

        [listing] = await collect(make_source())

        assert listing.platform is Platform.DEPOP
        assert listing.external_id == "a1"
        assert str(listing.asking_price) == "48.00"
        assert listing.currency == "GBP"
        assert listing.condition is ItemCondition.FAIR
        assert listing.brand_raw == "Carhartt"
        assert listing.seller_username == "thrifter"
        assert [str(u) for u in listing.image_urls] == [
            "https://img.invalid/1.jpg",
            "https://img.invalid/2.jpg",
        ]
        assert str(listing.url) == "https://www.depop.com/products/a1/"

    async def test_reads_products_inside_a_graph(self, mock: Any) -> None:
        mock.get(SEARCH, params={"page": 1}).respond(text=page(product("g1"), wrapper="graph"))
        mock.get(SEARCH, params={"page": 2}).respond(text=page())
        assert [i.external_id for i in await collect(make_source())] == ["g1"]

    async def test_id_falls_back_to_the_url_and_bad_records_are_skipped(self, mock: Any) -> None:
        no_sku = {**product("x"), "sku": None, "url": "/products/from-url/"}
        no_price = product("p1", offers={"priceCurrency": "GBP"})
        mock.get(SEARCH, params={"page": 1}).respond(text=page(no_sku, no_price, product("ok")))
        mock.get(SEARCH, params={"page": 2}).respond(text=page())
        source = make_source()

        listings = await collect(source)

        assert [i.external_id for i in listings] == ["from-url", "ok"]
        assert source.stats.listings_skipped == 1

    async def test_thredup_defaults_to_dollars(self, mock: Any) -> None:
        mock.get("https://www.thredup.com/robots.txt").respond(404)
        priced = product("t1", offers={"price": "30"})
        mock.get("https://www.thredup.com/products", params={"page": 1}).respond(text=page(priced))
        mock.get("https://www.thredup.com/products", params={"page": 2}).respond(text=page())

        [listing] = await collect(make_source(ThredUpSource))

        assert listing.currency == "USD"
        assert listing.platform is Platform.THREDUP


class TestFailureModes:
    async def test_a_page_without_structured_data_is_a_structural_failure(self, mock: Any) -> None:
        mock.get(SEARCH, params={"page": 1}).respond(text="<html><body>redesigned</body></html>")
        with pytest.raises(PermanentIngestionError, match="no JSON-LD"):
            await collect(make_source())

    async def test_an_empty_result_list_is_not_an_error(self, mock: Any) -> None:
        mock.get(SEARCH, params={"page": 1}).respond(text=page())
        assert await collect(make_source()) == []

    async def test_a_404_past_the_last_page_ends_pagination(self, mock: Any) -> None:
        mock.get(SEARCH, params={"page": 1}).respond(text=page(product("a1")))
        mock.get(SEARCH, params={"page": 2}).respond(404)
        assert len(await collect(make_source())) == 1

    async def test_a_404_on_the_first_page_is_permanent(self, mock: Any) -> None:
        mock.get(SEARCH, params={"page": 1}).respond(404)
        with pytest.raises(PermanentIngestionError):
            await collect(make_source())

    async def test_rate_limiting_returns_partial_results(self, mock: Any) -> None:
        mock.get(SEARCH, params={"page": 1}).respond(text=page(product("a1")))
        mock.get(SEARCH, params={"page": 2}).respond(429, headers={"Retry-After": "1"})
        assert len(await collect(make_source())) == 1

    async def test_malformed_json_ld_blocks_are_ignored(self, mock: Any) -> None:
        body = page(product("a1")).replace(
            "</html>", '<script type="application/ld+json">{oops</script></html>'
        )
        mock.get(SEARCH, params={"page": 1}).respond(text=body)
        mock.get(SEARCH, params={"page": 2}).respond(text=page())
        assert len(await collect(make_source())) == 1


class TestRobots:
    async def test_disallowed_path_is_refused(self, mock: Any) -> None:
        mock.get("https://www.depop.com/robots.txt").respond(
            text="User-agent: *\nDisallow: /search/\n"
        )
        search = mock.get(SEARCH).respond(text=page(product("a1")))

        with pytest.raises(PermanentIngestionError, match="robots.txt"):
            await collect(make_source())
        assert not search.called

    async def test_can_be_switched_off_for_sources_you_own(self, mock: Any) -> None:
        mock.get("https://www.depop.com/robots.txt").respond(
            text="User-agent: *\nDisallow: /search/\n"
        )
        mock.get(SEARCH, params={"page": 1}).respond(text=page(product("a1")))
        mock.get(SEARCH, params={"page": 2}).respond(text=page())

        assert len(await collect(make_source(respect_robots=False))) == 1

    async def test_robots_is_fetched_once(self, mock: Any) -> None:
        robots = mock.get("https://www.depop.com/robots.txt").respond(404)
        mock.get(SEARCH, params={"page": 1}).respond(text=page(product("a1")))
        mock.get(SEARCH, params={"page": 2}).respond(text=page(product("a2")))
        mock.get(SEARCH, params={"page": 3}).respond(text=page())
        await collect(make_source())
        assert robots.call_count == 1

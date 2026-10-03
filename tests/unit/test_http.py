"""Tests for the shared HTTP transport.

respx intercepts at the httpx transport layer, so the client under test is the
real one (real retry policy, real status translation, real throttling) with only
the socket replaced.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest
import respx

from mta.ingestion.base import (
    PermanentIngestionError,
    RateLimitedError,
    TransientIngestionError,
)
from mta.ingestion.http import (
    MAX_HONOURED_RETRY_AFTER_SECONDS,
    MAX_RESPONSE_BYTES,
    ThrottledHttpClient,
    parse_retry_after,
)
from mta.ingestion.rate_limiter import RateLimiter

URL = "https://marketplace.invalid/search"


def make_client(**kwargs: object) -> ThrottledHttpClient:
    defaults: dict[str, object] = {
        "user_agent": "mta-test/1.0",
        "max_retries": 1,
        # Real backoff policy, compressed. Tests assert on attempt counts,
        # never on the gaps between them.
        "retry_backoff_seconds": 0.001,
    }
    return ThrottledHttpClient(**{**defaults, **kwargs})  # type: ignore[arg-type]


class RecordingLimiter:
    """A limiter that records calls instead of sleeping."""

    def __init__(self) -> None:
        self.acquisitions = 0
        self.penalties: list[float] = []

    async def acquire(self, tokens: float = 1.0) -> float:
        self.acquisitions += 1
        return 0.0

    def penalize(self, seconds: float) -> None:
        self.penalties.append(seconds)


class TestParseRetryAfter:
    def test_parses_the_delay_seconds_form(self) -> None:
        assert parse_retry_after("120") == 120.0

    def test_parses_the_http_date_form(self) -> None:
        parsed = parse_retry_after(format_datetime(datetime.now(UTC) + timedelta(seconds=60)))
        assert parsed is not None
        assert 50 <= parsed <= 65

    def test_clamps_an_absurd_delay(self) -> None:
        assert parse_retry_after("999999") == MAX_HONOURED_RETRY_AFTER_SECONDS

    @pytest.mark.parametrize("value", [None, "", "soon", "0", "-30"])
    def test_unusable_values_yield_none(self, value: str | None) -> None:
        assert parse_retry_after(value) is None

    def test_a_past_date_yields_none(self) -> None:
        past = format_datetime(datetime.now(UTC) - timedelta(seconds=120))
        assert parse_retry_after(past) is None


class TestSuccessPath:
    @respx.mock
    async def test_returns_the_body(self) -> None:
        respx.get(URL).mock(return_value=httpx.Response(200, text="<html>ok</html>"))
        async with make_client() as client:
            assert await client.get_text(URL) == "<html>ok</html>"

    @respx.mock
    async def test_sends_the_user_agent(self) -> None:
        route = respx.get(URL).mock(return_value=httpx.Response(200, text="ok"))
        async with make_client(user_agent="mta/0.1 (contact: me@example.invalid)") as client:
            await client.get_text(URL)
        assert route.calls.last.request.headers["User-Agent"].startswith("mta/0.1")

    @respx.mock
    async def test_passes_query_parameters_through(self) -> None:
        route = respx.get(URL).mock(return_value=httpx.Response(200, text="ok"))
        async with make_client() as client:
            await client.get_text(URL, params={"q": "carhartt", "page": 2})
        assert route.calls.last.request.url.params["q"] == "carhartt"


class TestThrottling:
    @respx.mock
    async def test_every_attempt_spends_budget_including_retries(self) -> None:
        respx.get(URL).mock(
            side_effect=[httpx.Response(500), httpx.Response(500), httpx.Response(200, text="ok")]
        )
        limiter = RecordingLimiter()
        async with make_client(max_retries=2, rate_limiter=limiter) as client:
            await client.get_text(URL)
        assert limiter.acquisitions == 3

    def test_the_recording_limiter_satisfies_the_protocol(self) -> None:
        assert isinstance(RecordingLimiter(), RateLimiter)


class TestRetryPolicy:
    @respx.mock
    async def test_retries_a_5xx_and_succeeds(self) -> None:
        route = respx.get(URL).mock(
            side_effect=[httpx.Response(503), httpx.Response(200, text="recovered")]
        )
        async with make_client() as client:
            assert await client.get_text(URL) == "recovered"
        assert route.call_count == 2

    @respx.mock
    async def test_gives_up_after_the_configured_attempts(self) -> None:
        route = respx.get(URL).mock(return_value=httpx.Response(502))
        async with make_client(max_retries=1) as client:
            with pytest.raises(TransientIngestionError):
                await client.get_text(URL)
        assert route.call_count == 2

    @respx.mock
    async def test_retries_a_timeout(self) -> None:
        route = respx.get(URL).mock(
            side_effect=[httpx.TimeoutException("slow"), httpx.Response(200, text="ok")]
        )
        async with make_client() as client:
            assert await client.get_text(URL) == "ok"
        assert route.call_count == 2

    @respx.mock
    async def test_retries_a_transport_failure(self) -> None:
        route = respx.get(URL).mock(
            side_effect=[httpx.ConnectError("dns"), httpx.Response(200, text="ok")]
        )
        async with make_client() as client:
            assert await client.get_text(URL) == "ok"
        assert route.call_count == 2

    @respx.mock
    async def test_never_retries_a_permanent_failure(self) -> None:
        route = respx.get(URL).mock(return_value=httpx.Response(403))
        async with make_client(max_retries=3) as client:
            with pytest.raises(PermanentIngestionError):
                await client.get_text(URL)
        assert route.call_count == 1

    @respx.mock
    async def test_max_retries_zero_disables_retrying(self) -> None:
        route = respx.get(URL).mock(return_value=httpx.Response(500))
        async with make_client(max_retries=0) as client:
            with pytest.raises(TransientIngestionError):
                await client.get_text(URL)
        assert route.call_count == 1


class TestRateLimitHandling:
    @respx.mock
    async def test_429_carries_the_servers_own_delay(self) -> None:
        respx.get(URL).mock(return_value=httpx.Response(429, headers={"Retry-After": "90"}))
        async with make_client() as client:
            with pytest.raises(RateLimitedError) as caught:
                await client.get_text(URL)
        assert caught.value.retry_after == 90.0

    @respx.mock
    async def test_a_stated_wait_is_not_retried_on_our_own_backoff(self) -> None:
        # RateLimitedError subclasses TransientIngestionError, so a naive
        # predicate would retry it three times against a server asking for two
        # minutes.
        route = respx.get(URL).mock(
            return_value=httpx.Response(429, headers={"Retry-After": "120"})
        )
        async with make_client(max_retries=3) as client:
            with pytest.raises(RateLimitedError):
                await client.get_text(URL)
        assert route.call_count == 1

    @respx.mock
    async def test_429_without_a_header_still_signals_rate_limiting(self) -> None:
        respx.get(URL).mock(return_value=httpx.Response(429))
        async with make_client() as client:
            with pytest.raises(RateLimitedError) as caught:
                await client.get_text(URL)
        assert caught.value.retry_after is None

    @respx.mock
    async def test_503_with_a_retry_after_is_a_slow_down(self) -> None:
        respx.get(URL).mock(return_value=httpx.Response(503, headers={"Retry-After": "30"}))
        async with make_client() as client:
            with pytest.raises(RateLimitedError) as caught:
                await client.get_text(URL)
        assert caught.value.retry_after == 30.0

    @respx.mock
    async def test_503_without_a_header_is_transient(self) -> None:
        route = respx.get(URL).mock(
            side_effect=[httpx.Response(503), httpx.Response(200, text="ok")]
        )
        async with make_client() as client:
            assert await client.get_text(URL) == "ok"
        assert route.call_count == 2


class TestStatusTranslation:
    @respx.mock
    @pytest.mark.parametrize("status", [401, 403])
    async def test_refusals_are_permanent(self, status: int) -> None:
        respx.get(URL).mock(return_value=httpx.Response(status))
        async with make_client() as client:
            with pytest.raises(PermanentIngestionError, match="bot detection"):
                await client.get_text(URL)

    @respx.mock
    async def test_404_is_permanent(self) -> None:
        respx.get(URL).mock(return_value=httpx.Response(404))
        async with make_client() as client:
            with pytest.raises(PermanentIngestionError):
                await client.get_text(URL)

    @respx.mock
    async def test_a_redirect_loop_is_permanent(self) -> None:
        respx.get(URL).mock(side_effect=httpx.TooManyRedirects("loop"))
        async with make_client() as client:
            with pytest.raises(PermanentIngestionError, match="redirects"):
                await client.get_text(URL)

    @respx.mock
    async def test_an_oversized_body_is_refused(self) -> None:
        respx.get(URL).mock(return_value=httpx.Response(200, text="x" * (MAX_RESPONSE_BYTES + 1)))
        async with make_client() as client:
            with pytest.raises(PermanentIngestionError, match="ceiling"):
                await client.get_text(URL)


class TestArchiving:
    @respx.mock
    async def test_writes_the_payload_before_parsing(self, tmp_path: Path) -> None:
        respx.get(URL).mock(return_value=httpx.Response(200, text="<html>captured</html>"))
        async with make_client(archive_dir=tmp_path) as client:
            await client.get_text(URL, archive_key="fixture-carhartt-p1")

        written = list(tmp_path.glob("fixture-carhartt-p1-*.html"))
        assert len(written) == 1
        assert written[0].read_text(encoding="utf-8") == "<html>captured</html>"

    @respx.mock
    async def test_no_key_means_no_archive(self, tmp_path: Path) -> None:
        respx.get(URL).mock(return_value=httpx.Response(200, text="ok"))
        async with make_client(archive_dir=tmp_path) as client:
            await client.get_text(URL)
        assert list(tmp_path.iterdir()) == []

    @respx.mock
    async def test_an_unwritable_archive_does_not_fail_the_scrape(self, tmp_path: Path) -> None:
        blocked = tmp_path / "not-a-directory"
        blocked.write_text("I am a file", encoding="utf-8")
        respx.get(URL).mock(return_value=httpx.Response(200, text="ok"))
        async with make_client(archive_dir=blocked / "nested") as client:
            assert await client.get_text(URL, archive_key="k") == "ok"


class TestLifecycle:
    @respx.mock
    async def test_the_context_manager_closes_the_pool(self) -> None:
        respx.get(URL).mock(return_value=httpx.Response(200, text="ok"))
        client = make_client()
        async with client:
            await client.get_text(URL)
        assert client._client.is_closed

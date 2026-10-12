"""The single network boundary of the ingestion layer."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from mta.ingestion.base import (
    PermanentIngestionError,
    RateLimitedError,
    TransientIngestionError,
)
from mta.ingestion.rate_limiter import NoOpRateLimiter, RateLimiter

logger = logging.getLogger(__name__)

#: A marketplace search page is tens of kilobytes. Ten megabytes means a
#: misconfigured endpoint or something adversarial.
MAX_RESPONSE_BYTES = 10 * 1024 * 1024

#: Ceiling on a server-supplied Retry-After. Blocking a pipeline for two hours
#: on a header is worse than retrying on the next schedule.
MAX_HONOURED_RETRY_AFTER_SECONDS = 300.0

_TRANSIENT_STATUS = frozenset({408, 425, 500, 502, 503, 504})

#: Codes meaning "slow down", as opposed to "you are broken".
_RATE_LIMIT_STATUS = frozenset({429, 503})


def parse_retry_after(value: str | None) -> float | None:
    """Interpret a Retry-After header in either of its legal forms."""
    if not value:
        return None

    text = value.strip()
    try:
        seconds = float(text)
    except ValueError:
        try:
            when = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            logger.debug("Could not parse Retry-After header %r; ignoring.", value)
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        seconds = (when - datetime.now(UTC)).total_seconds()

    if seconds <= 0:
        return None
    return min(seconds, MAX_HONOURED_RETRY_AFTER_SECONDS)


class ThrottledHttpClient:
    """A rate-limited, retrying HTTP client speaking the ingestion exceptions."""

    def __init__(
        self,
        *,
        user_agent: str,
        timeout_seconds: float = 20.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
        rate_limiter: RateLimiter | None = None,
        archive_dir: Path | None = None,
    ) -> None:
        """Configure the client."""
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.rate_limiter: RateLimiter = rate_limiter or NoOpRateLimiter()
        self.archive_dir = archive_dir
        self.seconds_throttled = 0.0

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=True,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-GB,en;q=0.9",
            },
        )

    async def get_text(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        archive_key: str | None = None,
    ) -> str:
        """Fetch a URL, retrying transient failures, and return the body."""
        attempts = self.max_retries + 1
        async for attempt in AsyncRetrying(
            # RateLimitedError subclasses TransientIngestionError, so without
            # this exclusion a 429 asking for 120 seconds would be retried on
            # our own one-second backoff.
            retry=(
                retry_if_exception_type(TransientIngestionError)
                & retry_if_not_exception_type(RateLimitedError)
            ),
            stop=stop_after_attempt(attempts),
            wait=wait_exponential_jitter(initial=self.retry_backoff_seconds, max=30.0),
            reraise=True,
        ):
            with attempt:
                number = attempt.retry_state.attempt_number
                if number > 1:
                    logger.info("Retry %d/%d for %s", number - 1, attempts - 1, url)
                return await self._attempt_get(url, params=params, archive_key=archive_key)

        # Unreachable: reraise=True guarantees the last failure propagates.
        raise TransientIngestionError(f"Exhausted all attempts for {url}.")

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        """Enter the async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the client on exit, error or not."""
        await self.aclose()

    async def _attempt_get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None,
        archive_key: str | None,
    ) -> str:
        """Perform one throttled request and translate its outcome."""
        self.seconds_throttled += await self.rate_limiter.acquire()

        try:
            response = await self._client.get(url, params=params)
        except httpx.TimeoutException as exc:
            raise TransientIngestionError(
                f"Timed out after {self.timeout_seconds}s: {url}"
            ) from exc
        except httpx.TooManyRedirects as exc:
            raise PermanentIngestionError(f"Too many redirects for {url}.") from exc
        except httpx.TransportError as exc:
            raise TransientIngestionError(f"Transport failure for {url}: {exc}") from exc

        self._raise_for_status(response, url)

        content_length = len(response.content)
        if content_length > MAX_RESPONSE_BYTES:
            raise PermanentIngestionError(
                f"Response from {url} was {content_length} bytes, above the "
                f"{MAX_RESPONSE_BYTES} byte ceiling. Refusing to parse it."
            )

        body = response.text
        if archive_key:
            self._archive(archive_key, body)
        return body

    def _raise_for_status(self, response: httpx.Response, url: str) -> None:
        """Map an HTTP status code onto the ingestion exceptions."""
        status = response.status_code
        if status < 400:
            return

        retry_after = parse_retry_after(response.headers.get("Retry-After"))

        if status in _RATE_LIMIT_STATUS and (status == 429 or retry_after is not None):
            raise RateLimitedError(
                f"{status} from {url}; server asked for "
                f"{retry_after if retry_after else 'an unspecified'} second wait.",
                retry_after=retry_after,
            )

        if status in _TRANSIENT_STATUS:
            raise TransientIngestionError(f"{status} from {url}.")

        if status in (401, 403):
            # Usually detection rather than a credential problem, and retrying
            # it makes the situation worse.
            raise PermanentIngestionError(
                f"{status} from {url}. The request was refused, likely bot detection. "
                f"Do not retry; reduce the request rate or switch to the fixture source."
            )

        raise PermanentIngestionError(f"{status} from {url}.")

    def _archive(self, key: str, body: str) -> None:
        """Persist a raw payload for later debugging."""
        if self.archive_dir is None:
            return

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = self.archive_dir / f"{key}-{stamp}.html"
        try:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not archive payload to %s: %s", path, exc)
        else:
            logger.debug("Archived %d bytes to %s", len(body), path)

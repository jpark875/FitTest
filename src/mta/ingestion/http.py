"""The single network boundary of the ingestion layer.

Every outbound request goes through ThrottledHttpClient, which buys four things
that are painful to retrofit once a second scraper exists:

- One translation point. httpx exceptions and status codes become this
  package's exceptions in exactly one place, so no scraper can invent its own
  retry policy.
- One throttle. The limiter is applied per attempt, including retries.
- One archive. Raw payloads are written to data/raw/ before parsing, so a
  selector that breaks later can be debugged against the bytes that broke it.
- One place to be polite. User agent, timeouts and redirect handling are set
  once and cannot drift per platform.

Retries use exponential backoff with jitter. The jitter matters: without it,
coroutines that fail together retry together against a server that is already
struggling. Only TransientIngestionError is retried.

RateLimitedError is the deliberate exception to that. It is transient, but the
server has stated a wait, and answering a stated wait with our own shorter
guess is worse than not retrying at all. It propagates on the first response so
the caller can penalise the shared limiter by the server's figure.
"""

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
    """Interpret a Retry-After header in either of its legal forms.

    RFC 9110 permits a delay in seconds and an absolute HTTP date. Real servers
    use both, and handling only the integer form ignores the instruction from
    the ones that use dates, typically the CDNs in front of exactly the sites
    worth being careful with.

    Args:
        value: The raw header value, or None if absent.

    Returns:
        Seconds to wait, clamped to MAX_HONOURED_RETRY_AFTER_SECONDS, or None
        if the header is absent or unparseable.
    """
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
    """A rate-limited, retrying HTTP client speaking the ingestion exceptions.

    Takes primitives rather than a settings object: the contract is "a timeout,
    a retry count, and something to throttle against".

    One instance should be shared across every source hitting the same host, so
    the connection pool and the rate limiter are genuinely shared.
    """

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
        """Configure the client.

        Args:
            user_agent: Sent on every request. A contact address in the string
                costs nothing and occasionally turns a block into an email.
            timeout_seconds: Per-attempt ceiling covering connect, read and
                write, so worst-case wall time is roughly
                timeout x (max_retries + 1) plus backoff.
            max_retries: Additional attempts after the first. 0 disables them.
            retry_backoff_seconds: Base delay before the first retry, doubling
                with jitter thereafter. A parameter, not a literal, so it can
                be tuned per platform and tightened in tests.
            rate_limiter: Shared throttle. Defaults to no-op, which is only
                appropriate for tests and offline sources.
            archive_dir: If given, raw payloads are written here before parsing.
        """
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
        """Fetch a URL, retrying transient failures, and return the body.

        Args:
            url: Absolute URL to fetch.
            params: Query parameters.
            archive_key: Filename stem for the raw payload archive. Nothing is
                written when this or archive_dir is None.

        Returns:
            The response body decoded as text.

        Raises:
            RateLimitedError: On 429, or 503 with a Retry-After.
            TransientIngestionError: All attempts failed with a retryable
                condition.
            PermanentIngestionError: 403, 404, an oversized response, or too
                many redirects.
        """
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
        """Close the underlying connection pool.

        A leaked pool surfaces as an unclosed-session warning at exit and, in a
        long-running process, as file descriptors that never come back.
        """
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
        """Perform one throttled request and translate its outcome.

        Throttling happens inside the retried unit so retries spend budget like
        any other request. Acquiring outside the retry loop would let a request
        that fails ten times send eleven requests on one token.

        Args:
            url: Absolute URL to fetch.
            params: Query parameters.
            archive_key: Filename stem for the archive, or None.

        Returns:
            The response body as text.

        Raises:
            RateLimitedError: On 429, or 503 carrying a Retry-After.
            TransientIngestionError: On timeout, connection failure, or 5xx.
            PermanentIngestionError: On other 4xx, oversized bodies, or
                redirect loops.
        """
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
        """Map an HTTP status code onto the ingestion exceptions.

        429 and 503-with-Retry-After are checked first: they carry an explicit
        instruction from the server, which outranks any local guess about
        whether the failure is retryable.

        Args:
            response: The response to inspect.
            url: The requested URL, for error messages.

        Raises:
            RateLimitedError: On 429, or 503 with a Retry-After header.
            TransientIngestionError: On other retryable statuses.
            PermanentIngestionError: On statuses retrying cannot fix.
        """
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
        """Persist a raw payload for later debugging.

        Best-effort by design: the listings are the deliverable, the archive is
        a convenience, and a full disk must not abort a scrape that is
        otherwise succeeding.

        Args:
            key: Filename stem, typically platform-queryslug-pN.
            body: The raw response text.
        """
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

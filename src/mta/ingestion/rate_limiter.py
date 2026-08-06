"""Client-side request throttling for the ingestion layer.

A token bucket is the right shape for this problem because marketplace traffic
is naturally bursty — a search page yields twenty listing URLs at once — but the
*sustained* rate is what a server actually cares about. The bucket lets you
spend a small burst immediately and then paces you at the refill rate, instead
of either serialising everything (slow, and no smoother from the server's point
of view) or firing twenty concurrent requests (fast, rude, and the quickest way
to earn an IP block).

The throttle is deliberately separated from the HTTP client so that a single
bucket can be shared across every coroutine hitting one host. Two scrapers each
holding their own "10 requests per minute" limiter are, from the server's
perspective, a scraper doing 20.
"""

from __future__ import annotations

import asyncio
import logging
import time
from types import TracebackType
from typing import Protocol, Self, runtime_checkable

logger = logging.getLogger(__name__)

#: Refuse to construct a limiter that would sleep implausibly long for one token.
#: Guards against a misconfigured ``requests_per_minute`` of, say, 0.001.
_MAX_REASONABLE_WAIT_SECONDS = 3600.0


@runtime_checkable
class RateLimiter(Protocol):
    """The throttling contract the HTTP layer depends on.

    A ``Protocol`` rather than a base class so that tests can substitute a
    no-op implementation without inheriting anything, and so the HTTP client's
    dependency is "something that can be acquired from" rather than "this
    specific class". Structural typing is the lighter-weight seam here.
    """

    async def acquire(self, tokens: float = 1.0) -> float:
        """Block until ``tokens`` may be spent, returning seconds spent waiting."""
        ...

    def penalize(self, seconds: float) -> None:
        """Suspend all issuance for ``seconds``, e.g. on a server ``Retry-After``."""
        ...


class TokenBucketRateLimiter:
    """An asyncio-safe token bucket shared across concurrent requests.

    Tokens accrue continuously at ``requests_per_minute / 60`` per second, up to
    ``burst`` in reserve. Each request spends one.

    The default ``burst`` of 1 means *strictly spaced* requests: no bursting at
    all. That is the conservative default on purpose — for this project the cost
    of being slightly slower is nothing, and the cost of being impolite to a
    marketplace is the whole project. Raise it deliberately if a source
    genuinely tolerates bursts.

    Concurrency note: :meth:`acquire` holds its lock across the sleep. That
    serialises waiters, which costs a little throughput and buys two things
    worth more — the budget can never be oversubscribed by racing coroutines,
    and waiters are served roughly first-come-first-served instead of one
    unlucky coroutine starving while others repeatedly win the race.

    Attributes:
        requests_per_minute: Sustained issuance rate.
        burst: Maximum tokens held in reserve.
    """

    def __init__(self, requests_per_minute: float, burst: int = 1) -> None:
        """Initialise the bucket, starting full.

        Args:
            requests_per_minute: Sustained rate. Must be positive.
            burst: Bucket capacity in tokens. Must be at least 1.

        Raises:
            ValueError: If the rate or burst is non-positive, or if the rate is
                so low that a single token would take over an hour to accrue —
                almost always a units mistake (per *hour* typed into a per
                *minute* field) rather than an intent.
        """
        if requests_per_minute <= 0:
            raise ValueError(f"requests_per_minute must be positive, got {requests_per_minute}.")
        if burst < 1:
            raise ValueError(f"burst must be at least 1, got {burst}.")

        self.requests_per_minute = requests_per_minute
        self.burst = burst
        self._refill_per_second = requests_per_minute / 60.0

        if 1.0 / self._refill_per_second > _MAX_REASONABLE_WAIT_SECONDS:
            raise ValueError(
                f"requests_per_minute={requests_per_minute} implies a "
                f"{1.0 / self._refill_per_second:.0f}s wait per request. "
                f"Check the units."
            )

        self._tokens = float(burst)
        # Monotonic, not wall clock: a clock adjustment or DST change must not
        # hand out a windfall of tokens or hang the bucket for an hour.
        self._last_refill = time.monotonic()
        self._blocked_until = 0.0
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """Credit tokens accrued since the last refill. Caller must hold the lock."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        self._tokens = min(float(self.burst), self._tokens + elapsed * self._refill_per_second)

    async def acquire(self, tokens: float = 1.0) -> float:
        """Wait until ``tokens`` are available, then spend them.

        Args:
            tokens: Cost of the operation. Values above 1 let an expensive call
                (an image download, say) consume proportionally more budget.

        Returns:
            Seconds spent waiting. Returned rather than merely logged so callers
            can surface real throttling cost in run statistics — "this run took
            eleven minutes, nine of them waiting" is the kind of number that
            tells you whether to parallelise or to slow down.

        Raises:
            ValueError: If ``tokens`` exceeds the bucket's capacity, which would
                otherwise deadlock forever waiting for a level the bucket can
                never reach.
        """
        if tokens > self.burst:
            raise ValueError(
                f"Cannot acquire {tokens} tokens from a bucket with burst={self.burst}; "
                f"this would wait forever."
            )

        started = time.monotonic()
        async with self._lock:
            while True:
                now = time.monotonic()

                # A server-directed penalty outranks the local budget entirely.
                if now < self._blocked_until:
                    await asyncio.sleep(self._blocked_until - now)
                    continue

                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    break

                deficit = tokens - self._tokens
                await asyncio.sleep(deficit / self._refill_per_second)

        waited = time.monotonic() - started
        if waited > 1.0:
            logger.debug("Rate limiter delayed a request by %.2fs", waited)
        return waited

    def penalize(self, seconds: float) -> None:
        """Suspend all issuance for a period, on the server's instruction.

        Called when a response carries ``429 Too Many Requests`` or a
        ``Retry-After`` header. This is the important half of rate limiting that
        client-side budgets alone miss: your own estimate of a polite rate is a
        guess, but a 429 is the server telling you the answer. Ignoring it and
        continuing to drip requests at the configured rate is how a temporary
        throttle becomes a permanent ban.

        Extends but never shortens an existing penalty, so overlapping 429s from
        concurrent requests cannot accidentally release the block early.

        Args:
            seconds: How long to suspend issuance. Non-positive values are
                ignored rather than treated as an unblock.
        """
        if seconds <= 0:
            return
        self._blocked_until = max(self._blocked_until, time.monotonic() + seconds)
        logger.warning("Rate limiter penalised for %.1fs by server instruction.", seconds)

    async def __aenter__(self) -> Self:
        """Acquire one token, for use as ``async with limiter:``."""
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Release nothing — tokens are spent on acquisition, not held."""
        return None


class NoOpRateLimiter:
    """A limiter that never delays, for sources that perform no network I/O.

    Used by the fixture replay source and by unit tests. Exists so that calling
    code never needs an ``if self.limiter is not None`` branch — the null object
    is cheaper than the conditional, and it keeps the throttled and unthrottled
    paths identical so tests exercise the real code path.
    """

    async def acquire(self, tokens: float = 1.0) -> float:
        """Return immediately, having waited zero seconds."""
        return 0.0

    def penalize(self, seconds: float) -> None:
        """Ignore the penalty; there is no remote server to be polite to."""
        return None

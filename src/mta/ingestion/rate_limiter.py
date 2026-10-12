"""Client-side request throttling."""

from __future__ import annotations

import asyncio
import logging
import time
from types import TracebackType
from typing import Protocol, Self, runtime_checkable

logger = logging.getLogger(__name__)

#: Refuse a limiter that would sleep implausibly long for one token.
_MAX_REASONABLE_WAIT_SECONDS = 3600.0


@runtime_checkable
class RateLimiter(Protocol):
    """The throttling contract the HTTP layer depends on."""

    async def acquire(self, tokens: float = 1.0) -> float:
        """Block until tokens may be spent, returning seconds spent waiting."""
        ...

    def penalize(self, seconds: float) -> None:
        """Suspend all issuance for a period, e.g. on a server Retry-After."""
        ...


class TokenBucketRateLimiter:
    """An asyncio-safe token bucket shared across concurrent requests."""

    def __init__(self, requests_per_minute: float, burst: int = 1) -> None:
        """Initialise the bucket, starting full."""
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
        # Monotonic, not wall clock: a clock adjustment must not hand out a
        # windfall of tokens or hang the bucket for an hour.
        self._last_refill = time.monotonic()
        self._blocked_until = 0.0
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """Credit tokens accrued since the last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        self._tokens = min(float(self.burst), self._tokens + elapsed * self._refill_per_second)

    async def acquire(self, tokens: float = 1.0) -> float:
        """Wait until tokens are available, then spend them."""
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
        """Suspend all issuance for a period, on the server's instruction."""
        if seconds <= 0:
            return
        self._blocked_until = max(self._blocked_until, time.monotonic() + seconds)
        logger.warning("Rate limiter penalised for %.1fs by server instruction.", seconds)

    async def __aenter__(self) -> Self:
        """Acquire one token, for use as `async with limiter:`."""
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Release nothing: tokens are spent on acquisition, not held."""


class NoOpRateLimiter:
    """A limiter that never delays, for sources with no network I/O."""

    async def acquire(self, tokens: float = 1.0) -> float:  # noqa: ARG002 - protocol conformance
        """Return immediately, having waited zero seconds."""
        return 0.0

    def penalize(self, seconds: float) -> None:  # noqa: ARG002 - protocol conformance
        """Ignore the penalty; there is no remote server to be polite to."""

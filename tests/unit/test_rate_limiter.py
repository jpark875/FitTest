"""Tests for the token bucket.

Rates are fast enough to keep the suite quick but slow enough that a broken
bucket cannot pass by accident: at 1200/min a token takes 50ms.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from mta.ingestion.rate_limiter import NoOpRateLimiter, RateLimiter, TokenBucketRateLimiter

FAST_RPM = 1200.0
TOKEN_SECONDS = 60.0 / FAST_RPM


class TestConstruction:
    @pytest.mark.parametrize("rpm", [0, -1, -0.5])
    def test_rejects_non_positive_rates(self, rpm: float) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            TokenBucketRateLimiter(requests_per_minute=rpm)

    def test_rejects_a_burst_below_one(self) -> None:
        with pytest.raises(ValueError, match="burst must be at least 1"):
            TokenBucketRateLimiter(requests_per_minute=60, burst=0)

    def test_rejects_a_rate_implying_an_hour_long_wait(self) -> None:
        # Almost always a per-hour figure typed into a per-minute field.
        with pytest.raises(ValueError, match="Check the units"):
            TokenBucketRateLimiter(requests_per_minute=0.001)


class TestAcquire:
    async def test_the_first_request_is_not_delayed(self) -> None:
        limiter = TokenBucketRateLimiter(requests_per_minute=FAST_RPM)
        assert await limiter.acquire() < TOKEN_SECONDS

    async def test_subsequent_requests_are_paced(self) -> None:
        limiter = TokenBucketRateLimiter(requests_per_minute=FAST_RPM)
        await limiter.acquire()
        assert await limiter.acquire() >= TOKEN_SECONDS * 0.8

    async def test_burst_is_spent_then_paced(self) -> None:
        limiter = TokenBucketRateLimiter(requests_per_minute=FAST_RPM, burst=3)
        assert sum([await limiter.acquire() for _ in range(3)]) < TOKEN_SECONDS
        assert await limiter.acquire() >= TOKEN_SECONDS * 0.8

    async def test_concurrent_callers_cannot_oversubscribe(self) -> None:
        # The reason the lock is held across the sleep.
        limiter = TokenBucketRateLimiter(requests_per_minute=FAST_RPM)
        started = time.monotonic()
        await asyncio.gather(*(limiter.acquire() for _ in range(5)))
        assert time.monotonic() - started >= TOKEN_SECONDS * 4 * 0.8

    async def test_refuses_an_acquisition_it_can_never_satisfy(self) -> None:
        limiter = TokenBucketRateLimiter(requests_per_minute=FAST_RPM, burst=2)
        with pytest.raises(ValueError, match="wait forever"):
            await limiter.acquire(tokens=3)


class TestPenalize:
    async def test_a_server_penalty_outranks_the_local_budget(self) -> None:
        limiter = TokenBucketRateLimiter(requests_per_minute=FAST_RPM, burst=5)
        limiter.penalize(0.2)
        assert await limiter.acquire() >= 0.15

    async def test_penalties_extend_but_never_shorten(self) -> None:
        limiter = TokenBucketRateLimiter(requests_per_minute=FAST_RPM, burst=5)
        limiter.penalize(0.25)
        limiter.penalize(0.01)
        assert await limiter.acquire() >= 0.2

    @pytest.mark.parametrize("seconds", [0, -5])
    async def test_non_positive_penalties_are_not_an_unblock(self, seconds: float) -> None:
        limiter = TokenBucketRateLimiter(requests_per_minute=FAST_RPM, burst=5)
        limiter.penalize(0.2)
        limiter.penalize(seconds)
        assert await limiter.acquire() >= 0.15

    async def test_works_as_an_async_context_manager(self) -> None:
        limiter = TokenBucketRateLimiter(requests_per_minute=FAST_RPM)
        async with limiter:
            pass
        assert await limiter.acquire() >= TOKEN_SECONDS * 0.8


class TestNoOpRateLimiter:
    async def test_never_delays(self) -> None:
        limiter = NoOpRateLimiter()
        assert await limiter.acquire() == 0.0
        assert await limiter.acquire(tokens=100) == 0.0

    def test_swallows_penalties(self) -> None:
        NoOpRateLimiter().penalize(600)

    def test_both_implementations_satisfy_the_protocol(self) -> None:
        assert isinstance(NoOpRateLimiter(), RateLimiter)
        assert isinstance(TokenBucketRateLimiter(requests_per_minute=60), RateLimiter)

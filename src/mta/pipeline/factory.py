"""Composition root helpers: build sources and classifiers from Settings."""

from __future__ import annotations

from typing import Literal

from mta.config import ConfigurationError, Settings
from mta.ingestion.base import ListingSource
from mta.ingestion.fixtures import FixtureSource
from mta.ingestion.http import ThrottledHttpClient
from mta.ingestion.live import DepopSource, ThredUpSource
from mta.ingestion.rate_limiter import TokenBucketRateLimiter
from mta.models.listing import Platform
from mta.processing.classifier import AnthropicClassifier, KeywordClassifier, TrendClassifier

ClassifierChoice = Literal["auto", "keyword", "anthropic"]


def build_source(settings: Settings) -> ListingSource:
    """Create the configured source. Live sources need explicit authorisation."""
    if settings.ingestion_source is Platform.FIXTURE:
        return FixtureSource(settings.fixtures_dir)

    if not settings.authorised_to_collect:
        raise ConfigurationError(
            f"{settings.ingestion_source.value} prohibits automated collection. Set "
            "MTA_AUTHORISED_TO_COLLECT=true only if you are authorised to collect from it, "
            "or use MTA_INGESTION_SOURCE=fixture."
        )
    client = ThrottledHttpClient(
        user_agent=settings.user_agent,
        timeout_seconds=settings.request_timeout_seconds,
        max_retries=settings.max_retries,
        rate_limiter=TokenBucketRateLimiter(settings.requests_per_minute),
        archive_dir=settings.raw_dir,
    )
    if settings.ingestion_source is Platform.DEPOP:
        return DepopSource(client)
    return ThredUpSource(client)


def build_classifier(settings: Settings, choice: ClassifierChoice = "auto") -> TrendClassifier:
    """Create the classifier; `auto` uses the vision model when a key is configured."""
    if choice == "auto":
        choice = "anthropic" if settings.anthropic_api_key is not None else "keyword"
    if choice == "keyword":
        return KeywordClassifier()
    return AnthropicClassifier(
        settings.require_api_key(),
        model=settings.vision_model,
        prompt_version=settings.prompt_version,
        timeout_seconds=settings.llm_timeout_seconds,
    )

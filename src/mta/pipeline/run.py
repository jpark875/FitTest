"""One pass of the pipeline: ingest, classify, store, value."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal

from mta.ingestion.base import ListingSource, SearchQuery
from mta.models.listing import RawListing
from mta.processing.classifier import TrendClassifier, classify_all
from mta.processing.valuation import select_comparables, value_listing
from mta.storage.repository import Repository

logger = logging.getLogger(__name__)


@dataclass
class RunReport:
    """Counters for one pass."""

    ingested: int = 0
    unanalyzable: int = 0
    new: int = 0
    classified: int = 0
    valued: int = 0
    failures: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line description."""
        return (
            f"{self.ingested} ingested ({self.new} new), {self.unanalyzable} without images or "
            f"price, {self.classified} classified, {self.valued} valued, "
            f"{len(self.failures)} failed"
        )


async def run_pipeline(
    source: ListingSource,
    classifier: TrendClassifier,
    repository: Repository,
    query: SearchQuery,
    *,
    platform_fee_rate: Decimal,
    shipping_cost: Decimal,
    concurrency: int = 4,
) -> RunReport:
    """Ingest a query, store everything, then value each listing against what is stored.

    Listings are stored before any is valued so comparables include the rest of the batch.
    """
    report = RunReport()
    listings: list[RawListing] = []
    async with source:
        async for listing in source.fetch_listings(query):
            report.ingested += 1
            if listing.is_analyzable:
                listings.append(listing)
            else:
                report.unanalyzable += 1

    assessments, report.failures = await classify_all(classifier, listings, concurrency)
    report.classified = len(assessments)

    for listing in listings:
        assessment = assessments.get(listing.fingerprint)
        if assessment is None:
            continue
        report.new += repository.upsert_listing(listing)
        repository.upsert_assessment(listing.fingerprint, assessment)

    for listing in listings:
        assessment = assessments.get(listing.fingerprint)
        if assessment is None:
            continue
        candidates = repository.comparables(
            assessment.inferred_category, listing.currency, exclude=listing.fingerprint
        )
        comps = select_comparables(listing, assessment, candidates)
        valuation = value_listing(
            listing,
            assessment,
            comps,
            platform_fee_rate=platform_fee_rate,
            shipping_cost=shipping_cost,
        )
        repository.upsert_valuation(listing.fingerprint, valuation)
        report.valued += 1

    logger.info("pipeline pass complete: %s", report.summary())
    return report

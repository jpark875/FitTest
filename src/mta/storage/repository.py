"""Warehouse access: idempotent upserts and the queries the dashboard and CLI need."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from mta.models.listing import (
    EnrichedListing,
    ItemCondition,
    Platform,
    RawListing,
    TrendAssessment,
    TrendTag,
    Valuation,
)
from mta.processing.valuation import Comparable
from mta.storage.tables import AssessmentRow, Base, ListingRow, ValuationRow


@dataclass(frozen=True)
class Opportunity:
    """A stored listing with its latest assessment and valuation."""

    listing: RawListing
    assessment: TrendAssessment
    valuation: Valuation

    @property
    def enriched(self) -> EnrichedListing:
        """The same data as a validated EnrichedListing."""
        return EnrichedListing(
            listing=self.listing, assessment=self.assessment, valuation=self.valuation
        )


class Repository:
    """Thin persistence layer over SQLAlchemy. Sessions are short-lived and per call."""

    def __init__(self, url: str, *, echo: bool = False) -> None:
        """Open the database, creating the parent directory for file-backed SQLite."""
        parsed = make_url(url)
        if parsed.get_backend_name() == "sqlite" and parsed.database not in (None, "", ":memory:"):
            Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = create_engine(url, echo=echo)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    def create_schema(self) -> None:
        """Create any missing tables. Idempotent."""
        Base.metadata.create_all(self.engine)

    def dispose(self) -> None:
        """Release pooled connections."""
        self.engine.dispose()

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Commit on success, roll back on error."""
        with self._sessions() as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    # -- Writes ------------------------------------------------------------- #

    def upsert_listing(self, listing: RawListing) -> bool:
        """Insert or refresh a listing by fingerprint. Returns True if it was new."""
        with self.session() as session:
            row = session.scalar(
                select(ListingRow).where(ListingRow.fingerprint == listing.fingerprint)
            )
            created = row is None
            if row is None:
                row = ListingRow(fingerprint=listing.fingerprint, first_seen_at=datetime.now(UTC))
                session.add(row)
            row.platform = listing.platform.value
            row.external_id = listing.external_id
            row.url = str(listing.url)
            row.title = listing.title
            row.description = listing.description
            row.asking_price = listing.asking_price
            row.currency = listing.currency
            row.condition = listing.condition.value
            row.brand_raw = listing.brand_raw
            row.size_raw = listing.size_raw
            row.seller_username = listing.seller_username
            row.image_urls = [str(u) for u in listing.image_urls]
            row.listed_at = listing.listed_at
            row.captured_at = listing.captured_at
            return created

    def upsert_assessment(self, fingerprint: str, assessment: TrendAssessment) -> None:
        """Store the verdict for a stored listing, replacing one from the same model and prompt."""
        with self.session() as session:
            listing_id = self._listing_id(session, fingerprint)
            row = session.scalar(
                select(AssessmentRow).where(
                    AssessmentRow.listing_id == listing_id,
                    AssessmentRow.model_id == assessment.model_id,
                    AssessmentRow.prompt_version == assessment.prompt_version,
                )
            )
            if row is None:
                row = AssessmentRow(
                    listing_id=listing_id,
                    model_id=assessment.model_id,
                    prompt_version=assessment.prompt_version,
                )
                session.add(row)
            row.trend_tags = [t.value for t in assessment.trend_tags]
            row.primary_trend = assessment.primary_trend.value
            row.inferred_brand = assessment.inferred_brand
            row.inferred_category = assessment.inferred_category
            row.inferred_decade = assessment.inferred_decade
            row.style_confidence = assessment.style_confidence
            row.estimated_retail_value = assessment.estimated_retail_value
            row.reasoning = assessment.reasoning
            row.assessed_at = assessment.assessed_at

    def upsert_valuation(self, fingerprint: str, valuation: Valuation) -> None:
        """Store the latest valuation, tied to the listing's newest assessment."""
        with self.session() as session:
            listing_id = self._listing_id(session, fingerprint)
            assessment_id = session.scalar(
                select(AssessmentRow.id)
                .where(AssessmentRow.listing_id == listing_id)
                .order_by(AssessmentRow.assessed_at.desc(), AssessmentRow.id.desc())
            )
            if assessment_id is None:
                raise LookupError(f"no assessment stored for {fingerprint}")
            row = session.scalar(select(ValuationRow).where(ValuationRow.listing_id == listing_id))
            if row is None:
                row = ValuationRow(listing_id=listing_id)
                session.add(row)
            row.assessment_id = assessment_id
            row.asking_price = valuation.asking_price
            row.currency = valuation.currency
            row.estimated_resale_value = valuation.estimated_resale_value
            row.comparable_count = valuation.comparable_count
            row.median_comp_price = valuation.median_comp_price
            row.platform_fee_rate = float(valuation.platform_fee_rate)
            row.shipping_cost = valuation.shipping_cost
            row.valuation_confidence = valuation.valuation_confidence
            row.arbitrage_score = valuation.arbitrage_score
            row.net_proceeds = valuation.net_proceeds
            row.absolute_margin = valuation.absolute_margin
            row.margin_pct = float(valuation.margin_pct)
            row.enriched_at = datetime.now(UTC)

    @staticmethod
    def _listing_id(session: Session, fingerprint: str) -> int:
        listing_id = session.scalar(
            select(ListingRow.id).where(ListingRow.fingerprint == fingerprint)
        )
        if listing_id is None:
            raise LookupError(f"listing {fingerprint} is not stored")
        return listing_id

    # -- Reads -------------------------------------------------------------- #

    def count_listings(self) -> int:
        """Number of stored listings."""
        with self.session() as session:
            return session.scalar(select(func.count()).select_from(ListingRow)) or 0

    def comparables(
        self, category: str, currency: str, *, exclude: str | None = None, limit: int = 300
    ) -> list[Comparable]:
        """Stored listings in a category and currency, newest first, as price references."""
        stmt = (
            select(ListingRow, AssessmentRow)
            .join(AssessmentRow, AssessmentRow.listing_id == ListingRow.id)
            .where(AssessmentRow.inferred_category == category, ListingRow.currency == currency)
            .order_by(ListingRow.captured_at.desc())
            .limit(limit)
        )
        seen: set[str] = set()
        comps = []
        with self.session() as session:
            for listing, assessment in session.execute(stmt):
                if listing.fingerprint == exclude or listing.fingerprint in seen:
                    continue
                seen.add(listing.fingerprint)
                comps.append(
                    Comparable(
                        fingerprint=listing.fingerprint,
                        title=listing.title,
                        price=listing.asking_price,
                        currency=listing.currency,
                        condition=ItemCondition(listing.condition),
                        category=assessment.inferred_category,
                        brand=assessment.inferred_brand,
                    )
                )
        return comps

    def opportunities(
        self,
        *,
        min_margin_pct: Decimal | None = None,
        min_absolute_margin: Decimal | None = None,
        min_confidence: float = 0.0,
        trends: Sequence[TrendTag] | None = None,
        limit: int | None = None,
    ) -> list[Opportunity]:
        """Valued listings above the given floors, best arbitrage score first."""
        stmt = (
            select(ListingRow, AssessmentRow, ValuationRow)
            .join(ValuationRow, ValuationRow.listing_id == ListingRow.id)
            .join(AssessmentRow, AssessmentRow.id == ValuationRow.assessment_id)
            .where(ValuationRow.valuation_confidence >= min_confidence)
            .order_by(ValuationRow.arbitrage_score.desc(), ListingRow.fingerprint)
        )
        if min_margin_pct is not None:
            stmt = stmt.where(ValuationRow.margin_pct >= float(min_margin_pct))
        if min_absolute_margin is not None:
            stmt = stmt.where(ValuationRow.absolute_margin >= min_absolute_margin)
        if trends:
            stmt = stmt.where(AssessmentRow.primary_trend.in_([t.value for t in trends]))
        if limit is not None:
            stmt = stmt.limit(limit)
        with self.session() as session:
            return [_hydrate(*row) for row in session.execute(stmt)]

    def trend_summary(self) -> list[tuple[str, int, float]]:
        """Per primary trend: listing count and mean margin fraction, largest trend first."""
        stmt = (
            select(
                AssessmentRow.primary_trend,
                func.count(ValuationRow.id),
                func.avg(ValuationRow.margin_pct),
            )
            .join(ValuationRow, ValuationRow.assessment_id == AssessmentRow.id)
            .group_by(AssessmentRow.primary_trend)
            .order_by(func.count(ValuationRow.id).desc(), AssessmentRow.primary_trend)
        )
        with self.session() as session:
            return [(trend, int(n), float(avg or 0.0)) for trend, n, avg in session.execute(stmt)]


def _hydrate(
    listing: ListingRow, assessment: AssessmentRow, valuation: ValuationRow
) -> Opportunity:
    return Opportunity(
        listing=RawListing(
            platform=Platform(listing.platform),
            external_id=listing.external_id,
            url=listing.url,
            title=listing.title,
            description=listing.description,
            asking_price=listing.asking_price,
            currency=listing.currency,
            condition=ItemCondition(listing.condition),
            brand_raw=listing.brand_raw,
            size_raw=listing.size_raw,
            seller_username=listing.seller_username,
            image_urls=listing.image_urls,
            listed_at=listing.listed_at,
            captured_at=listing.captured_at,
        ),
        assessment=TrendAssessment(
            trend_tags=[TrendTag(t) for t in assessment.trend_tags],
            inferred_brand=assessment.inferred_brand,
            inferred_category=assessment.inferred_category,
            inferred_decade=assessment.inferred_decade,
            style_confidence=assessment.style_confidence,
            estimated_retail_value=assessment.estimated_retail_value,
            reasoning=assessment.reasoning,
            model_id=assessment.model_id,
            prompt_version=assessment.prompt_version,
            assessed_at=assessment.assessed_at,
        ),
        valuation=Valuation(
            asking_price=valuation.asking_price,
            currency=valuation.currency,
            estimated_resale_value=valuation.estimated_resale_value,
            comparable_count=valuation.comparable_count,
            median_comp_price=valuation.median_comp_price,
            platform_fee_rate=Decimal(str(valuation.platform_fee_rate)),
            shipping_cost=valuation.shipping_cost,
            valuation_confidence=valuation.valuation_confidence,
            arbitrage_score=valuation.arbitrage_score,
        ),
    )

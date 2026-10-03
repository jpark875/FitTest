"""SQLAlchemy schema. Money is stored as integer minor units so SQLite stays exact."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

_CENTS = Decimal("0.01")


class MinorUnits(TypeDecorator[Decimal]):
    """Decimal money persisted as an integer count of hundredths."""

    impl = Integer
    cache_ok = True

    def process_bind_param(self, value: Decimal | None, dialect: Dialect) -> int | None:  # noqa: ARG002
        """Convert to integer hundredths on the way in."""
        return None if value is None else int((value.quantize(_CENTS) * 100).to_integral_value())

    def process_result_value(self, value: Any, dialect: Dialect) -> Decimal | None:  # noqa: ARG002
        """Convert back to a two-place Decimal on the way out."""
        return None if value is None else (Decimal(int(value)) / 100).quantize(_CENTS)


class UtcDateTime(TypeDecorator[datetime]):
    """Timezone-aware datetimes; SQLite drops tzinfo, so values are normalised to UTC."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:  # noqa: ARG002
        """Store as naive UTC."""
        return None if value is None else value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:  # noqa: ARG002
        """Restore the UTC tzinfo."""
        return None if value is None else value.replace(tzinfo=UTC)


class Base(DeclarativeBase):
    """Declarative base for every table."""


class ListingRow(Base):
    """One marketplace listing, keyed by platform-scoped fingerprint."""

    __tablename__ = "listings"

    id: Mapped[int] = mapped_column(primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    platform: Mapped[str] = mapped_column(String(32))
    external_id: Mapped[str] = mapped_column(String(128))
    url: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(String(500))
    description: Mapped[str | None] = mapped_column(Text)
    asking_price: Mapped[Decimal] = mapped_column(MinorUnits)
    currency: Mapped[str] = mapped_column(String(3))
    condition: Mapped[str] = mapped_column(String(32))
    brand_raw: Mapped[str | None] = mapped_column(String(200))
    size_raw: Mapped[str | None] = mapped_column(String(64))
    seller_username: Mapped[str | None] = mapped_column(String(128))
    image_urls: Mapped[list[str]] = mapped_column(JSON)
    listed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    captured_at: Mapped[datetime] = mapped_column(UtcDateTime)
    first_seen_at: Mapped[datetime] = mapped_column(UtcDateTime)


class AssessmentRow(Base):
    """The vision verdict for a listing, one per (model, prompt version)."""

    __tablename__ = "assessments"
    __table_args__ = (UniqueConstraint("listing_id", "model_id", "prompt_version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    listing_id: Mapped[int] = mapped_column(ForeignKey("listings.id"), index=True)
    trend_tags: Mapped[list[str]] = mapped_column(JSON)
    primary_trend: Mapped[str] = mapped_column(String(32), index=True)
    inferred_brand: Mapped[str | None] = mapped_column(String(200))
    inferred_category: Mapped[str] = mapped_column(String(64), index=True)
    inferred_decade: Mapped[str | None] = mapped_column(String(16))
    style_confidence: Mapped[float] = mapped_column(Float)
    estimated_retail_value: Mapped[Decimal | None] = mapped_column(MinorUnits)
    reasoning: Mapped[str] = mapped_column(Text)
    model_id: Mapped[str] = mapped_column(String(100))
    prompt_version: Mapped[str] = mapped_column(String(32))
    assessed_at: Mapped[datetime] = mapped_column(UtcDateTime)


class ValuationRow(Base):
    """The latest valuation for a listing. Margins are stored for SQL ranking."""

    __tablename__ = "valuations"

    id: Mapped[int] = mapped_column(primary_key=True)
    listing_id: Mapped[int] = mapped_column(ForeignKey("listings.id"), unique=True)
    assessment_id: Mapped[int] = mapped_column(ForeignKey("assessments.id"))
    asking_price: Mapped[Decimal] = mapped_column(MinorUnits)
    currency: Mapped[str] = mapped_column(String(3))
    estimated_resale_value: Mapped[Decimal] = mapped_column(MinorUnits)
    comparable_count: Mapped[int] = mapped_column(Integer)
    median_comp_price: Mapped[Decimal | None] = mapped_column(MinorUnits)
    platform_fee_rate: Mapped[float] = mapped_column(Float)
    shipping_cost: Mapped[Decimal] = mapped_column(MinorUnits)
    valuation_confidence: Mapped[float] = mapped_column(Float)
    arbitrage_score: Mapped[float] = mapped_column(Float, index=True)
    net_proceeds: Mapped[Decimal] = mapped_column(MinorUnits)
    absolute_margin: Mapped[Decimal] = mapped_column(MinorUnits)
    margin_pct: Mapped[float] = mapped_column(Float)
    enriched_at: Mapped[datetime] = mapped_column(UtcDateTime)

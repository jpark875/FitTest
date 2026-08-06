"""Typed contracts exchanged between every layer of the engine.

This module is deliberately dependency-free with respect to the rest of the
package: ingestion, processing, storage, and the dashboard all import *from*
here, and nothing here imports from them. That one-way dependency is what makes
the layers independently replaceable.

Business context
----------------
The engine's thesis is that peer-to-peer resale sellers systematically
misprice alternative/streetwear garments because they list against the
*brand's* market value rather than the *micro-trend's* market value. A plain
90s Carhartt jacket and a currently-viral one look identical to a keyword
search; they do not look identical to a vision model. These models encode the
three stages of that thesis:

1. :class:`RawListing`  — what a seller actually published (untrusted input).
2. :class:`TrendAssessment` — what the vision model inferred about it.
3. :class:`Valuation` — what that inference is worth in money, net of fees.

Money is modelled as :class:`~decimal.Decimal` throughout. Floats are banned
here: ``0.1 + 0.2 != 0.3`` is an amusing footnote in a tutorial and a
reconciliation bug in a pipeline that aggregates thousands of margins.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Annotated, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    computed_field,
    field_validator,
    model_validator,
)

# --------------------------------------------------------------------------- #
# Module constants
# --------------------------------------------------------------------------- #

#: Cap on images sent to the vision model. Beyond ~8 the marginal signal is
#: negligible while per-listing cost keeps rising linearly.
MAX_IMAGES_PER_LISTING = 8

#: Matches the first numeric run in a price string, tolerating thousands
#: separators and either decimal convention: "£1,299.00", "18,50 EUR", "$24".
_PRICE_PATTERN = re.compile(r"(\d[\d,\s.]*)")

#: Symbol -> ISO 4217. Sellers rarely write the currency code explicitly.
_CURRENCY_SYMBOLS: dict[str, str] = {"£": "GBP", "$": "USD", "€": "EUR", "¥": "JPY"}

#: Two decimal places is the resolution of every currency we handle.
_CENTS = Decimal("0.01")

#: A non-negative monetary amount. Reused everywhere money appears.
Money = Annotated[Decimal, Field(ge=Decimal(0), max_digits=10, decimal_places=2)]

#: A model-reported probability. Used for calibration, never for hard gating.
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class MalformedListingError(ValueError):
    """Raised when scraped data cannot be coerced into a :class:`RawListing`.

    Ingestion catches this per-listing and increments a skip counter rather than
    aborting the batch. A single seller who typed "make me an offer :)" into the
    price field must not cost us the other 499 listings in the page.
    """


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class Platform(StrEnum):
    """A marketplace the engine can ingest from.

    ``FIXTURE`` is a first-class member, not a test artifact: it identifies
    listings replayed from recorded payloads so the pipeline runs end-to-end
    with no network access.
    """

    DEPOP = "depop"
    THREDUP = "thredup"
    FIXTURE = "fixture"


class TrendTag(StrEnum):
    """A micro-trend the engine tracks.

    This is a *closed* vocabulary, which is the point. Letting the model invent
    free-text tags produces an unqueryable long tail ("y2k", "Y2K vibes",
    "2000s") and makes trend-level aggregation impossible. Adding a trend is a
    deliberate schema change with a migration, not an emergent side effect.
    """

    Y2K = "y2k"
    GRUNGE = "grunge"
    RETRO_SKATER = "retro_skater"
    GORPCORE = "gorpcore"
    WORKWEAR = "workwear"
    ARCHIVE_TECHWEAR = "archive_techwear"
    VINTAGE_BAND_TEE = "vintage_band_tee"
    COQUETTE = "coquette"
    NONE = "none"


class ItemCondition(StrEnum):
    """Normalized condition ladder.

    Platforms use incompatible vocabularies (Depop: "Brand new"/"Used - like
    new"; ThredUp: "New With Tags"/"Excellent"). Normalizing at the boundary
    means the valuation layer applies one condition discount curve instead of
    one per platform.
    """

    NEW_WITH_TAGS = "new_with_tags"
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Defensive parsing helpers
# --------------------------------------------------------------------------- #


def parse_price(raw: str, default_currency: str = "GBP") -> tuple[Decimal, str]:
    """Coerce a seller-authored price string into an amount and a currency.

    Seller input is free text. Observed in the wild: ``"£24"``, ``"24.00 GBP"``,
    ``"$1,299"``, ``"18,50"``, ``"FREE"``, ``"offers"``, ``"£20 ono"``. We accept
    what we can and reject the rest loudly rather than silently defaulting to
    zero, which would manufacture an infinite-margin arbitrage opportunity out
    of a listing that has no price at all.

    Args:
        raw: The price text exactly as scraped.
        default_currency: ISO 4217 code assumed when no symbol is present.
            Set this per-platform (Depop UK defaults to GBP, ThredUp to USD).

    Returns:
        A ``(amount, currency_code)`` pair, amount quantized to two decimals.

    Raises:
        MalformedListingError: If no parseable number is present, or the number
            is present but not representable as a decimal amount.
    """
    if not raw or not raw.strip():
        raise MalformedListingError("Price field was empty.")

    currency = next(
        (code for symbol, code in _CURRENCY_SYMBOLS.items() if symbol in raw),
        default_currency,
    )

    match = _PRICE_PATTERN.search(raw)
    if match is None:
        raise MalformedListingError(f"No numeric value found in price {raw!r}.")

    candidate = match.group(1).strip().rstrip(".,").replace(" ", "")

    # Disambiguate separators: if the final separator is followed by exactly two
    # digits it is a decimal point; otherwise every separator is a thousands mark.
    if re.search(r"[.,]\d{2}$", candidate):
        candidate = candidate[:-3].replace(",", "").replace(".", "") + "." + candidate[-2:]
    else:
        candidate = candidate.replace(",", "").replace(".", "")

    try:
        amount = Decimal(candidate).quantize(_CENTS)
    except (InvalidOperation, ArithmeticError) as exc:
        raise MalformedListingError(f"Could not parse price {raw!r} as a decimal.") from exc

    if amount < 0:
        raise MalformedListingError(f"Negative price parsed from {raw!r}.")

    return amount, currency


def parse_condition(raw: str | None) -> ItemCondition:
    """Map a platform's condition string onto the normalized ladder.

    Unknown vocabulary degrades to :attr:`ItemCondition.UNKNOWN` rather than
    raising: an unrecognized condition makes a listing *less confidently*
    valued, not unusable. The valuation layer applies its most conservative
    discount to ``UNKNOWN``, so the failure mode is a missed opportunity rather
    than a bad buy.

    Args:
        raw: The platform's condition label, or ``None`` if absent.

    Returns:
        The matching :class:`ItemCondition`, defaulting to ``UNKNOWN``.
    """
    if not raw:
        return ItemCondition.UNKNOWN

    text = raw.strip().lower()
    if "tag" in text or text in {"brand new", "new"}:
        return ItemCondition.NEW_WITH_TAGS
    if "excellent" in text or "like new" in text:
        return ItemCondition.EXCELLENT
    if "very good" in text or "good" in text:
        return ItemCondition.GOOD
    if "fair" in text or "used" in text:
        return ItemCondition.FAIR
    if "poor" in text or "distress" in text or "flaw" in text:
        return ItemCondition.POOR
    return ItemCondition.UNKNOWN


# --------------------------------------------------------------------------- #
# Stage 1 — what the seller published
# --------------------------------------------------------------------------- #


class RawListing(BaseModel):
    """A single marketplace listing as captured at a point in time.

    Frozen, because a capture is a historical fact. If the seller edits the
    price we want a *second* row, not a mutated first one — price movement over
    time is itself a signal that an item is not selling and the seller may be
    ready to negotiate.

    ``extra="forbid"`` is intentional. When a platform changes its markup and a
    scraper starts emitting a field this contract does not know about, the
    pipeline should fail loudly in CI rather than quietly discarding data.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    platform: Platform
    external_id: str = Field(min_length=1, description="The platform's own listing ID.")
    url: HttpUrl
    title: str = Field(min_length=1, max_length=500)
    description: str | None = Field(default=None, max_length=10_000)
    asking_price: Money
    currency: str = Field(min_length=3, max_length=3, description="ISO 4217 code.")
    condition: ItemCondition = ItemCondition.UNKNOWN
    brand_raw: str | None = Field(default=None, description="Seller-declared brand, untrusted.")
    size_raw: str | None = Field(default=None, description="Seller-declared size, unnormalized.")
    seller_username: str | None = None
    image_urls: list[HttpUrl] = Field(default_factory=list, max_length=MAX_IMAGES_PER_LISTING)
    listed_at: AwareDatetime | None = Field(
        default=None, description="When the seller posted it, if the platform exposes it."
    )
    captured_at: AwareDatetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When we scraped it. Drives staleness checks in the dashboard.",
    )

    @field_validator("currency")
    @classmethod
    def _uppercase_currency(cls, value: str) -> str:
        """Normalize the currency code so ``gbp`` and ``GBP`` do not fragment joins."""
        return value.upper()

    @field_validator("image_urls")
    @classmethod
    def _cap_images(cls, value: list[HttpUrl]) -> list[HttpUrl]:
        """Truncate to :data:`MAX_IMAGES_PER_LISTING` to bound per-listing vision cost."""
        return value[:MAX_IMAGES_PER_LISTING]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fingerprint(self) -> str:
        """Stable cross-run identity for this listing.

        Used as the natural key for upserts. Scoped by platform because listing
        IDs are only unique within a marketplace, and two platforms will
        eventually collide on a bare integer ID.
        """
        return f"{self.platform.value}:{self.external_id}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_analyzable(self) -> bool:
        """Whether this listing carries enough signal to be worth an API call.

        Vision enrichment is the dominant per-listing cost, so this gate runs
        before the model is invoked. A listing with no imagery cannot be
        trend-classified with any confidence, and paying for a text-only guess
        is how a portfolio project burns its API budget on noise.
        """
        return bool(self.image_urls) and self.asking_price > 0


# --------------------------------------------------------------------------- #
# Stage 2 — what the vision model inferred
# --------------------------------------------------------------------------- #


class TrendAssessment(BaseModel):
    """The structured verdict returned by the vision/LLM layer for one listing.

    This doubles as the JSON schema handed to the model as a structured-output
    contract, which is why every field is primitive and closed-vocabulary: the
    same definition that validates the response also constrains it.

    ``model_id`` and ``prompt_version`` are recorded on every row so that when
    scoring quality shifts, you can attribute the shift to a model change or a
    prompt change instead of guessing. Treat them as you would a code version.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trend_tags: list[TrendTag] = Field(min_length=1, max_length=3)
    inferred_brand: str | None = Field(
        default=None, description="Brand read from the garment itself, not the seller's text."
    )
    inferred_category: str = Field(min_length=1, description="e.g. 'outerwear', 'denim'.")
    inferred_decade: str | None = Field(default=None, description="e.g. '1990s'.")
    style_confidence: Confidence
    estimated_retail_value: Money | None = Field(
        default=None, description="Model's view of comparable market price, before fees."
    )
    reasoning: str = Field(
        max_length=1_000, description="One-paragraph justification, surfaced in the dashboard."
    )
    model_id: str = Field(description="Exact model that produced this, e.g. 'claude-opus-5'.")
    prompt_version: str = Field(description="Version of the prompt template used.")
    assessed_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def primary_trend(self) -> TrendTag:
        """The highest-signal trend, by convention the first tag the model returned.

        The prompt instructs the model to order tags by confidence, so position
        carries meaning. Dashboards group by this; the full list is kept for
        cross-trend analysis.
        """
        return self.trend_tags[0]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_on_trend(self) -> bool:
        """Whether the model placed this garment in any tracked micro-trend."""
        return self.primary_trend is not TrendTag.NONE


# --------------------------------------------------------------------------- #
# Stage 3 — what the inference is worth
# --------------------------------------------------------------------------- #


class Valuation(BaseModel):
    """The money view of a listing: what you'd net, and how confident we are.

    The split of responsibility here is deliberate and worth internalizing:
    this model owns *arithmetic* (margin is definitionally revenue minus cost),
    while :mod:`mta.processing.valuation` owns *judgment* (which comps count,
    how to weight confidence, where the buy threshold sits). Arithmetic in the
    model guarantees the dashboard and the pipeline can never disagree about
    what a margin is; judgment stays in the processing layer where it can be
    tuned and backtested without a schema migration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    asking_price: Money
    currency: str = Field(min_length=3, max_length=3)
    estimated_resale_value: Money = Field(
        description="Expected achievable sale price, condition-adjusted."
    )
    comparable_count: int = Field(
        ge=0, description="Number of comps behind the estimate. Zero means model-only."
    )
    median_comp_price: Money | None = None
    platform_fee_rate: Decimal = Field(
        default=Decimal("0.10"),
        ge=Decimal(0),
        le=Decimal(1),
        description="Marketplace commission on resale. ~10% on Depop at time of writing.",
    )
    shipping_cost: Money = Field(
        default=Decimal("0.00"), description="Estimated cost to receive and re-ship."
    )
    valuation_confidence: Confidence = Field(
        description="Combines model confidence with comp depth; set by the valuation layer."
    )
    arbitrage_score: float = Field(
        ge=0.0,
        description=(
            "Ranking scalar assigned by mta.processing.valuation. Stored rather than "
            "derived so historical rows survive a change to the scoring formula."
        ),
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def net_proceeds(self) -> Decimal:
        """Cash actually received after the marketplace takes its cut and you ship.

        Reasoning against gross resale price is the single most common way a
        flipping model overstates itself: a £40 gross on a £25 buy looks like a
        60% return and is closer to 20% once commission and postage land.
        """
        fees = (self.estimated_resale_value * self.platform_fee_rate).quantize(_CENTS)
        return (self.estimated_resale_value - fees - self.shipping_cost).quantize(_CENTS)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def absolute_margin(self) -> Decimal:
        """Expected profit in currency units. Can legitimately be negative."""
        return (self.net_proceeds - self.asking_price).quantize(_CENTS)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def margin_pct(self) -> Decimal:
        """Return on capital deployed, as a decimal fraction (``0.35`` is +35%).

        Percentage rather than absolute margin is what makes a £8 profit on a £10
        item comparable to a £30 profit on a £120 item, which is the comparison
        the ranking actually needs to make.
        """
        if self.asking_price <= 0:
            return Decimal("0.00")
        return (self.absolute_margin / self.asking_price).quantize(Decimal("0.0001"))

    @model_validator(mode="after")
    def _check_currency_coherence(self) -> Self:
        """Reject valuations whose comps were priced in a different currency.

        Cross-currency comparison is a real requirement (a UK buyer sourcing US
        stock) but it needs an explicit FX step. Silently subtracting dollars
        from pounds is the kind of error that produces a confident, wrong, and
        very expensive recommendation.

        Raises:
            ValueError: If the currency code is not a well-formed ISO 4217 code.
        """
        if not self.currency.isalpha():
            raise ValueError(f"Currency {self.currency!r} is not a valid ISO 4217 code.")
        return self


# --------------------------------------------------------------------------- #
# The composed record
# --------------------------------------------------------------------------- #


class EnrichedListing(BaseModel):
    """A listing that has completed the full pipeline: captured, assessed, valued.

    This is the unit the storage layer persists and the dashboard renders. It
    is a composition rather than a flat model so that each stage's provenance
    stays intact — you can always answer "which model version priced this, and
    what did the seller originally write?" without a join.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    listing: RawListing
    assessment: TrendAssessment
    valuation: Valuation
    enriched_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fingerprint(self) -> str:
        """Delegate identity to the underlying listing so keys stay consistent."""
        return self.listing.fingerprint

    @model_validator(mode="after")
    def _check_price_agreement(self) -> Self:
        """Guard against a valuation being attached to the wrong listing.

        The valuation carries its own copy of the asking price for storage
        denormalization. If the two ever disagree, something upstream has
        mismatched records, and every downstream margin is wrong.

        Raises:
            ValueError: If the valuation's asking price or currency does not
                match the listing it is attached to.
        """
        if self.valuation.asking_price != self.listing.asking_price:
            raise ValueError(
                f"Valuation asking price {self.valuation.asking_price} does not match "
                f"listing {self.listing.fingerprint} price {self.listing.asking_price}."
            )
        if self.valuation.currency != self.listing.currency:
            raise ValueError(
                f"Currency mismatch on {self.listing.fingerprint}: "
                f"listing {self.listing.currency} vs valuation {self.valuation.currency}."
            )
        return self

    def meets_threshold(
        self,
        *,
        min_margin_pct: Decimal,
        min_confidence: float,
        min_absolute_margin: Decimal,
    ) -> bool:
        """Whether this listing clears the buy bar.

        Thresholds are arguments rather than constants because they are a
        business decision that belongs in configuration, not in the type system.
        A reseller with £200 of working capital and one with £20,000 want very
        different bars, and both should be expressible without a code change.

        The three conditions are ANDed on purpose. Percentage margin alone
        promotes £3-profit trades that are not worth the postage; absolute
        margin alone promotes low-return trades on expensive stock; and either
        without a confidence floor promotes the model's own hallucinations.

        Args:
            min_margin_pct: Minimum return on capital, as a fraction (``0.30``).
            min_confidence: Minimum valuation confidence, ``0.0``–``1.0``.
            min_absolute_margin: Minimum profit in currency units.

        Returns:
            ``True`` if the listing should surface as an opportunity.
        """
        return (
            self.assessment.is_on_trend
            and self.valuation.margin_pct >= min_margin_pct
            and self.valuation.absolute_margin >= min_absolute_margin
            and self.valuation.valuation_confidence >= min_confidence
        )

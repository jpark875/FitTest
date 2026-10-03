"""Typed contracts shared by every layer of the engine.

Nothing here imports from the rest of the package, which is what keeps the
layers independently replaceable. Money is always Decimal, never float.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Annotated, Any, Self

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

#: Cap on images sent to the vision model. Beyond ~8 the marginal signal is
#: negligible while per-listing cost keeps rising.
MAX_IMAGES_PER_LISTING = 8

#: First numeric run in a price string: "£1,299.00", "18,50 EUR", "$24".
_PRICE_PATTERN = re.compile(r"(\d[\d,\s.]*)")

_CURRENCY_SYMBOLS: dict[str, str] = {"£": "GBP", "$": "USD", "€": "EUR", "¥": "JPY"}

#: Codes sellers write out in full. Word-bounded so a brand name cannot match.
_CURRENCY_CODE_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(set(_CURRENCY_SYMBOLS.values()))) + r")\b", re.IGNORECASE
)

_CENTS = Decimal("0.01")

#: A non-negative monetary amount.
Money = Annotated[Decimal, Field(ge=Decimal(0), max_digits=10, decimal_places=2)]

#: A model-reported probability. Used for calibration, never for hard gating.
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class MalformedListingError(ValueError):
    """Raised when scraped data cannot be coerced into a RawListing.

    Ingestion catches this per listing and counts a skip rather than aborting
    the batch.
    """


class Platform(StrEnum):
    """A marketplace the engine can ingest from.

    FIXTURE is a first-class member, not a test artifact: it identifies
    listings replayed from recorded payloads.
    """

    DEPOP = "depop"
    THREDUP = "thredup"
    FIXTURE = "fixture"


class TrendTag(StrEnum):
    """A micro-trend the engine tracks.

    A closed vocabulary on purpose. Free-text tags produce an unqueryable long
    tail ("y2k", "Y2K vibes", "2000s") and make trend-level aggregation
    impossible.
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

    Platforms use incompatible vocabularies. Normalizing at the boundary means
    one condition discount curve instead of one per platform.
    """

    NEW_WITH_TAGS = "new_with_tags"
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"


def parse_price(raw: str, default_currency: str = "GBP") -> tuple[Decimal, str]:
    """Coerce a seller-authored price string into an amount and a currency.

    Accepts "£24", "24.00 GBP", "$1,299", "18,50", "£20 ono". Rejects "FREE"
    and "offers" rather than defaulting to zero, which would manufacture an
    infinite-margin opportunity out of a listing that has no price.

    Args:
        raw: The price text as scraped.
        default_currency: ISO 4217 code assumed when nothing is marked.

    Returns:
        An (amount, currency_code) pair, quantized to two decimals.

    Raises:
        MalformedListingError: If no parseable amount is present.
    """
    if not raw or not raw.strip():
        raise MalformedListingError("Price field was empty.")

    # An explicit ISO code outranks a symbol, which outranks the default.
    # Reading only symbols files "34,50 EUR" as GBP: a plausible number in the
    # wrong currency, which nothing downstream can detect.
    code_match = _CURRENCY_CODE_PATTERN.search(raw)
    if code_match is not None:
        currency = code_match.group(1).upper()
    else:
        currency = next(
            (code for symbol, code in _CURRENCY_SYMBOLS.items() if symbol in raw),
            default_currency,
        )

    match = _PRICE_PATTERN.search(raw)
    if match is None:
        raise MalformedListingError(f"No numeric value found in price {raw!r}.")

    candidate = match.group(1).strip().rstrip(".,").replace(" ", "")

    # A final separator followed by exactly two digits is a decimal point;
    # otherwise every separator is a thousands mark.
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

    Unknown vocabulary degrades to UNKNOWN rather than raising: it makes a
    listing less confidently valued, not unusable. The valuation layer applies
    its most conservative discount, so the failure mode is a missed buy.

    Args:
        raw: The platform's condition label, or None.

    Returns:
        The matching ItemCondition, defaulting to UNKNOWN.
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


class RawListing(BaseModel):
    """A marketplace listing as captured at a point in time.

    Frozen, because a capture is a historical fact: an edited price deserves a
    second row, and price movement over time is itself a signal. extra="forbid"
    so a scraper emitting an unknown field fails in CI rather than silently
    discarding data.
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
        """Normalize the code so "gbp" and "GBP" do not fragment joins."""
        return value.upper()

    @field_validator("image_urls", mode="before")
    @classmethod
    def _cap_images(cls, value: Any) -> Any:
        """Trim to MAX_IMAGES_PER_LISTING to bound per-listing vision cost.

        Runs before validation, not after: the field's own max_length would
        otherwise reject an eleven-photo listing outright instead of trimming
        it. Extra photographs are surplus signal, not a malformed record.
        """
        if isinstance(value, list):
            return value[:MAX_IMAGES_PER_LISTING]
        return value

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fingerprint(self) -> str:
        """Natural key for upserts, scoped by platform.

        Listing IDs are only unique within a marketplace, and two platforms
        will eventually collide on a bare integer.
        """
        return f"{self.platform.value}:{self.external_id}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_analyzable(self) -> bool:
        """Whether this listing carries enough signal to be worth an API call.

        Vision enrichment is the dominant per-listing cost, so this gate runs
        before the model is invoked.
        """
        return bool(self.image_urls) and self.asking_price > 0


class TrendAssessment(BaseModel):
    """The structured verdict returned by the vision layer for one listing.

    Doubles as the schema handed to the model, which is why every field is
    primitive and closed-vocabulary. model_id and prompt_version are recorded
    on every row so a shift in quality can be attributed rather than guessed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trend_tags: list[TrendTag] = Field(min_length=1, max_length=3)
    inferred_brand: str | None = Field(
        default=None, description="Brand read from the garment, not the seller's text."
    )
    inferred_category: str = Field(min_length=1, description="e.g. 'outerwear', 'denim'.")
    inferred_decade: str | None = Field(default=None, description="e.g. '1990s'.")
    style_confidence: Confidence
    estimated_retail_value: Money | None = Field(
        default=None, description="Comparable market price, before fees."
    )
    reasoning: str = Field(
        max_length=1_000, description="One-paragraph justification, shown in the dashboard."
    )
    model_id: str = Field(description="Exact model that produced this.")
    prompt_version: str = Field(description="Version of the prompt template used.")
    assessed_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def primary_trend(self) -> TrendTag:
        """The highest-signal trend. The prompt orders tags by confidence."""
        return self.trend_tags[0]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_on_trend(self) -> bool:
        """Whether the model placed this garment in any tracked micro-trend."""
        return self.primary_trend is not TrendTag.NONE


class Valuation(BaseModel):
    """What a listing is worth, and how confident we are.

    This model owns arithmetic; mta.processing.valuation owns judgment (which
    comps count, where the buy threshold sits). Keeping the arithmetic here
    means the dashboard and the pipeline cannot disagree about a margin.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    asking_price: Money
    currency: str = Field(min_length=3, max_length=3)
    estimated_resale_value: Money = Field(
        description="Expected achievable sale price, condition-adjusted."
    )
    comparable_count: int = Field(
        ge=0, description="Comps behind the estimate. Zero means model-only."
    )
    median_comp_price: Money | None = None
    platform_fee_rate: Decimal = Field(
        default=Decimal("0.10"),
        ge=Decimal(0),
        le=Decimal(1),
        description="Marketplace commission on resale.",
    )
    shipping_cost: Money = Field(
        default=Decimal("0.00"), description="Estimated cost to receive and re-ship."
    )
    valuation_confidence: Confidence = Field(
        description="Model confidence combined with comp depth."
    )
    arbitrage_score: float = Field(
        ge=0.0,
        description=(
            "Ranking scalar from mta.processing.valuation. Stored rather than derived "
            "so historical rows survive a change to the formula."
        ),
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def net_proceeds(self) -> Decimal:
        """Cash actually received, after commission and postage.

        Reasoning against gross resale is the most common way a flipping model
        overstates itself.
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
        """Return on capital as a fraction, so 0.35 is +35%.

        This is what makes an £8 profit on a £10 item comparable to a £30
        profit on a £120 one.
        """
        if self.asking_price <= 0:
            return Decimal("0.00")
        return (self.absolute_margin / self.asking_price).quantize(Decimal("0.0001"))

    @model_validator(mode="after")
    def _check_currency_coherence(self) -> Self:
        """Reject a malformed currency code.

        Cross-currency comparison is a real requirement but needs an explicit
        FX step. Subtracting dollars from pounds produces a confident, wrong
        and expensive recommendation.

        Raises:
            ValueError: If the code is not well-formed.
        """
        if not self.currency.isalpha():
            raise ValueError(f"Currency {self.currency!r} is not a valid ISO 4217 code.")
        return self


class EnrichedListing(BaseModel):
    """A listing that has been captured, assessed and valued.

    A composition rather than a flat model so each stage's provenance stays
    intact: which model priced this, and what the seller originally wrote.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    listing: RawListing
    assessment: TrendAssessment
    valuation: Valuation
    enriched_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fingerprint(self) -> str:
        """Delegate identity to the listing so keys stay consistent."""
        return self.listing.fingerprint

    @model_validator(mode="after")
    def _check_price_agreement(self) -> Self:
        """Guard against a valuation attached to the wrong listing.

        The valuation carries its own copy of the price for storage
        denormalization. If the two disagree, records have been mismatched
        upstream and every downstream margin is wrong.

        Raises:
            ValueError: If the price or currency does not match the listing.
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

        Thresholds are arguments because they are a business decision, not a
        property of the type. The conditions are ANDed on purpose: percentage
        margin alone promotes £3 trades not worth the postage, absolute margin
        alone promotes low-return trades on expensive stock, and either without
        a confidence floor promotes the model's hallucinations.

        Args:
            min_margin_pct: Minimum return on capital, as a fraction.
            min_confidence: Minimum valuation confidence, 0.0-1.0.
            min_absolute_margin: Minimum profit in currency units.

        Returns:
            True if the listing should surface as an opportunity.
        """
        return (
            self.assessment.is_on_trend
            and self.valuation.margin_pct >= min_margin_pct
            and self.valuation.absolute_margin >= min_absolute_margin
            and self.valuation.valuation_confidence >= min_confidence
        )

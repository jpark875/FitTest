"""Trend classification: an offline keyword classifier and a vision-model classifier."""

from __future__ import annotations

import asyncio
import logging
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from mta.models.listing import ItemCondition, RawListing, TrendAssessment, TrendTag
from mta.processing.normalize import find_brand_in_text, normalize_brand

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent / "prompts"

KEYWORD_MODEL_ID = "keyword-heuristic"

#: Phrases that put a garment in a trend. Matched against title and description.
_TREND_KEYWORDS: dict[TrendTag, tuple[str, ...]] = {
    TrendTag.Y2K: ("y2k", "low rise", "tie hem", "ed hardy", "baby tee", "2000s", "rhinestone"),
    TrendTag.GRUNGE: ("grunge", "flannel", "thrashed", "distressed", "oversized", "faded black"),
    TrendTag.RETRO_SKATER: ("skate", "baggy", "stussy", "thrasher", "dickies", "jnco"),
    TrendTag.GORPCORE: ("gore-tex", "goretex", "gorpcore", "fleece", "shell", "synchilla", "dwr"),
    TrendTag.WORKWEAR: ("carhartt", "duck canvas", "chore coat", "work trousers", "detroit"),
    TrendTag.ARCHIVE_TECHWEAR: ("archive", "techwear", "snopants", "maharishi", "taped seams"),
    TrendTag.VINTAGE_BAND_TEE: ("tour tee", "band tee", "single stitch", "nirvana", "metallica"),
    TrendTag.COQUETTE: ("coquette", "lace trim", "bow", "ribbon", "pointelle"),
}

#: Typical resale value in GBP for a good-condition item, by brand. Deliberately coarse.
_BRAND_BASELINE: dict[str, Decimal] = {
    "Arc'teryx": Decimal("210"),
    "Maharishi": Decimal("320"),
    "Patagonia": Decimal("85"),
    "Carhartt": Decimal("85"),
    "Stussy": Decimal("55"),
    "Ed Hardy": Decimal("60"),
    "Dickies": Decimal("35"),
    "Levi's": Decimal("40"),
}

#: Baseline by trend when the brand is unknown.
_TREND_BASELINE: dict[TrendTag, Decimal] = {
    TrendTag.VINTAGE_BAND_TEE: Decimal("220"),
    TrendTag.ARCHIVE_TECHWEAR: Decimal("150"),
    TrendTag.GORPCORE: Decimal("80"),
    TrendTag.WORKWEAR: Decimal("70"),
    TrendTag.Y2K: Decimal("45"),
    TrendTag.RETRO_SKATER: Decimal("40"),
    TrendTag.GRUNGE: Decimal("35"),
    TrendTag.COQUETTE: Decimal("35"),
}

_CATEGORY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("outerwear", ("jacket", "coat", "shell", "parka", "overshirt", "fleece")),
    ("denim", ("jeans", "denim")),
    ("trousers", ("trousers", "cargo", "pants", "snopants")),
    ("hoodie", ("hoodie", "zip-up", "sweatshirt")),
    ("knitwear", ("cardigan", "jumper", "sweater", "knit")),
    ("tops", ("tee", "t-shirt", "shirt", "top")),
)

_RETAIL_CONDITION_UPLIFT = {
    ItemCondition.NEW_WITH_TAGS: Decimal("1.15"),
    ItemCondition.EXCELLENT: Decimal("1.05"),
    ItemCondition.GOOD: Decimal("1.00"),
    ItemCondition.FAIR: Decimal("0.80"),
    ItemCondition.POOR: Decimal("0.55"),
    ItemCondition.UNKNOWN: Decimal("0.90"),
}


class TrendClassifier(Protocol):
    """Anything that can turn a listing into a TrendAssessment."""

    async def classify(self, listing: RawListing) -> TrendAssessment:
        """Assess one listing."""
        ...


def _listing_text(listing: RawListing) -> str:
    return f"{listing.title} {listing.description or ''}".lower()


class KeywordClassifier:
    """Offline fallback that reads the seller's text. Cheap, deterministic, and weaker.

    It cannot see the garment, so confidence is capped well below what a vision model
    can report.
    """

    prompt_version = "kw1"
    max_confidence = 0.85

    async def classify(self, listing: RawListing) -> TrendAssessment:
        """Score each trend by keyword hits and pick the strongest."""
        text = _listing_text(listing)
        hits = {
            trend: sum(1 for word in words if word in text)
            for trend, words in _TREND_KEYWORDS.items()
        }
        ranked = sorted(
            (item for item in hits.items() if item[1] > 0), key=lambda item: (-item[1], item[0])
        )
        tags = [trend for trend, _ in ranked[:3]] or [TrendTag.NONE]

        brand = normalize_brand(listing.brand_raw) or find_brand_in_text(text)
        category = next(
            (name for name, words in _CATEGORY_KEYWORDS if any(w in text for w in words)),
            "other",
        )
        confidence = (
            0.15
            if tags == [TrendTag.NONE]
            else min(self.max_confidence, 0.5 + 0.1 * ranked[0][1] + (0.05 if brand else 0.0))
        )

        reasoning = (
            f"Keyword match on {', '.join(t.value for t in tags)} from the listing text."
            if tags != [TrendTag.NONE]
            else "No tracked trend keywords in the listing text."
        )
        return TrendAssessment(
            trend_tags=tags,
            inferred_brand=brand,
            inferred_category=category,
            inferred_decade=_decade(text),
            style_confidence=round(confidence, 2),
            estimated_retail_value=self._estimate(brand, tags[0], listing.condition),
            reasoning=reasoning,
            model_id=KEYWORD_MODEL_ID,
            prompt_version=self.prompt_version,
        )

    @staticmethod
    def _estimate(brand: str | None, trend: TrendTag, condition: ItemCondition) -> Decimal | None:
        base = _BRAND_BASELINE.get(brand or "") or _TREND_BASELINE.get(trend)
        if base is None:
            return None
        return (base * _RETAIL_CONDITION_UPLIFT[condition]).quantize(Decimal("0.01"))


_DECADE_PATTERNS = (
    (re.compile(r"\b(?:19)?80s\b"), "1980s"),
    (re.compile(r"\b(?:19)?90s\b"), "1990s"),
    (re.compile(r"\b(?:20)?00s\b|\by2k\b"), "2000s"),
)
_YEAR = re.compile(r"\b(19[89]\d|200\d)\b")


def _decade(text: str) -> str | None:
    for pattern, decade in _DECADE_PATTERNS:
        if pattern.search(text):
            return decade
    year = _YEAR.search(text)
    return f"{int(year.group(1)) // 10 * 10}s" if year else None


class ModelVerdict(BaseModel):
    """The shape requested from the model; provenance fields are added locally."""

    trend_tags: list[TrendTag] = Field(min_length=1, max_length=3)
    inferred_brand: str | None = None
    inferred_category: str
    inferred_decade: str | None = None
    style_confidence: float
    estimated_retail_value: float | None = None
    reasoning: str


def load_prompt(version: str) -> str:
    """Read a prompt template by version name."""
    path = PROMPT_DIR / f"{version}.md"
    if not path.is_file():
        raise FileNotFoundError(f"no prompt template {path}")
    return path.read_text(encoding="utf-8")


class AnthropicClassifier:
    """Reads the listing photographs with a vision model and returns structured output."""

    def __init__(
        self,
        api_key: str | None,
        *,
        model: str,
        prompt_version: str,
        timeout_seconds: float = 120.0,
        max_tokens: int = 1500,
        client: Any = None,
    ) -> None:
        """Build the classifier; the SDK client is created lazily unless one is injected."""
        self.model = model
        self.prompt_version = prompt_version
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._client = client
        self._prompt = load_prompt(prompt_version)

    @property
    def client(self) -> Any:
        """The SDK client, created on first use."""
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=self._api_key, timeout=self._timeout)
        return self._client

    def _content(self, listing: RawListing) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = [
            {"type": "image", "source": {"type": "url", "url": str(url)}}
            for url in listing.image_urls
        ]
        details = (
            f"Title: {listing.title}\n"
            f"Seller description: {listing.description or 'none'}\n"
            f"Condition: {listing.condition.value}\n"
            f"Asking price: {listing.asking_price} {listing.currency}\n"
            f"Seller-declared brand: {listing.brand_raw or 'none'}"
        )
        blocks.append({"type": "text", "text": f"{self._prompt}\n\n{details}"})
        return blocks

    async def classify(self, listing: RawListing) -> TrendAssessment:
        """Send the photographs and listing details, then validate the verdict."""
        response = await self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": self._content(listing)}],
            output_format=ModelVerdict,
        )
        verdict: ModelVerdict | None = response.parsed_output
        if verdict is None:
            raise ValueError(f"model returned no parsable verdict (stop {response.stop_reason})")

        value = (
            Decimal(str(round(verdict.estimated_retail_value, 2)))
            if verdict.estimated_retail_value is not None and verdict.estimated_retail_value >= 0
            else None
        )
        return TrendAssessment(
            trend_tags=_dedupe(verdict.trend_tags),
            inferred_brand=normalize_brand(verdict.inferred_brand),
            inferred_category=verdict.inferred_category.strip().lower() or "other",
            inferred_decade=verdict.inferred_decade,
            style_confidence=min(max(verdict.style_confidence, 0.0), 1.0),
            estimated_retail_value=value,
            reasoning=verdict.reasoning[:1000],
            model_id=self.model,
            prompt_version=self.prompt_version,
        )


def _dedupe(tags: list[TrendTag]) -> list[TrendTag]:
    """Drop repeats and let `none` stand only when nothing else was chosen."""
    ordered = list(dict.fromkeys(tags))
    real: list[TrendTag] = [tag for tag in ordered if tag is not TrendTag.NONE]
    return real or [TrendTag.NONE]


async def classify_all(
    classifier: TrendClassifier, listings: list[RawListing], concurrency: int
) -> tuple[dict[str, TrendAssessment], dict[str, str]]:
    """Classify listings with bounded concurrency; failures are returned, not raised."""
    semaphore = asyncio.Semaphore(concurrency)
    results: dict[str, TrendAssessment] = {}
    failures: dict[str, str] = {}

    async def one(listing: RawListing) -> None:
        async with semaphore:
            try:
                results[listing.fingerprint] = await classifier.classify(listing)
            except Exception as exc:  # noqa: BLE001 - one bad listing must not end the batch
                logger.warning("classification failed for %s: %s", listing.fingerprint, exc)
                failures[listing.fingerprint] = f"{type(exc).__name__}: {exc}"

    await asyncio.gather(*(one(listing) for listing in listings))
    return results, failures

"""Tests for the keyword and vision classifiers."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from mta.models.listing import ItemCondition, Platform, RawListing, TrendTag
from mta.processing.classifier import (
    KEYWORD_MODEL_ID,
    AnthropicClassifier,
    KeywordClassifier,
    ModelVerdict,
    classify_all,
    load_prompt,
)


def listing(title: str, **kwargs: Any) -> RawListing:
    defaults: dict[str, Any] = {
        "platform": Platform.FIXTURE,
        "external_id": title[:12],
        "url": "https://x.invalid/1",
        "title": title,
        "asking_price": Decimal("20.00"),
        "currency": "GBP",
        "image_urls": ["https://x.invalid/1.jpg"],
    }
    return RawListing(**{**defaults, **kwargs})


class TestKeywordClassifier:
    async def test_reads_trend_brand_and_category_from_text(self) -> None:
        item = listing("Vintage 90s Carhartt Detroit jacket duck canvas", brand_raw="Carhartt")
        result = await KeywordClassifier().classify(item)
        assert result.primary_trend is TrendTag.WORKWEAR
        assert result.inferred_brand == "Carhartt"
        assert result.inferred_category == "outerwear"
        assert result.inferred_decade == "1990s"
        assert result.model_id == KEYWORD_MODEL_ID

    async def test_no_keywords_means_no_trend_and_low_confidence(self) -> None:
        result = await KeywordClassifier().classify(listing("Plain navy socks"))
        assert result.trend_tags == [TrendTag.NONE]
        assert not result.is_on_trend
        assert result.style_confidence < 0.2

    async def test_confidence_is_capped_below_a_vision_model(self) -> None:
        text = "y2k low rise tie hem baby tee rhinestone ed hardy 2000s"
        result = await KeywordClassifier().classify(listing(text))
        assert result.style_confidence <= KeywordClassifier.max_confidence

    async def test_brand_found_in_title_when_seller_left_it_blank(self) -> None:
        result = await KeywordClassifier().classify(listing("Stussy hoodie faded"))
        assert result.inferred_brand == "Stussy"

    async def test_estimate_follows_condition(self) -> None:
        good = await KeywordClassifier().classify(
            listing("Carhartt chore coat", condition=ItemCondition.GOOD)
        )
        poor = await KeywordClassifier().classify(
            listing("Carhartt chore coat", condition=ItemCondition.POOR)
        )
        assert good.estimated_retail_value is not None
        assert poor.estimated_retail_value is not None
        assert poor.estimated_retail_value < good.estimated_retail_value

    async def test_unknown_brand_and_trend_has_no_estimate(self) -> None:
        result = await KeywordClassifier().classify(listing("Plain navy socks"))
        assert result.estimated_retail_value is None


class FakeMessages:
    def __init__(self, verdict: ModelVerdict | None) -> None:
        self.verdict = verdict
        self.kwargs: dict[str, Any] = {}

    async def parse(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return SimpleNamespace(parsed_output=self.verdict, stop_reason="end_turn")


def make_anthropic(verdict: ModelVerdict | None) -> tuple[AnthropicClassifier, FakeMessages]:
    messages = FakeMessages(verdict)
    classifier = AnthropicClassifier(
        None,
        model="claude-opus-5-5",
        prompt_version="v1",
        client=SimpleNamespace(messages=messages),
    )
    return classifier, messages


class TestAnthropicClassifier:
    async def test_sends_every_image_then_the_prompt(self) -> None:
        verdict = ModelVerdict(
            trend_tags=[TrendTag.WORKWEAR],
            inferred_category="Outerwear",
            style_confidence=0.9,
            reasoning="Duck canvas.",
        )
        classifier, messages = make_anthropic(verdict)
        item = listing(
            "Carhartt jacket",
            image_urls=["https://x.invalid/1.jpg", "https://x.invalid/2.jpg"],
        )

        result = await classifier.classify(item)

        content = messages.kwargs["messages"][0]["content"]
        assert [block["type"] for block in content] == ["image", "image", "text"]
        assert content[0]["source"] == {"type": "url", "url": "https://x.invalid/1.jpg"}
        assert messages.kwargs["output_format"] is ModelVerdict
        assert messages.kwargs["model"] == "claude-opus-5-5"
        assert result.inferred_category == "outerwear"
        assert result.model_id == "claude-opus-5-5"
        assert result.prompt_version == "v1"

    async def test_verdict_is_cleaned_before_it_becomes_an_assessment(self) -> None:
        verdict = ModelVerdict(
            trend_tags=[TrendTag.NONE, TrendTag.Y2K, TrendTag.Y2K],
            inferred_brand="levis",
            inferred_category="  ",
            style_confidence=1.7,
            estimated_retail_value=-3.0,
            reasoning="x" * 2000,
        )
        classifier, _ = make_anthropic(verdict)

        result = await classifier.classify(listing("Baggy jeans"))

        assert result.trend_tags == [TrendTag.Y2K]
        assert result.inferred_brand == "Levi's"
        assert result.inferred_category == "other"
        assert result.style_confidence == 1.0
        assert result.estimated_retail_value is None
        assert len(result.reasoning) == 1000

    async def test_unparsable_response_raises(self) -> None:
        classifier, _ = make_anthropic(None)
        with pytest.raises(ValueError, match="no parsable verdict"):
            await classifier.classify(listing("Baggy jeans"))

    def test_prompt_template_loads_and_missing_one_fails(self) -> None:
        assert "micro-trend" in load_prompt("v1")
        with pytest.raises(FileNotFoundError):
            load_prompt("v999")


class TestClassifyAll:
    async def test_failures_are_collected_not_raised(self) -> None:
        class Flaky:
            async def classify(self, item: RawListing) -> Any:
                if "bad" in item.title:
                    raise RuntimeError("boom")
                return await KeywordClassifier().classify(item)

        items = [listing("good carhartt coat"), listing("bad one")]
        results, failures = await classify_all(Flaky(), items, concurrency=2)

        assert list(results) == [items[0].fingerprint]
        assert "RuntimeError: boom" in failures[items[1].fingerprint]

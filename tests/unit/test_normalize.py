"""Tests for brand and size normalization."""

from __future__ import annotations

import pytest

from mta.processing.normalize import find_brand_in_text, normalize_brand, normalize_size


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("carhartt", "Carhartt"),
        ("Carhartt WIP", "Carhartt"),
        ("Levis", "Levi's"),
        ("arcteryx", "Arc'teryx"),
        ("Pataganía", "Pataganía"),
        ("  Stussy  ", "Stussy"),
        ("Obscure Label", "Obscure Label"),
    ],
)
def test_brand(raw: str, expected: str) -> None:
    assert normalize_brand(raw) == expected


def test_typo_within_cutoff_maps_to_canonical() -> None:
    assert normalize_brand("Patagonai") == "Patagonia"


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_empty_brand_is_none(raw: str | None) -> None:
    assert normalize_brand(raw) is None


def test_brand_found_in_text_requires_word_boundaries() -> None:
    assert find_brand_in_text("vintage carhartt detroit jacket") == "Carhartt"
    assert find_brand_in_text("the north face nuptse") == "The North Face"
    assert find_brand_in_text("nikeish thing") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("L", "L"),
        ("medium", "M"),
        ("Extra Large", "XL"),
        ("UK 12", "12"),
        ("W32", "32"),
        ("one size", None),
        (None, None),
        ("", None),
    ],
)
def test_size(raw: str | None, expected: str | None) -> None:
    assert normalize_size(raw) == expected

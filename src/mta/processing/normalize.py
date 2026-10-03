"""Cleaning of seller-authored brand and size strings."""

from __future__ import annotations

import re

from rapidfuzz import fuzz, process

#: Canonical brand names. Aliases map the spellings sellers actually use.
_BRANDS: dict[str, tuple[str, ...]] = {
    "Carhartt": ("carhartt wip", "carhartt active", "carhart"),
    "Arc'teryx": ("arcteryx", "arc teryx", "arc'teryx"),
    "Patagonia": (),
    "Stussy": ("stüssy",),
    "Dickies": (),
    "Levi's": ("levis", "levi"),
    "Maharishi": (),
    "Ed Hardy": ("edhardy",),
    "The North Face": ("north face", "tnf"),
    "Nike": (),
    "Adidas": (),
    "Supreme": (),
    "Diesel": (),
    "Evisu": (),
    "Fila": (),
    "Champion": (),
}

_BRAND_LOOKUP: dict[str, str] = {}
for _canonical, _aliases in _BRANDS.items():
    _BRAND_LOOKUP[_canonical.lower()] = _canonical
    for _alias in _aliases:
        _BRAND_LOOKUP[_alias.lower()] = _canonical

_FUZZY_CUTOFF = 88

_ALPHA_SIZES = {
    "xxs": "XXS",
    "xs": "XS",
    "s": "S",
    "small": "S",
    "m": "M",
    "medium": "M",
    "l": "L",
    "large": "L",
    "xl": "XL",
    "x-large": "XL",
    "extra large": "XL",
    "xxl": "XXL",
    "2xl": "XXL",
    "xxxl": "3XL",
    "3xl": "3XL",
}

_NUMERIC_SIZE = re.compile(r"\b(?:uk|eu|us|w)?\s*(\d{1,2})\b", re.IGNORECASE)


def normalize_brand(raw: str | None) -> str | None:
    """Return the canonical brand for a seller's spelling, or the cleaned input if unknown."""
    if not raw or not raw.strip():
        return None
    cleaned = " ".join(raw.split())
    exact = _BRAND_LOOKUP.get(cleaned.lower())
    if exact:
        return exact
    match = process.extractOne(
        cleaned.lower(), _BRAND_LOOKUP.keys(), scorer=fuzz.ratio, score_cutoff=_FUZZY_CUTOFF
    )
    if match:
        return _BRAND_LOOKUP[match[0]]
    return cleaned


def find_brand_in_text(text: str) -> str | None:
    """Find a known brand mentioned in free text, longest name first."""
    lowered = text.lower()
    for name in sorted(_BRAND_LOOKUP, key=len, reverse=True):
        if re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", lowered):
            return _BRAND_LOOKUP[name]
    return None


def normalize_size(raw: str | None) -> str | None:
    """Map free-text sizes onto XS-3XL or a bare number; None if nothing recognisable."""
    if not raw or not raw.strip():
        return None
    text = raw.strip().lower()
    if text in _ALPHA_SIZES:
        return _ALPHA_SIZES[text]
    match = _NUMERIC_SIZE.search(text)
    if match:
        return match.group(1)
    return None

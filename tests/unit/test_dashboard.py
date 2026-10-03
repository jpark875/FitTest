"""Tests for the dashboard's data layer and a smoke run of the Streamlit script."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from mta.config import get_settings
from mta.dashboard.data import COLUMNS, load_frame, to_frame
from mta.models.listing import RawListing, TrendAssessment, TrendTag, Valuation
from mta.storage import Repository

APP = Path(__file__).resolve().parents[2] / "src" / "mta" / "dashboard" / "app.py"


@pytest.fixture
def repository(
    tmp_path: Path,
    raw_listing: RawListing,
    assessment: TrendAssessment,
    valuation: Valuation,
) -> Repository:
    repo = Repository(f"sqlite:///{tmp_path / 'mta.db'}")
    repo.create_schema()
    repo.upsert_listing(raw_listing)
    repo.upsert_assessment(raw_listing.fingerprint, assessment)
    repo.upsert_valuation(raw_listing.fingerprint, valuation)
    return repo


def test_frame_has_one_row_per_opportunity(repository: Repository) -> None:
    frame = to_frame(repository.opportunities())
    assert list(frame.columns) == COLUMNS
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["trend"] == "workwear"
    assert row["price"] == 25.0
    assert row["return_pct"] == pytest.approx(float(Decimal("2.06")) * 100)


def test_empty_frame_keeps_its_columns(tmp_path: Path) -> None:
    repo = Repository(f"sqlite:///{tmp_path / 'empty.db'}")
    repo.create_schema()
    assert list(to_frame(repo.opportunities()).columns) == COLUMNS


def test_load_frame_applies_floors_and_drops_off_trend(repository: Repository) -> None:
    kwargs = {"min_margin_pct": Decimal("0.3"), "min_absolute_margin": Decimal("10")}
    assert len(load_frame(repository, min_confidence=0.6, **kwargs)) == 1
    assert load_frame(repository, min_confidence=0.95, **kwargs).empty
    assert load_frame(repository, min_confidence=0.6, trends=[TrendTag.Y2K], **kwargs).empty


def test_app_renders_for_an_empty_and_a_populated_warehouse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repository: Repository
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MTA_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("MTA_DATABASE_URL", f"sqlite:///{tmp_path / 'empty.db'}")
    get_settings.cache_clear()
    empty = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not empty.exception
    assert "empty" in empty.info[0].value

    monkeypatch.setenv("MTA_DATABASE_URL", repository.engine.url.render_as_string())
    get_settings.cache_clear()
    full = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not full.exception
    assert full.metric[0].value == "1"
    get_settings.cache_clear()

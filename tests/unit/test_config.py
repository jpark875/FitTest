"""Tests for the configuration surface.

The politeness ceiling and the credential guard get the most attention: one is
a policy commitment, the other is the difference between a readable startup
error and a library stack trace.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from mta.config import POLITE_RPM_CEILING, ConfigurationError, Settings, get_settings
from mta.models.listing import Platform


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run every test against a clean environment, not the developer's own."""
    for name in list(os.environ):
        if name.startswith("MTA_") or name == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(name, raising=False)
    yield


def build(**overrides: object) -> Settings:
    # _env_file=None so a local .env cannot change the outcome.
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


class TestDefaults:
    def test_defaults_to_the_offline_source(self) -> None:
        assert build().ingestion_source is Platform.FIXTURE

    def test_defaults_are_conservative(self) -> None:
        settings = build()
        assert settings.requests_per_minute <= POLITE_RPM_CEILING
        assert settings.max_pages_per_run <= 100

    def test_is_frozen(self) -> None:
        with pytest.raises(ValidationError):
            build().requests_per_minute = 5  # type: ignore[misc]


class TestPolitenessCeiling:
    def test_a_live_source_above_the_ceiling_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="exceeds the"):
            build(ingestion_source=Platform.DEPOP, requests_per_minute=60)

    def test_a_live_source_at_the_ceiling_is_allowed(self) -> None:
        settings = build(ingestion_source=Platform.DEPOP, requests_per_minute=POLITE_RPM_CEILING)
        assert settings.requests_per_minute == POLITE_RPM_CEILING

    def test_the_offline_source_is_exempt(self) -> None:
        assert build(ingestion_source=Platform.FIXTURE, requests_per_minute=120)

    def test_the_error_names_the_way_out(self) -> None:
        with pytest.raises(ValidationError, match="MTA_INGESTION_SOURCE=fixture"):
            build(ingestion_source=Platform.THREDUP, requests_per_minute=90)


class TestDatabaseUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "sqlite:///data/warehouse/mta.db",
            "sqlite+aiosqlite:///data/mta.db",
            "postgresql://user:pw@localhost/mta",
            "postgresql+psycopg://user:pw@localhost/mta",
        ],
    )
    def test_accepts_supported_backends(self, url: str) -> None:
        assert build(database_url=url).database_url == url

    @pytest.mark.parametrize("url", ["mysql://localhost/mta", "redis://localhost", "mta.db"])
    def test_rejects_unsupported_backends(self, url: str) -> None:
        with pytest.raises(ValidationError, match="Unsupported database scheme"):
            build(database_url=url)


class TestCredentials:
    def test_a_missing_key_fails_with_an_actionable_message(self) -> None:
        with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY is not set"):
            build().require_api_key()

    def test_the_key_is_returned_for_the_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        assert build().require_api_key() == "test-key"

    def test_the_key_never_appears_in_a_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-value")
        settings = build()
        assert "secret-value" not in repr(settings)
        assert "secret-value" not in str(settings.model_dump())

    def test_read_only_paths_work_without_a_key(self) -> None:
        assert build().database_url


class TestPaths:
    def test_derived_paths_hang_off_the_project_root(self, tmp_path: Path) -> None:
        settings = build(project_root=tmp_path)
        assert settings.raw_dir == tmp_path / "data" / "raw"
        assert settings.warehouse_dir == tmp_path / "data" / "warehouse"
        assert settings.fixtures_dir == tmp_path / "tests" / "fixtures"

    def test_ensure_directories_is_idempotent(self, tmp_path: Path) -> None:
        settings = build(project_root=tmp_path)
        settings.ensure_directories()
        settings.ensure_directories()
        assert settings.raw_dir.is_dir()
        assert settings.warehouse_dir.is_dir()

    def test_an_uncreatable_directory_is_reported(self, tmp_path: Path) -> None:
        (tmp_path / "data").write_text("a file where a directory should be", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Could not create required directory"):
            build(project_root=tmp_path).ensure_directories()


class TestThresholds:
    def test_buy_thresholds_are_decimals(self) -> None:
        settings = build()
        assert isinstance(settings.min_margin_pct, Decimal)
        assert isinstance(settings.min_absolute_margin, Decimal)

    def test_a_fee_rate_above_one_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            build(default_platform_fee_rate=Decimal("1.5"))


class TestGetSettings:
    def test_is_cached(self) -> None:
        get_settings.cache_clear()
        try:
            assert get_settings() is get_settings()
        finally:
            get_settings.cache_clear()

    def test_invalid_configuration_is_reported_as_a_flat_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MTA_INGESTION_SOURCE", "depop")
        monkeypatch.setenv("MTA_REQUESTS_PER_MINUTE", "119")
        get_settings.cache_clear()
        try:
            with pytest.raises(ConfigurationError, match="Invalid configuration"):
                get_settings()
        finally:
            get_settings.cache_clear()

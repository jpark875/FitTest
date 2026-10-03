"""Tests for the command-line interface."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mta.config import get_settings
from mta.pipeline.cli import app

runner = CliRunner()


@pytest.fixture
def workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixtures_dir: Path
) -> Iterator[Path]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MTA_DATABASE_URL", f"sqlite:///{tmp_path / 'mta.db'}")
    monkeypatch.setenv("MTA_INGESTION_SOURCE", "fixture")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_run_then_top_and_stats(workspace: Path) -> None:
    result = runner.invoke(
        app, ["run", "--query", "alternative streetwear", "--classifier", "keyword"]
    )
    assert result.exit_code == 0, result.output
    assert "9 valued" in result.output

    top = runner.invoke(app, ["top", "--all", "--limit", "3"])
    assert top.exit_code == 0
    assert "All valued listings" in top.output

    stats = runner.invoke(app, ["stats"])
    assert stats.exit_code == 0
    assert "9 listings" in stats.output


def test_top_on_an_empty_warehouse_explains_itself(workspace: Path) -> None:
    result = runner.invoke(app, ["top"])
    assert result.exit_code == 0
    assert "mta run" in result.output


def test_live_source_without_authorisation_exits_nonzero(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MTA_INGESTION_SOURCE", "depop")
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 1
    assert "MTA_AUTHORISED_TO_COLLECT" in result.output


def test_invalid_configuration_exits_with_status_2(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MTA_DATABASE_URL", "mysql://nope")
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 2
    assert "Unsupported database scheme" in result.output

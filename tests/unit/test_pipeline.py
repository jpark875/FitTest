"""End-to-end tests of one pipeline pass over the recorded fixtures."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from mta.config import ConfigurationError, Settings
from mta.ingestion.base import SearchQuery
from mta.ingestion.fixtures import FixtureSource
from mta.models.listing import Platform, RawListing
from mta.pipeline.factory import build_classifier, build_source
from mta.pipeline.run import run_pipeline
from mta.processing.classifier import AnthropicClassifier, KeywordClassifier
from mta.storage import Repository

FEE = Decimal("0.10")
POSTAGE = Decimal("4.50")


@pytest.fixture
def repository(tmp_path: Path) -> Repository:
    repo = Repository(f"sqlite:///{tmp_path / 'mta.db'}")
    repo.create_schema()
    return repo


async def run(
    fixtures_dir: Path, repository: Repository, keywords: str, classifier: Any = None
) -> Any:
    return await run_pipeline(
        FixtureSource(fixtures_dir),
        classifier or KeywordClassifier(),
        repository,
        SearchQuery(keywords=keywords),
        platform_fee_rate=FEE,
        shipping_cost=POSTAGE,
    )


async def test_default_fixtures_flow_to_valued_rows(
    fixtures_dir: Path, repository: Repository
) -> None:
    report = await run(fixtures_dir, repository, "alternative streetwear")

    # 10 unique recordings; fx-1010 has no photographs and is not analysable.
    assert report.ingested == 10
    assert report.unanalyzable == 1
    assert report.classified == report.valued == 9
    assert repository.count_listings() == 9
    assert not report.failures


async def test_a_second_pass_is_idempotent(fixtures_dir: Path, repository: Repository) -> None:
    await run(fixtures_dir, repository, "alternative streetwear")
    again = await run(fixtures_dir, repository, "alternative streetwear")

    assert again.new == 0
    assert repository.count_listings() == 9
    assert len(repository.opportunities()) == 9


async def test_batch_peers_become_comparables(fixtures_dir: Path, repository: Repository) -> None:
    await run(fixtures_dir, repository, "vintage carhartt jacket")
    again = await run(fixtures_dir, repository, "vintage carhartt jacket")

    assert again.valued == 4
    assert any(o.valuation.comparable_count > 0 for o in repository.opportunities())


async def test_a_failing_classification_does_not_end_the_run(
    fixtures_dir: Path, repository: Repository
) -> None:
    class FailsOnArcteryx(KeywordClassifier):
        async def classify(self, listing: RawListing) -> Any:
            if "Arc'teryx" in listing.title:
                raise RuntimeError("model down")
            return await super().classify(listing)

    report = await run(fixtures_dir, repository, "alternative streetwear", FailsOnArcteryx())

    assert len(report.failures) == 1
    assert report.valued == 8
    assert "1 failed" in report.summary()


class TestFactory:
    def test_fixture_source_is_the_default(self, tmp_path: Path) -> None:
        source = build_source(Settings(_env_file=None, project_root=tmp_path))
        assert source.platform is Platform.FIXTURE

    def test_live_source_needs_explicit_authorisation(self, tmp_path: Path) -> None:
        settings = Settings(_env_file=None, project_root=tmp_path, ingestion_source="depop")
        with pytest.raises(ConfigurationError, match="MTA_AUTHORISED_TO_COLLECT"):
            build_source(settings)

    def test_authorised_live_source_is_built(self, tmp_path: Path) -> None:
        settings = Settings(
            _env_file=None,
            project_root=tmp_path,
            ingestion_source="thredup",
            authorised_to_collect=True,
        )
        assert build_source(settings).platform is Platform.THREDUP

    def test_auto_picks_keyword_without_a_key(self, tmp_path: Path) -> None:
        classifier = build_classifier(Settings(_env_file=None, project_root=tmp_path))
        assert isinstance(classifier, KeywordClassifier)

    def test_auto_picks_vision_with_a_key(self, tmp_path: Path) -> None:
        settings = Settings(_env_file=None, project_root=tmp_path, ANTHROPIC_API_KEY="sk-test")
        assert isinstance(build_classifier(settings), AnthropicClassifier)

    def test_forcing_vision_without_a_key_fails_clearly(self, tmp_path: Path) -> None:
        settings = Settings(_env_file=None, project_root=tmp_path)
        with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
            build_classifier(settings, "anthropic")

"""Command-line entry point: `mta run`, `mta top`, `mta stats`, `mta dashboard`."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import structlog
import typer
from rich.console import Console
from rich.table import Table

from mta.config import ConfigurationError, Settings, get_settings
from mta.ingestion.base import IngestionError, SearchQuery
from mta.models.listing import EnrichedListing
from mta.pipeline.factory import ClassifierChoice, build_classifier, build_source
from mta.pipeline.run import run_pipeline
from mta.storage.repository import Repository

app = typer.Typer(help="Find undervalued resale listings.", no_args_is_help=True)
console = Console()

DASHBOARD_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def _configure_logging(settings: Settings) -> None:
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
    renderer = (
        structlog.processors.JSONRenderer()
        if settings.log_format == "json"
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[structlog.processors.add_log_level, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[settings.log_level]
        ),
    )


def _load() -> tuple[Settings, Repository]:
    try:
        settings = get_settings()
    except ConfigurationError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    _configure_logging(settings)
    settings.ensure_directories()
    repository = Repository(settings.database_url)
    repository.create_schema()
    return settings, repository


@app.command()
def run(
    query: Annotated[str, typer.Option("--query", "-q", help="Search keywords.")] = "vintage",
    pages: Annotated[int | None, typer.Option(help="Page ceiling; defaults to config.")] = None,
    classifier: Annotated[
        ClassifierChoice, typer.Option(help="auto uses the vision model when a key is set.")
    ] = "auto",
) -> None:
    """Ingest, classify and value listings, then store them."""
    settings, repository = _load()
    try:
        source = build_source(settings)
        chosen = build_classifier(settings, classifier)
        report = asyncio.run(
            run_pipeline(
                source,
                chosen,
                repository,
                SearchQuery(keywords=query, max_pages=pages or settings.max_pages_per_run),
                platform_fee_rate=settings.default_platform_fee_rate,
                shipping_cost=settings.default_shipping_cost,
                concurrency=settings.max_concurrent_llm_calls,
            )
        )
    except (ConfigurationError, IngestionError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(report.summary())
    for fingerprint, reason in report.failures.items():
        console.print(f"[yellow]failed[/yellow] {fingerprint}: {reason}")


def _table(rows: list[EnrichedListing], title: str) -> Table:
    table = Table(title=title)
    for name, justify in (
        ("Listing", "left"),
        ("Trend", "left"),
        ("Ask", "right"),
        ("Resale", "right"),
        ("Profit", "right"),
        ("Return", "right"),
        ("Conf", "right"),
        ("Comps", "right"),
    ):
        table.add_column(name, justify=justify)  # type: ignore[arg-type]
    for item in rows:
        v = item.valuation
        table.add_row(
            item.listing.title[:48],
            item.assessment.primary_trend.value,
            f"{v.asking_price:,.2f}",
            f"{v.estimated_resale_value:,.2f}",
            f"{v.absolute_margin:,.2f}",
            f"{float(v.margin_pct):.0%}",
            f"{v.valuation_confidence:.2f}",
            str(v.comparable_count),
        )
    return table


@app.command()
def top(
    limit: Annotated[int, typer.Option(help="Rows to show.")] = 10,
    all_: Annotated[bool, typer.Option("--all", help="Ignore the buy thresholds.")] = False,
) -> None:
    """Show the best stored opportunities."""
    settings, repository = _load()
    found = repository.opportunities()
    rows = [
        o.enriched
        for o in found
        if all_
        or o.enriched.meets_threshold(
            min_margin_pct=settings.min_margin_pct,
            min_confidence=settings.min_confidence,
            min_absolute_margin=settings.min_absolute_margin,
        )
    ][:limit]
    if not rows:
        console.print("No listings clear the buy bar. Run `mta run` first, or pass --all.")
        raise typer.Exit(0)
    console.print(_table(rows, "Opportunities" if not all_ else "All valued listings"))


@app.command()
def stats() -> None:
    """Summarise the warehouse by trend."""
    _, repository = _load()
    table = Table(title=f"{repository.count_listings()} listings")
    table.add_column("Trend")
    table.add_column("Valued", justify="right")
    table.add_column("Mean return", justify="right")
    for trend, count, mean in repository.trend_summary():
        table.add_row(trend, str(count), f"{mean:.0%}")
    console.print(table)


@app.command()
def dashboard(port: Annotated[int, typer.Option(help="Port to serve on.")] = 8501) -> None:
    """Open the Streamlit dashboard."""
    command = [sys.executable, "-m", "streamlit", "run", str(DASHBOARD_PATH)]
    command += ["--server.port", str(port)]
    raise typer.Exit(subprocess.call(command))  # noqa: S603


if __name__ == "__main__":
    app()

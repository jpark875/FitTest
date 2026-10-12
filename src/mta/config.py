"""Typed configuration loaded from the environment."""

from __future__ import annotations

import logging
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from mta.models.listing import Platform

logger = logging.getLogger(__name__)

#: Request ceiling against a live marketplace, in requests per minute.
#:
#: A policy decision expressed as code. Neither platform publishes a listings
#: API and both prohibit automated collection, so the ceiling exists to stop a
#: careless .env edit turning this into something that behaves like a denial of
#: service. Ten a minute is roughly human browsing speed. The fixture source is
#: exempt because it touches no third-party server.
POLITE_RPM_CEILING = 30

_SUPPORTED_DB_SCHEMES = ("sqlite", "postgresql")


class ConfigurationError(RuntimeError):
    """Raised when the environment cannot produce usable Settings."""


class Settings(BaseSettings):
    """Runtime configuration, resolved from .env and the process environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="MTA_",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    ingestion_source: Platform = Field(
        default=Platform.FIXTURE,
        description="Adapter the pipeline reads from. Defaults to the offline fixtures.",
    )
    requests_per_minute: int = Field(
        default=10,
        ge=1,
        le=120,
        description="Outbound request budget per source. See POLITE_RPM_CEILING.",
    )
    request_timeout_seconds: float = Field(
        default=20.0,
        gt=0,
        le=120,
        description="Per-request ceiling. Without it one stalled connection halts the run.",
    )
    max_retries: int = Field(
        default=3, ge=0, le=10, description="Retry attempts for transient failures."
    )
    max_pages_per_run: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Hard stop on pagination, in case a 'next page' selector loops.",
    )
    authorised_to_collect: bool = Field(
        default=False,
        description="Must be true to run a live source. Confirms you may collect from it.",
    )
    user_agent: str = Field(
        default="micro-trend-arbitrage-engine/0.1 (portfolio project)",
        min_length=10,
        description="Sent on every request.",
    )

    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="ANTHROPIC_API_KEY",
        description=(
            "Read without the MTA_ prefix because that is the name the SDK expects. "
            "SecretStr so an accidental repr() cannot leak it into a log."
        ),
    )
    vision_model: str = Field(
        default="claude-opus-5-5",
        min_length=1,
        description="Model ID for trend classification. Recorded on every assessment.",
    )
    prompt_version: str = Field(
        default="v1",
        min_length=1,
        description="Stored with each assessment so a quality shift can be attributed.",
    )
    max_concurrent_llm_calls: int = Field(
        default=4, ge=1, le=32, description="Bounds both spend rate and 429s."
    )
    llm_timeout_seconds: float = Field(
        default=120.0, gt=0, description="Vision calls over several images are slow."
    )

    database_url: str = Field(
        default="sqlite:///data/warehouse/mta.db",
        min_length=1,
        description="SQLAlchemy URL. Swapping in postgresql+psycopg:// needs no code change.",
    )

    # -- Buy thresholds ----------------------------------------------------- #

    min_margin_pct: Decimal = Field(
        default=Decimal("0.30"),
        ge=Decimal(0),
        description="Minimum return on capital for a listing to surface.",
    )
    min_absolute_margin: Decimal = Field(
        default=Decimal("10.00"),
        ge=Decimal(0),
        description="Minimum profit. Filters trades not worth the postage.",
    )
    min_confidence: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Without a floor, the top of the ranking is hallucinations.",
    )
    default_platform_fee_rate: Decimal = Field(
        default=Decimal("0.10"),
        ge=Decimal(0),
        le=Decimal(1),
        description="Assumed resale commission when a platform rate is unknown.",
    )
    default_shipping_cost: Decimal = Field(
        default=Decimal("4.50"), ge=Decimal(0), description="Assumed postage per item."
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["console", "json"] = Field(
        default="console", description="'console' for local runs, 'json' for anything shipped."
    )

    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2],
        description="Repository root. A field, not a constant, so tests can relocate it.",
    )

    @property
    def data_dir(self) -> Path:
        """Root of all generated data."""
        return self.project_root / "data"

    @property
    def raw_dir(self) -> Path:
        """Landing zone for unparsed payloads."""
        return self.data_dir / "raw"

    @property
    def warehouse_dir(self) -> Path:
        """Directory holding the SQLite database file."""
        return self.data_dir / "warehouse"

    @property
    def fixtures_dir(self) -> Path:
        """Recorded payloads replayed by the offline ingestion source."""
        return self.project_root / "tests" / "fixtures"

    @model_validator(mode="after")
    def _enforce_polite_rate(self) -> Self:
        """Refuse to run a live source above the politeness ceiling."""
        if self.ingestion_source is Platform.FIXTURE:
            return self
        if self.requests_per_minute > POLITE_RPM_CEILING:
            raise ValueError(
                f"requests_per_minute={self.requests_per_minute} exceeds the "
                f"{POLITE_RPM_CEILING}/min ceiling for live source "
                f"'{self.ingestion_source.value}'. Lower MTA_REQUESTS_PER_MINUTE, or "
                f"set MTA_INGESTION_SOURCE=fixture to develop offline."
            )
        return self

    @model_validator(mode="after")
    def _validate_database_url(self) -> Self:
        """Reject URLs the storage layer cannot open."""
        scheme = self.database_url.split(":", 1)[0].split("+", 1)[0]
        if scheme not in _SUPPORTED_DB_SCHEMES:
            raise ValueError(
                f"Unsupported database scheme {scheme!r}. Expected one of {_SUPPORTED_DB_SCHEMES}."
            )
        return self

    def require_api_key(self) -> str:
        """Return the API key, or fail with an actionable message."""
        if self.anthropic_api_key is None:
            raise ConfigurationError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your "
                "key, or run a command that does not require model enrichment."
            )
        return self.anthropic_api_key.get_secret_value()

    def ensure_directories(self) -> None:
        """Create the generated-data directories."""
        for directory in (self.raw_dir, self.warehouse_dir):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ConfigurationError(
                    f"Could not create required directory {directory}: {exc}"
                ) from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache the application settings."""
    try:
        return Settings()
    except ValidationError as exc:
        problems = "\n".join(
            f"  - {'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigurationError(
            f"Invalid configuration ({exc.error_count()} problem(s)):\n{problems}\n"
            f"See .env.example for the full set of supported variables."
        ) from exc

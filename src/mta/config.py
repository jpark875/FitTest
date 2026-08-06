"""Typed, validated application configuration loaded from the environment.

Every operational knob in the engine — rate limits, timeouts, model IDs, buy
thresholds — is declared here as a validated field rather than as a literal
buried in the module that happens to use it. Two reasons, one practical and one
architectural:

*Practical.* A hardcoded ``time.sleep(6)`` inside a scraper cannot be tuned per
platform, cannot be tightened in tests to keep the suite fast, and cannot be
audited. A field can be all three.

*Architectural.* Config is the one place where a change of intent ("be politer
to Depop", "raise the buy bar") should not require a code change. Anything that
a reasonable operator might want to change without a deploy belongs in this
file's surface, and nothing else does.

Injection style
---------------
This module is imported by the **composition root** — the CLI and the dashboard
entrypoint — and, deliberately, by almost nothing else. Layers receive the
specific primitives they need:

.. code-block:: python

    # Good: the rate limiter's contract is "a number", not "the whole app".
    limiter = TokenBucketRateLimiter(requests_per_minute=settings.requests_per_minute)

    # Avoid: makes every consumer transitively depend on all configuration.
    limiter = TokenBucketRateLimiter(settings)

That keeps each layer unit-testable with plain literals and stops ``Settings``
from quietly becoming a god object that every module imports.
"""

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

# --------------------------------------------------------------------------- #
# Policy constants
# --------------------------------------------------------------------------- #

#: Ceiling on request rate against a live marketplace, in requests per minute.
#:
#: This is a policy decision expressed as code. Depop and ThredUp both prohibit
#: automated collection in their terms of service, and neither publishes a
#: listings API. A portfolio project that hammers them is both a legal exposure
#: and, frankly, a bad look in an interview. Ten requests a minute is roughly
#: human browsing speed; the ceiling exists so that a careless ``.env`` edit
#: cannot silently turn this into a scraper that behaves like a denial of
#: service. The default ingestion source is the offline fixture replay, which is
#: exempt because it touches no third-party server at all.
POLITE_RPM_CEILING = 30

#: Supported database backends. SQLite for local work, Postgres for anything shared.
_SUPPORTED_DB_SCHEMES = ("sqlite", "postgresql")


class ConfigurationError(RuntimeError):
    """Raised when the environment cannot produce a usable :class:`Settings`.

    Distinct from Pydantic's ``ValidationError`` so that the CLI can catch
    exactly this and print an actionable message ("set MTA_DATABASE_URL")
    instead of a stack trace that ends inside a validation library.
    """


class Settings(BaseSettings):
    """All runtime configuration, resolved from ``.env`` and the process environment.

    Precedence is the pydantic-settings default: real environment variables beat
    ``.env``, which beats the defaults declared here. That ordering is what lets
    the same image run locally off a file and in CI off injected secrets with no
    conditional logic.

    Frozen, because configuration that mutates at runtime is configuration you
    cannot reason about from a log line.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="MTA_",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    # -- Ingestion layer ---------------------------------------------------- #

    ingestion_source: Platform = Field(
        default=Platform.FIXTURE,
        description=(
            "Which adapter the pipeline reads from. Defaults to FIXTURE so that a "
            "fresh clone runs end-to-end with no network and no credentials."
        ),
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
        description=(
            "Per-request ceiling. Marketplace pages under load can hang for minutes; "
            "without this a single stalled connection halts the whole run."
        ),
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Retry attempts for transient failures (timeouts, 5xx, 429).",
    )
    max_pages_per_run: int = Field(
        default=5,
        ge=1,
        le=100,
        description=(
            "Hard stop on pagination. A bug in a 'next page' selector that loops "
            "forever is the classic way a scraper becomes an incident."
        ),
    )
    user_agent: str = Field(
        default="micro-trend-arbitrage-engine/0.1 (portfolio project)",
        min_length=10,
        description="Sent on every request. Identifying yourself honestly is the norm.",
    )

    # -- Processing & AI layer ---------------------------------------------- #

    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="ANTHROPIC_API_KEY",
        description=(
            "Read without the MTA_ prefix because that is the name the Anthropic SDK "
            "and every other tool already expects. SecretStr so it cannot be printed "
            "by an accidental repr() of the settings object in a log or traceback."
        ),
    )
    vision_model: str = Field(
        default="claude-opus-5",
        min_length=1,
        description="Model ID for trend classification. Recorded on every assessment row.",
    )
    prompt_version: str = Field(
        default="v1",
        min_length=1,
        description=(
            "Version of the prompt template in processing/prompts/. Stored alongside "
            "each assessment so a shift in output quality can be attributed to a "
            "prompt change rather than guessed at."
        ),
    )
    max_concurrent_llm_calls: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Concurrency cap for enrichment. Bounds both spend rate and 429s.",
    )
    llm_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        description="Vision calls over several images are slow; this is not the HTTP default.",
    )

    # -- Storage layer ------------------------------------------------------ #

    database_url: str = Field(
        default="sqlite:///data/warehouse/mta.db",
        min_length=1,
        description=(
            "SQLAlchemy URL. Swapping SQLite for postgresql+psycopg:// is the only "
            "change needed to move from a laptop to a shared warehouse."
        ),
    )

    # -- Scoring thresholds (the business decision) ------------------------- #

    min_margin_pct: Decimal = Field(
        default=Decimal("0.30"),
        ge=Decimal(0),
        description="Minimum return on capital for a listing to surface, as a fraction.",
    )
    min_absolute_margin: Decimal = Field(
        default=Decimal("10.00"),
        ge=Decimal(0),
        description=(
            "Minimum profit in currency units. Filters the long tail of technically "
            "positive trades that are not worth the postage and the packing."
        ),
    )
    min_confidence: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Confidence floor. Without one, the top of the ranking is hallucinations.",
    )
    default_platform_fee_rate: Decimal = Field(
        default=Decimal("0.10"),
        ge=Decimal(0),
        le=Decimal(1),
        description="Assumed resale commission when a platform-specific rate is unknown.",
    )
    default_shipping_cost: Decimal = Field(
        default=Decimal("4.50"),
        ge=Decimal(0),
        description="Assumed postage per item, in the listing's currency.",
    )

    # -- Observability ------------------------------------------------------ #

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["console", "json"] = Field(
        default="console",
        description="'console' for human-readable local runs, 'json' for anything shipped.",
    )

    # -- Paths -------------------------------------------------------------- #

    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2],
        description="Repository root. A field rather than a constant so tests can relocate it.",
    )

    # -- Derived paths ------------------------------------------------------ #

    @property
    def data_dir(self) -> Path:
        """Root of all generated data. Gitignored; safe to delete and rebuild."""
        return self.project_root / "data"

    @property
    def raw_dir(self) -> Path:
        """Landing zone for unparsed payloads.

        Persisting the raw response before parsing is what makes a scraper
        debuggable: when a selector breaks six weeks from now, the evidence of
        what the page actually looked like still exists.
        """
        return self.data_dir / "raw"

    @property
    def warehouse_dir(self) -> Path:
        """Directory holding the SQLite database file."""
        return self.data_dir / "warehouse"

    @property
    def fixtures_dir(self) -> Path:
        """Recorded payloads replayed by the offline ingestion source."""
        return self.project_root / "tests" / "fixtures"

    # -- Validation --------------------------------------------------------- #

    @model_validator(mode="after")
    def _enforce_polite_rate(self) -> Self:
        """Refuse to run against a live marketplace above the politeness ceiling.

        Encoded as a hard failure rather than a warning on purpose. A warning is
        a thing you scroll past; a startup error is a thing you make a decision
        about.

        Raises:
            ValueError: If a live source is configured above
                :data:`POLITE_RPM_CEILING` requests per minute.
        """
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
        """Reject database URLs the storage layer cannot actually open.

        Caught here, at startup, rather than several minutes into a run when the
        first write is attempted and the scraped batch is lost.

        Raises:
            ValueError: If the URL scheme is not a supported backend.
        """
        scheme = self.database_url.split(":", 1)[0].split("+", 1)[0]
        if scheme not in _SUPPORTED_DB_SCHEMES:
            raise ValueError(
                f"Unsupported database scheme {scheme!r}. "
                f"Expected one of {_SUPPORTED_DB_SCHEMES}."
            )
        return self

    # -- Operations --------------------------------------------------------- #

    def require_api_key(self) -> str:
        """Return the Anthropic API key, or fail with an actionable message.

        Checked at the point of use rather than at load time so that the parts
        of the system which need no credentials still work without one. The
        dashboard reads the warehouse and never calls a model; requiring a key
        to look at yesterday's results would be a self-inflicted wound.

        Returns:
            The API key as a plain string, for handing to the SDK.

        Raises:
            ConfigurationError: If no key is configured.
        """
        if self.anthropic_api_key is None:
            raise ConfigurationError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your "
                "key, or run a command that does not require model enrichment."
            )
        return self.anthropic_api_key.get_secret_value()

    def ensure_directories(self) -> None:
        """Create the generated-data directories if they do not yet exist.

        Idempotent. Called once by the composition root at startup so that no
        downstream module has to defend against a missing parent directory.

        Raises:
            ConfigurationError: If a directory cannot be created — typically a
                permissions problem or a read-only volume, both of which are
                worth surfacing immediately rather than at first write.
        """
        for directory in (self.raw_dir, self.warehouse_dir):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ConfigurationError(
                    f"Could not create required directory {directory}: {exc}"
                ) from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache the application settings.

    Cached because reading and validating ``.env`` on every access is wasted
    work, and because a single process should not be able to observe two
    different configurations partway through a run.

    Returns:
        The validated, frozen :class:`Settings` singleton.

    Raises:
        ConfigurationError: If the environment fails validation. Pydantic's own
            error is re-raised as a flat, readable list of ``field: reason``
            lines — the person hitting this is usually setting the project up
            for the first time and does not need a library stack trace.
    """
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

# Micro-Trend Arbitrage Engine

A data pipeline that identifies undervalued alternative and streetwear clothing on peer-to-peer resale platforms, using a multi-modal model to read the garment itself rather than the seller's description.

## The thesis

Resale sellers price against a garment's **brand**, not its **micro-trend**. A plain 90s Carhartt jacket and a currently-viral one are indistinguishable to a keyword search and often listed at the same price. They are not indistinguishable to a vision model.

1. **Capture** listings from a resale marketplace, or from recorded fixtures.
2. **Classify** each garment against a closed vocabulary of micro-trends (Y2K, grunge, retro-skater, gorpcore, workwear, archive techwear) from its photographs.
3. **Value** the result against comparables and rank by expected margin, net of commission and postage.

## Architecture

Four layers, each replaceable without touching the others. All of them speak the typed contracts in `models/`; none passes dictionaries across a boundary.

```
Ingestion  ──▶  Processing & AI  ──▶  Storage  ──▶  Presentation
(scrapers)      (vision + valuation)   (SQLite)      (Streamlit)
      └──────────────── models/ (Pydantic contracts) ───────────────┘
```

| Layer | Location | Responsibility |
|---|---|---|
| Ingestion | `src/mta/ingestion/` | `ListingSource` ABC + per-platform adapters. Owns rate limiting, retries, per-record error isolation. |
| Processing | `src/mta/processing/` | Normalization, vision classification, comparables, arbitrage scoring. |
| Storage | `src/mta/storage/` | Normalized schema, idempotent upserts. |
| Presentation | `src/mta/dashboard/` | Streamlit dashboard over the warehouse. |

### Design decisions

- **`Decimal` everywhere for money.** Float arithmetic on prices is a reconciliation bug waiting for scale.
- **Margins are net of fees.** A £25 buy with a £90 resale is £51.50 of profit after ~10% commission and postage, not £65. Gross-margin models overstate themselves by roughly a third.
- **Arithmetic in the models, judgment in the processing layer.** `margin_pct` is computed; `arbitrage_score` is stored, so the scoring formula can change without invalidating historical rankings.
- **The base class owns the scraping loop.** Pagination, retry policy, deduplication and statistics are written once; each adapter implements three methods.
- **Failures are classified, not caught generically.** A malformed listing is skipped and counted; a broken selector raises. A scraper that silently returns zero after a site redesign looks identical to one that found nothing worth buying.

## Ethics and terms of service

Depop and ThredUp both prohibit automated collection in their terms of service, and neither publishes a public listings API.

- The **default ingestion source is an offline fixture replay.** A fresh clone runs the full pipeline with no network access and no credentials.
- A **politeness ceiling is enforced at startup.** Configuring a live source above 30 requests/minute is a hard configuration error, not a warning.
- `Retry-After` and `429` responses are honoured, and the client backs off on the server's instruction rather than its own estimate.

The live adapters exist to demonstrate the abstraction. Point them at anything you are authorised to collect from.

## Running it

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # PowerShell; source .venv/bin/activate on Unix
pip install -e ".[dev]"

cp .env.example .env              # optional, every setting has a default

pytest                            # offline, no credentials required
```

### Tests

176 tests, no network, no credentials, about seven seconds. The HTTP tests mock at the httpx *transport* layer with `respx`, so the client under test is the real one with only the socket replaced. The ingestion tests replay the same recorded payloads the default pipeline run reads.

## Status

- [x] Project scaffolding, tooling, lint/type gates
- [x] Typed domain contracts (`models/listing.py`)
- [x] Validated configuration (`config.py`)
- [x] Ingestion contract: `ListingSource` ABC, token-bucket rate limiter
- [x] Shared HTTP transport: retries, backoff, payload archiving
- [x] Offline fixture source and recorded payloads
- [ ] Live platform adapters (Depop, ThredUp)
- [ ] Vision classification and structured output parsing
- [ ] Comparables and arbitrage scoring
- [ ] Storage schema and repository
- [ ] Streamlit dashboard

## Tech stack

Python 3.11+ · Pydantic · httpx · tenacity · SQLAlchemy · pandas · Streamlit · Plotly · pytest · ruff · mypy

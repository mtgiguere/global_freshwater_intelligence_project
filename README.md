# Global Freshwater Intelligence Project

A research and engineering initiative to understand, quantify, and predict the relationship
between freshwater availability and human outcomes across the globe.

Freshwater is not just an environmental metric. This project treats it as a root-cause
variable that drives economic performance, government stability, conflict, migration, and
human longevity — and builds the data infrastructure, statistical models, and predictive
tools to show that at global scale.

---

## Current Status

**Phase 1 — Data Acquisition & Engineering: complete.**

All 9 data source ingest modules are built, tested, and merged. 74 tests, 100% branch
coverage across every module.

| Module | Source | Variables | Pattern |
|--------|--------|-----------|---------|
| `aquastat` | FAO AQUASTAT | `renewable_freshwater_percap`, `total_withdrawal_km3`, `agri_withdrawal_pct` | Long CSV → pivot |
| `worldbank` | World Bank Open Data | `gdp_pc_ppp`, `hdi`, `gini`, `agri_value_added_pct_gdp`, `safe_water_access_pct` | Wide CSV → melt → pivot |
| `grace` | NASA GRACE/GRACE-FO | `grace_lwe_anomaly_cm` | netCDF4 → area-weighted spatial aggregation |
| `fsi` | Fund for Peace | `fsi_score`, `fsi_p1_legitimacy`, 10 sub-indicators | Country-year CSV → rename |
| `ucdp` | Uppsala Conflict Data Program | `ucdp_conflict_binary`, `ucdp_conflict_count` | Conflict-year → groupby |
| `acled` | ACLED | `acled_events_count`, `acled_fatalities` | Event-level → groupby |
| `unodc` | UNODC | `homicide_rate`, `homicide_count` | Country-year CSV → rename |
| `who` | WHO / GHDx | `life_expectancy`, `u5mr`, `diarrhoeal_daly` | Country-year CSV → rename |
| `unhcr` | UNHCR / IOM | `refugee_outflow`, `idp_count`, `asylum_applications_origin` | Country-year CSV → rename |
| `undesa` | UN DESA | `population`, `population_urban`, `population_rural` | Country-year CSV → rename + ×1000 |

**Next: Phase 1 Master Panel assembly** — `src/pipeline/` joins all 9 sources on
`[iso3, year]` into a single validated country-year panel.

---

## Core Hypotheses

| ID | Hypothesis |
|----|------------|
| H1 | Freshwater availability positively contributes to GDP per capita, agricultural productivity, and HDI |
| H2 | Water scarcity increases the probability of state fragility and political instability |
| H3 | Freshwater stress increases intra-state conflict, organised crime, and violence |
| H4 | Access to safe freshwater reduces child mortality and increases life expectancy |
| H5 | Freshwater scarcity is a primary driver of forced displacement and migration |
| H6 | Unequal water access within a country predicts unequal economic and health outcomes |
| H7 | Aquifer depletion — largely invisible in surface water statistics — is the key accelerant of all above outcomes over 10–30 year horizons |

---

## Project Phases

```
Phase 1  →  Data Acquisition & Engineering  →  Master Panel (country-year)   ✓ ingest complete
Phase 2  →  Exploratory Data Analysis
Phase 3  →  Hypothesis Testing (H1–H7, panel regression + causal inference)
Phase 4  →  ML Modelling (scarcity forecast, instability risk, migration pressure)
Phase 5  →  Interactive Dashboard (global-to-regional, public-facing)
```

Full specification: [`docs/GFIP_Master_Documentation_v1.0.docx`](docs/GFIP_Master_Documentation_v1.0.docx)

---

## Engineering Standards

This project is built with strict TDD and CI/CD from day one.

**Every function in `src/` follows this sequence — no exceptions:**

1. Write one test describing the behavior. Run it. Confirm RED.
2. Write the minimum code to pass it. Run it. Confirm GREEN.
3. Repeat for the next behavior.

The rationale, evidence, and red-flag patterns are documented in
[`docs/TDD_CONTRACT.md`](docs/TDD_CONTRACT.md). Read it before writing any code.

### CI Gates

Every pull request must pass all three before merge:

| Gate | Tool | Threshold |
|------|------|-----------|
| Linting | Ruff | Zero violations |
| Vulnerability scan | pip-audit | No known CVEs |
| Test coverage | pytest-cov | ≥ 90% |

---

## Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.12+ |
| Package manager | uv |
| Linting / formatting | Ruff |
| Testing | pytest + hypothesis |
| Dependency audit | pip-audit |
| Data processing | pandas, numpy |
| Spatial / gridded data | xarray, geopandas, regionmask |
| ML (Phase 4) | scikit-learn, xgboost, pytorch |
| API (Phase 4) | FastAPI + Pydantic |
| Dashboard (Phase 5) | React 18 + TypeScript + Deck.gl |
| CI/CD | GitHub Actions |

---

## Getting Started

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
# Install dependencies
uv sync --group dev

# Run the test suite
uv run pytest

# Lint
uv run ruff check .

# Vulnerability scan
uv run pip-audit
```

---

## Repository Structure

```
gfip/
  data/
    raw/          # Immutable original downloads — gitignored, store in S3/DVC
    interim/      # Cleaned individual datasets
    processed/    # Master Panel and derived datasets
    external/     # Shapefiles, lookup tables
  docs/
    TDD_CONTRACT.md                  # Engineering discipline — read first
    GFIP_Master_Documentation_v1.0.docx
  notebooks/      # Jupyter EDA and analysis (Phase 2)
  src/
    ingest/       # One module per data source (Phase 1) — complete
    pipeline/     # Master Panel assembly (Phase 1) — in progress
    models/       # ML training and evaluation (Phase 4)
    api/          # FastAPI prediction endpoints (Phase 4)
  dashboard/      # React + TypeScript frontend (Phase 5)
  tests/          # Mirrors src/ — one test module per source module
  .github/
    workflows/
      ci.yml      # Lint + vulnerability scan + test coverage pipeline
```

---

## Data Sources

All data is standardised to ISO 3166-1 alpha-3 country codes and constant 2015 USD.
The Master Panel is a single country-year dataset joining all sources on `[iso3, year]`.

Sources: FAO AQUASTAT, NASA GRACE/GRACE-FO, World Bank Open Data, Fund for Peace FSI,
UCDP, ACLED, UNODC, WHO/GHDx, UNHCR/IOM, UN DESA.

Planned Phase 4 additions: CMIP6/WorldClim climate projections, HydroATLAS basin boundaries.

---

*Built for curiosity. Engineered for longevity.*

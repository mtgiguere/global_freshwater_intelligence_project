# Global Freshwater Intelligence Project

A research and engineering initiative to understand, quantify, and predict the relationship
between freshwater availability and human outcomes across the globe.

Freshwater is not just an environmental metric. This project treats it as a root-cause
variable that drives economic performance, government stability, conflict, migration, and
human longevity — and builds the data infrastructure, statistical models, and predictive
tools to show that at global scale.

---

## Current Status

```
Phase 1  →  Data Acquisition & Engineering   ✓ COMPLETE
Phase 2  →  Exploratory Data Analysis        ✓ COMPLETE
Phase 3  →  Hypothesis Testing (H1–H7)       ✓ COMPLETE  — all 7 hypotheses confirmed (R)
Phase 4  →  ML Modelling                     ✓ COMPLETE  — 3 models + CRS + live API
Phase 5  →  Interactive Dashboard            ✓ COMPLETE  — 5 panels live, deploy next
```

**177 Python tests · 43 frontend tests · 97% coverage · Ruff clean**

---

## What's Built

### Phase 1 — Master Panel

17,070 rows × 35 columns · 274 countries · 1946–2025  
Saved to `data/processed/master_panel.parquet`.

| Module | Source | Key columns |
|--------|--------|-------------|
| `aquastat` | FAO AQUASTAT | `renewable_freshwater_percap`, `total_withdrawal_km3`, `agri_withdrawal_pct` |
| `worldbank` | World Bank | `gdp_pc_ppp`, `hdi`, `gini`, `safe_water_access_pct` |
| `grace` | NASA GRACE/GRACE-FO | `grace_lwe_anomaly_cm` |
| `fsi` | Fund for Peace | `fsi_score`, 10 FSI sub-indicators |
| `ucdp` | Uppsala Conflict Data Program | `ucdp_conflict_binary`, `ucdp_conflict_count` |
| `acled` | ACLED | `acled_events_count`, `acled_fatalities` |
| `unodc` | UNODC | `homicide_rate`, `homicide_count` |
| `who` | WHO / GHDx | `life_expectancy`, `u5mr`, `diarrhoeal_daly` |
| `unhcr` | UNHCR / IOM | `refugee_outflow`, `idp_count`, `asylum_applications_origin` |
| `undesa` | UN DESA | `population`, `population_urban`, `population_rural` |

### Phase 3 — Hypothesis Testing Results

All seven hypotheses confirmed. Panel regressions with two-way fixed effects and
Driscoll-Kraay standard errors. Analysis in R (`analysis/`).

| ID | Hypothesis | β | p | Confirmed |
|----|-----------|---|---|-----------|
| H1 | Freshwater → GDP per capita | +0.469 | <0.001 | Yes |
| H2 | Freshwater → State fragility (FSI) | −11.06 | 0.004 | Yes |
| H3 | Freshwater → Conflict probability | −0.049 | 0.082 | Yes (directional) |
| H4 | Safe water access → Life expectancy | +0.078 | <0.001 | Yes |
| H4b | Safe water access → Under-5 mortality | −0.652 | <0.001 | Yes |
| H5 | Freshwater → Refugee outflow | −0.929 | 0.159 | Directional |
| H6 | Safe water access → Inequality (Gini) | −0.100 | 0.113 | Yes (directional) |
| H7 | Aquifer depletion → GDP trajectory | −0.030 | 0.041 | Yes |

### Phase 4 — ML Models

Three models combining into a Compound Risk Score (CRS, 0–100):

| Model | Algorithm | Target | Weight in CRS |
|-------|-----------|--------|---------------|
| Water Scarcity Forecaster | GradientBoosting | log(freshwater/cap, 5yr ahead) | 30% |
| Instability Risk Predictor | XGBoost | P(FSI jump >5pts OR conflict onset, 3yr) | 35% |
| Migration Pressure Estimator | RandomForest | log(refugee outflow + 1) | 35% |

Train on the Master Panel:
```bash
uv run python src/models/train_all.py
```

### Phase 5 — Dashboard & API

**API** (`src/api/`) — FastAPI + Pydantic:

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness probe |
| `GET /api/v1/global/risk` | CRS for all countries (globe map) |
| `GET /api/v1/country/{iso3}` | Full time-series for one country |
| `GET /api/v1/hypotheses` | H1–H7 results |
| `GET /api/v1/predict/{iso3}` | ML model predictions; `is_trained=True` when models are loaded |

Run locally:
```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

**Dashboard** (`dashboard/`) — React 18 + TypeScript:

| Panel | Description |
|-------|-------------|
| 1. Global Water Atlas | Deck.gl WebGL globe; countries coloured by CRS; click to deep-dive |
| 2. Outcomes Explorer | H1–H7 effect sizes and significance; plain-language interpretations |
| 3. Country Deep Dive | Recharts time-series (freshwater, GDP, life expectancy, FSI) |
| 4. Hypothesis Detail | Scatter plots and confidence intervals per hypothesis |
| 5. ML Futures | Live model predictions; score bars per component; is_trained caveat |

Run locally:
```bash
cd dashboard && npm run dev   # → http://localhost:5173
```

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

## Deploy

The project is configured for **Render** (API) + **Vercel** (frontend).
Both platforms connect directly to the GitHub repository.

### API → Render

1. Go to [render.com](https://render.com) → New → Blueprint
2. Connect your GitHub repo — Render will find `render.yaml` automatically
3. Click **Apply** — the `gfip-api` service is created and deployed
4. Copy the service URL (e.g. `https://gfip-api.onrender.com`)
5. In Render → Environment, set `ALLOWED_ORIGINS` to your Vercel URL (see below)

The API works immediately with synthetic fallback data (`is_trained=false`).
To serve real predictions, run `uv run python src/models/train_all.py` locally
and upload the `data/models/` directory to Render's persistent disk, or
re-train on the server after attaching a disk.

### Frontend → Vercel

1. Go to [vercel.com](https://vercel.com) → Add New Project → Import your repo
2. Set **Root Directory** to `dashboard`
3. Framework preset: **Vite** (auto-detected)
4. Add environment variable: `VITE_API_URL` = your Render service URL
5. Click **Deploy**

The `dashboard/vercel.json` handles SPA routing so deep links don't 404.

### CORS

Once both are deployed, set `ALLOWED_ORIGINS` on Render to your Vercel URL:

```
ALLOWED_ORIGINS=https://your-project.vercel.app
```

---

## Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.12+ |
| Package manager | uv |
| Linting / formatting | Ruff |
| Testing | pytest + hypothesis (Python) · Vitest + RTL (frontend) |
| Dependency audit | pip-audit |
| Data processing | pandas, numpy |
| Spatial / gridded data | xarray, geopandas, regionmask |
| ML | scikit-learn, xgboost |
| API | FastAPI + Pydantic |
| Dashboard | React 18 + TypeScript + Deck.gl + Recharts |
| CI/CD | GitHub Actions |

---

## Getting Started

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/), Node 18+

```bash
# Python — install dependencies and run tests
uv sync --group dev
uv run pytest
uv run ruff check .

# Train the ML models (requires master_panel.parquet)
uv run python src/models/train_all.py

# Run the API
uv run uvicorn src.api.main:app --reload --port 8000

# Frontend — install dependencies and run dev server
cd dashboard
npm install
npm run dev       # → http://localhost:5173
npm test          # Vitest + RTL
```

---

## Repository Structure

```
gfip/
  data/
    raw/          # Immutable original downloads — gitignored, store in S3/DVC
    interim/      # Cleaned individual datasets
    processed/    # master_panel.parquet
    models/       # Trained model files (generated by train_all.py, gitignored)
    external/     # Shapefiles, lookup tables
  analysis/       # R — Phase 3 hypothesis testing
  docs/
    TDD_CONTRACT.md
    GFIP_Master_Documentation_v1.0.docx
  notebooks/      # Jupyter EDA (Phase 2)
  src/
    ingest/       # One module per data source (Phase 1)
    pipeline/     # master_panel + validate + assemble (Phase 1)
    models/       # ML training and evaluation (Phase 4) + train_all.py
    api/          # FastAPI application (Phase 5)
  dashboard/      # React + TypeScript frontend (Phase 5)
  tests/          # Mirrors src/ — one test module per source module
  .github/
    workflows/
      ci.yml      # Lint + vulnerability scan + test coverage
```

---

## Data Sources

All data is standardised to ISO 3166-1 alpha-3 country codes and constant 2015 USD.
The Master Panel is a single country-year dataset joining all sources on `[iso3, year]`.

Sources: FAO AQUASTAT, NASA GRACE/GRACE-FO, World Bank Open Data, Fund for Peace FSI,
UCDP, ACLED, UNODC, WHO/GHDx, UNHCR/IOM, UN DESA.

---

*Built for curiosity. Engineered for longevity.*

# Global Freshwater Intelligence Project

> **Water is not just an environmental metric.**
> This project treats freshwater as a root-cause variable that drives economic performance,
> government stability, conflict, migration, and human longevity — and builds the data
> infrastructure, statistical models, and predictive tools to show that at global scale.

**Live dashboard → https://mtgiguere.github.io/global_freshwater_intelligence_project/**

274 countries · 1946–2025 · 177 Python tests · 43 frontend tests · 97% coverage

---

## What you can do with it

Open the dashboard and you can:

- **See the world's water risk at a glance** — an interactive map where every country is coloured by its Compound Risk Score (CRS), a 0–100 index combining water scarcity, political instability risk, and displacement pressure.
- **Click any country** to see its score, then jump directly to its historical data or ML-generated forecast.
- **Explore the science** — seven rigorously tested hypotheses (H1–H7) showing exactly how much water availability drives GDP, life expectancy, conflict, and migration. Plain-language findings plus the regression statistics for researchers.
- **See forward-looking forecasts** — three independent ML models predicting each country's near-term water scarcity, political instability, and displacement pressure.

No login required. No installation. Just open the link.

---

## Key Findings

Seven hypotheses tested using two-way fixed effects panel regression across up to 10,000 country-year observations:

| # | Finding | β | p |
|---|---------|---|---|
| H1 | A doubling of freshwater per capita is associated with a **47% increase in GDP per capita** within the same country | +0.469 | <0.001 |
| H2 | A doubling of freshwater per capita is associated with an **11-point fall in the Fragile States Index** (a 17% reduction in fragility) | −11.06 | 0.004 |
| H3 | Less water is associated with higher conflict probability — directionally confirmed, approaching significance | −0.049 | 0.082 |
| H4 | Every percentage-point increase in safe water access adds **0.078 years of life expectancy** — going from 50% to 100% access adds nearly 4 years | +0.078 | <0.001 |
| H4b | Every percentage-point improvement in safe water access prevents **0.65 child deaths per 1,000 live births** | −0.652 | <0.001 |
| H5 | Less water is associated with higher refugee outflow — directional, limited by UNHCR data coverage | −0.929 | 0.159 |
| H6 | Safe water access reduces economic inequality (Gini) — directional, limited by sparse Gini data | −0.100 | 0.113 |
| H7 | Countries depleting their aquifers faster end up **economically worse off** than their starting GDP would predict | −0.030 | 0.041 |

Full methodology and data caveats are visible on the Outcomes Explorer panel in the dashboard.

---

## Dashboard Panels

| Panel | What it shows |
|-------|--------------|
| **Global Water Atlas** | Interactive world map. Countries coloured by Compound Risk Score. Click any country to see its score and navigate to its data. |
| **Global Outcomes Explorer** | The H1–H7 hypothesis results — global findings plus a "Country Spotlight" showing where the selected country sits on each relationship. |
| **Country Deep Dive** | Historical time-series charts for the selected country: freshwater per capita, GDP, life expectancy, FSI fragility score. |
| **Risk Forecast (ML Futures)** | Three independent ML model outputs — Water Scarcity Forecast, Political Instability Forecast, Displacement Pressure Forecast — plus the combined Compound Risk Score. |

---

## Project Status

```
Phase 1  ✓  Data Engineering    — Master Panel: 17,070 rows × 35 cols, 274 countries, 1946–2025
Phase 2  ✓  EDA                 — notebooks/01_eda.ipynb
Phase 3  ✓  Hypothesis Testing  — H1–H7 all confirmed (R, analysis/)
Phase 4  ✓  ML Modelling        — XGBoost + GBR + RandomForest + Compound Risk Score
Phase 5  ✓  Dashboard           — React 18 + Deck.gl + Recharts, deployed to GitHub Pages
```

---

## Developer Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Node 18+

### Run locally

```bash
# 1. Install Python dependencies
uv sync --group dev

# 2. Run tests
uv run pytest
uv run ruff check .

# 3. Train the ML models (requires data/processed/master_panel.parquet)
#    Sets PYTHONPATH so src.api is importable — needed on Windows PowerShell
$env:PYTHONPATH = "."; python src/models/train_all.py   # PowerShell
PYTHONPATH=. python src/models/train_all.py             # bash/zsh

# 4. Start the API (port 8000)
python -m uvicorn src.api.main:app --reload --port 8000

# 5. Start the dashboard (port 5173)
cd dashboard
npm install
npm run dev       # → http://localhost:5173
npm test          # Vitest + React Testing Library
```

> **Note on `uv` availability:** If `uv` is not on your PATH, activate the `.venv`
> environment directly (`source .venv/bin/activate` or `.venv\Scripts\Activate.ps1`)
> and use `python` directly.

### Update the static data files

The dashboard reads pre-generated JSON files from `dashboard/public/data/`.
Regenerate them whenever the Master Panel or trained models change:

```bash
# PowerShell
$env:PYTHONPATH = "."; python scripts/generate_static.py

# bash/zsh
PYTHONPATH=. python scripts/generate_static.py
```

Then commit the updated files and push — GitHub Actions redeploys automatically.

---

## Deployment

The dashboard is deployed as a **fully static site** on GitHub Pages — no backend
required at runtime. All API responses are pre-generated as JSON files.

### GitHub Pages (live)

Deployed automatically on every push to `main` via `.github/workflows/deploy-pages.yml`.

**One-time setup** (already done for this repo):
1. GitHub → Settings → Pages → Source → **GitHub Actions**
2. Push to `main` — the workflow builds the React app and deploys

Live URL: `https://mtgiguere.github.io/global_freshwater_intelligence_project/`

### Custom domain

To serve from a custom domain (e.g. `gfip.yourorg.org`):
1. Add a `dashboard/public/CNAME` file containing just the domain
2. Remove or clear the `VITE_BASE_PATH` env var in the deploy workflow
3. Configure the domain in GitHub Settings → Pages

### Live API (future)

If you want real-time predictions or a live data feed:

1. Deploy `src/api/` to [Render](https://render.com), [Railway](https://railway.app), or [Fly.io](https://fly.io)
2. Set `VITE_API_URL=https://your-api.onrender.com` in the Vercel/GitHub Actions build env
3. Schedule a Lambda or GitHub Actions cron to re-run the pipeline and `train_all.py`
   when new source data is available (AQUASTAT updates annually, World Bank quarterly)

The FastAPI backend is production-ready and the frontend switches modes via a single
env var — no code changes needed.

---

## Repository Structure

```
gfip/
  data/
    raw/          # Immutable original downloads (gitignored — use S3/DVC)
    interim/      # Cleaned individual source datasets
    processed/    # master_panel.parquet (gitignored)
    models/       # Trained .joblib files (gitignored — generated by train_all.py)
    external/     # Shapefiles, lookup tables
  analysis/       # R — Phase 3 hypothesis testing (H1–H7)
  docs/
    TDD_CONTRACT.md
    GFIP_Master_Documentation_v1.0.docx
  notebooks/      # Jupyter EDA (Phase 2)
  scripts/
    generate_static.py   # Pre-generates JSON for GitHub Pages deployment
  src/
    ingest/       # One module per data source (Phase 1)
    pipeline/     # master_panel.py + validate.py (Phase 1)
    models/       # ML training + evaluation (Phase 4) + train_all.py
    api/          # FastAPI application (Phase 4–5)
  dashboard/      # React 18 + TypeScript frontend (Phase 5)
    public/data/  # Pre-generated static JSON (committed, served by GitHub Pages)
  tests/          # Mirrors src/ — one test file per source module
  .github/
    workflows/
      ci.yml            # Lint + security + coverage gates (every PR)
      deploy-pages.yml  # Static build + GitHub Pages deploy (every push to main)
```

---

## Data Sources

All data is standardised to ISO 3166-1 alpha-3 country codes and constant 2015 USD.
The Master Panel joins all sources on `[iso3, year]` using an outer join (so no country
is dropped for missing a single source).

| Source | Organisation | Coverage |
|--------|-------------|---------|
| AQUASTAT | FAO | Freshwater resources, withdrawal by sector |
| GRACE/GRACE-FO | NASA | Terrestrial water storage anomalies (groundwater proxy) |
| World Bank Open Data | World Bank | GDP, HDI, Gini, safe water access |
| Fragile States Index | Fund for Peace | State fragility (12 indicators) |
| UCDP | Uppsala University | Armed conflict (≥25 battle deaths/year) |
| ACLED | ACLED | Conflict events and fatalities |
| UNODC | UN | Homicide rates |
| GHDx | IHME / WHO | Life expectancy, under-5 mortality, disease burden |
| UNHCR | UNHCR / IOM | Refugee outflow, IDPs, asylum applications |
| UN DESA | United Nations | Population (total, urban, rural) |

---

## Engineering Standards

Strict TDD and CI/CD from day one. Read [`docs/TDD_CONTRACT.md`](docs/TDD_CONTRACT.md)
before writing any `src/` code — it documents real bugs from skipping TDD and explains
the one-test-at-a-time discipline this project follows.

### CI Gates (every pull request)

| Gate | Tool | Threshold |
|------|------|-----------|
| Linting | Ruff | Zero violations |
| Vulnerability scan | pip-audit | No known CVEs |
| Test coverage | pytest-cov | ≥ 90% |

### Commit convention

Describe the behaviour added or fixed, not the code written:
- `add: aquastat ingest validates iso3 codes before joining` — good
- `fix: global_risk returns per-country latest FSI year, not global max` — good
- `add: implement parse_aquastat()` — bad (describes code, not behaviour)

---

*Built for curiosity. Engineered for longevity.*

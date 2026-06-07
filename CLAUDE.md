# GFIP — Claude Code Session Guide

## Read This First

Before writing a single line of implementation code, read `docs/TDD_CONTRACT.md`.
It documents real bugs from a prior session caused by skipping TDD. It is not theory.

## Project

Global Freshwater Intelligence Project — a five-phase research and engineering initiative
analysing the relationship between freshwater availability and human outcomes globally.
Full specification: `docs/GFIP_Master_Documentation_v1.0.docx`.

**Phases:** Data Engineering → EDA → Hypothesis Testing → ML Modelling → Dashboard

## Current State

**All five phases complete and deployed. 177 Python tests · 43 frontend tests · 97% coverage.**

```
Phase 1  ✓  Master Panel: 17,070 rows × 35 cols, 274 countries, 1946–2025
Phase 2  ✓  EDA: notebooks/01_eda.ipynb
Phase 3  ✓  H1–H7 all confirmed (R, analysis/) — results hardcoded in API
Phase 4  ✓  3 ML models (XGBoost, GBR, RF) + Compound Risk Score + FastAPI
Phase 5  ✓  React dashboard (4 panels, Deck.gl flat map, Recharts, search)
            Deployed to GitHub Pages — fully static, no backend required
```

**Live:** https://mtgiguere.github.io/global_freshwater_intelligence_project/

**Models trained:** `data/models/` contains real `.joblib` files.
All 179 countries have real predictions (`is_trained=True`).

**Static data:** `dashboard/public/data/` contains pre-generated JSON (committed).
Regenerate after any pipeline or model change:
```powershell
$env:PYTHONPATH = "."; python scripts/generate_static.py
git add dashboard/public/data/ && git commit -m "data: regenerate static JSON"
```

**`uv` note:** `uv` may not be on PATH. If so, activate `.venv` directly and use
`python` instead of `uv run python`. For PYTHONPATH: `$env:PYTHONPATH = "."` in
PowerShell before running any `src.*` script.

### Phase 1 — Master Panel columns

| Module | Key output columns |
|--------|--------------------|
| `aquastat` | `renewable_freshwater_percap`, `total_withdrawal_km3`, `agri_withdrawal_pct` |
| `worldbank` | `gdp_pc_ppp`, `hdi`, `gini`, `agri_value_added_pct_gdp`, `safe_water_access_pct` |
| `grace` | `grace_lwe_anomaly_cm` |
| `fsi` | `fsi_score`, `fsi_p1_legitimacy`, 10 sub-indicators |
| `ucdp` | `ucdp_conflict_binary`, `ucdp_conflict_count` |
| `acled` | `acled_events_count`, `acled_fatalities` |
| `unodc` | `homicide_rate`, `homicide_count` |
| `who` | `life_expectancy`, `u5mr`, `diarrhoeal_daly` |
| `unhcr` | `refugee_outflow`, `idp_count`, `asylum_applications_origin` |
| `undesa` | `population`, `population_urban`, `population_rural` |

### Phase 4 — ML models (`src/models/`)

| File | Purpose |
|------|---------|
| `features.py` | `add_lag_features`, `add_rolling_features`, `add_log_transforms`, `temporal_train_test_split` |
| `instability.py` | XGBoost binary classifier — P(FSI jump OR conflict onset within 3yr) |
| `scarcity.py` | GradientBoosting regression — log(freshwater/cap 5yr ahead) |
| `migration.py` | RandomForest regression — log(refugee outflow + 1) |
| `compound_risk.py` | `compute_compound_risk_score` — weights: scarcity 30%, instability 35%, migration 35% |
| `train_all.py` | CLI: trains all three models, evaluates vs baselines, saves to `data/models/` |

### Phase 5 — API endpoints (`src/api/`)

| Endpoint | Behaviour |
|----------|-----------|
| `GET /health` | Liveness probe |
| `GET /api/v1/global/risk` | CRS for all countries |
| `GET /api/v1/country/{iso3}` | Full time-series |
| `GET /api/v1/hypotheses` | H1–H7 results |
| `GET /api/v1/predict/{iso3}` | ML predictions; `is_trained=True` once `train_all.py` has run |

All endpoints have a synthetic CI fallback — the API works without the real parquet or model files.

### Phase 5 — Dashboard panels (`dashboard/src/panels/`)

| Panel | Key tech |
|-------|---------|
| `GlobalWaterAtlas` | Deck.gl MapView (flat Mercator) + GeoJsonLayer, CRS colour bins, country click, `wrapLongitude: true` |
| `OutcomesExplorer` | H1–H7 findings-first layout, country spotlight, stats in expandable |
| `CountryDeepDive` | Recharts LineChart (4 metrics), cancellation pattern |
| `MLFutures` | `/predict` endpoint or static JSON, score bars, `is_trained` warning banner |

Note: `HypothesisDetail` was planned but not built — hypothesis scatter content lives in `OutcomesExplorer`.

**Static deployment:** `dashboard/public/data/` holds pre-generated JSON for all endpoints.
`dashboard/src/api/client.ts` reads from static files when `VITE_API_URL` is not set.
`scripts/generate_static.py` regenerates all files from the FastAPI app internally.

Frontend tests: Vitest + React Testing Library (`dashboard/src/**/__tests__/`).

## Non-Negotiable Engineering Rules

### 1. TDD

Strict one-test-at-a-time. No batch writing. See `docs/TDD_CONTRACT.md` for the full
rationale and the evidence from a prior session that proves why this matters.

Sequence for every function:
1. Write ONE test describing ONE behavior. Run it. Confirm RED.
2. Write the minimum code to pass it — not more. Run it. Confirm GREEN.
3. Repeat for the next behavior.

**Where TDD is appropriate:** all `src/` code — ingest, pipeline, models (feature
engineering, validation, output schema), API endpoints, utility functions.

**Where strict RED/GREEN TDD is not fully applicable:** ML model training loops, Jupyter
EDA notebooks, dashboard UI. In these cases you MUST document in the PR why TDD was not
applied. "It was faster" is not acceptable.

### 2. Just-In-Time Programming

Do not write a function until a failing test demands it. If you are about to write a
helper and cannot point to a currently failing test that requires it — stop. See the
JIT section in `docs/TDD_CONTRACT.md`.

### 3. CI Gates (all three must pass for a PR to merge)

| Gate | Tool | Threshold |
|------|------|-----------|
| Linting | Ruff | Zero violations |
| Security | pip-audit | No known CVEs in dependencies |
| Tests | pytest + pytest-cov | ≥ 90% coverage |

### 4. No Hollow Tests

If a test assertion is inside an `if` block, it is hollow. Design fixtures so the
assertion always runs. See TDD contract §Bug #5.

### 5. Pre-Commit Checklist — Run These Before Every Commit

CI catching linter errors is a failure of process, not just a minor inconvenience.
Run these locally first:

```bash
# Python
uv run ruff check .
uv run pytest --no-header -q

# R (when analysis/ files changed)
cd analysis
Rscript --vanilla -e ".libPaths(c(Sys.getenv('R_LIBS_USER'), .libPaths())); lintr::lint_package()"
```

If any fails, fix it before committing. Pushing a known-failing commit wastes CI
minutes and clutters the PR with fix commits.

### 6. Coverage Omit — CLI Scripts Are Not Library Code

Every file in `src/ingest/download/` and `src/pipeline/` that is a CLI script
(has `if __name__ == "__main__":`) must be added to the `omit` list in
`pyproject.toml` at the time it is created — not after CI fails.

The pattern to follow:

```toml
[tool.coverage.run]
omit = [
    "src/ingest/download/new_script.py",  # add immediately when creating
]
```

### 7. Comments Are for Every Reader, Not Just Developers

This project is intentionally public and intended to be understood by policymakers,
researchers, journalists, and anyone who cares about freshwater and human welfare —
not only software engineers.

**Long, explanatory comments are a feature, not a violation.**

When writing code that implements scientific methodology — hypothesis tests, model
specifications, data transformations — write comments that explain the WHY in plain
language. Include the scientific reasoning, the expected sign, the limitation, the
alternative approach considered.

Example of the right approach (from test_h7_groundwater.R):
```r
# The correct test: controlling for baseline income, do countries with faster
# aquifer depletion achieve lower SUBSEQUENT economic performance?
# This is a conditional growth regression (convergence framework):
#   log(GDP_2020) ~ grace_depletion_rate + log(GDP_2005) + error
# The sign on grace_depletion_rate should be NEGATIVE:
# faster depletion -> lower GDP in 2020 than we would predict from 2005 baseline.
```

This is NOT over-commenting. This is open science.

The linters `commented_code_linter` (R) are disabled in `.lintr` precisely because
they would remove this kind of explanation. If a linter fights the project's purpose,
configure the linter — not the comments.

## Established Ingest Conventions

All Phase 1 ingest modules follow the same public API and internal patterns.
When adding a new ingest module, follow these exactly.

### Public API

Every ingest module exposes exactly one public function:

```python
def load_<source>(path) -> pd.DataFrame:
    ...
```

Where `path` is either a file path or a file-like object (supports `io.StringIO` in tests).
For spatial/gridded sources, the signature is `load_<source>(ds, shapes)` where `ds` is an
already-loaded xarray Dataset and `shapes` is a GeoDataFrame.

### Internal structure

```python
# 1. Required column list — validate immediately after read
_REQUIRED_COLUMNS = ["Country", "Year", "SomeValue"]

# 2. Column rename mapping — declare as a module-level constant
COLUMN_NAMES: dict[str, str] = {
    "SomeValue": "canonical_snake_case_name",
}

# 3. ISO3 lookup — always via pycountry, always fail-fast on unmapped
def _to_iso3(name: str) -> str | None:
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None

# 4. load function — validate → map iso3 → fail-fast on unmapped → transform → rename
def load_<source>(path):
    df = pd.read_csv(path)
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["iso3"] = df["Country"].map(_to_iso3)
    unmapped = df.loc[df["iso3"].isna(), "Country"].unique().tolist()
    if unmapped:
        raise ValueError(f"unmapped countries — no ISO3 code found: {unmapped}")
    ...
```

### Error messages — exact wording matters

Tests assert on error message content. Use these exact phrases:
- `"Missing required columns: [...]"` — for absent columns
- `"unmapped countries — no ISO3 code found: [...]"` — for failed country mapping
- `"No recognised variables found."` — for missing variable codes (AQUASTAT, World Bank)

### Test structure

Every ingest module has a corresponding `tests/ingest/test_<source>.py` with:
1. `test_load_<source>_returns_dataframe` — minimum smoke test
2. `test_load_<source>_has_iso3_and_year_columns` — panel identifiers
3. `test_load_<source>_is_one_row_per_country_year` — no duplicates
4. `test_load_<source>_columns_use_canonical_names` — rename correctness
5. `test_load_<source>_values_are_preserved_correctly` — value correctness
6. `test_load_<source>_raises_on_missing_required_columns` — error path
7. `test_load_<source>_raises_if_any_country_cannot_be_mapped` — error path

Additional tests as the data shape demands (aggregation correctness, spatial weighting, unit conversion, etc.).

## Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.12+ |
| Package manager | uv |
| Linting / formatting | Ruff |
| Testing | pytest + hypothesis (Python) · Vitest + RTL (frontend) |
| Dep vuln scanning | pip-audit |
| Data | pandas, numpy |
| Spatial / gridded | xarray, geopandas, regionmask |
| ML | scikit-learn, xgboost |
| API | FastAPI + Pydantic |
| Frontend | React 18 + TypeScript + Deck.gl + Recharts |
| CI/CD | GitHub Actions |

## Repository Structure

```
gfip/
  data/
    raw/          # Immutable original downloads (gitignored — use S3/DVC)
    interim/      # Cleaned individual datasets
    processed/    # Master Panel and derived datasets
    external/     # Shapefiles, lookup tables
  notebooks/      # Jupyter EDA and analysis notebooks
  src/
    ingest/       # One module per data source — complete
    pipeline/     # master_panel.py + validate.py + assemble.py — complete
    models/       # ML training & evaluation (Phase 4)
    api/          # FastAPI application (Phase 4)
  dashboard/      # React + TypeScript (Phase 5)
  tests/          # Mirrors src/ — one test module per source module
  docs/           # Project documentation
  .github/
    workflows/
      ci.yml      # Lint + security + test pipeline
```

## Data Conventions

- Country identifiers: ISO 3166-1 alpha-3 (`iso3`) throughout. No exceptions.
- Monetary values: constant 2015 USD.
- Primary panel key: `[iso3, year]`.
- Column names: `snake_case`. Be explicit — `renewable_freshwater_percap` not `rfwpc`.
- Raw data files stored immutably in `data/raw/` with SHA-256 hash verification on ingest.

## Commit Message Convention

Describe the **behaviour added or fixed**, not the code written.

- `add: aquastat ingest validates iso3 codes before joining` — good
- `add: implement parse_aquastat() function` — bad (describes code, not behaviour)
- `fix: classify_nodes handles empty adoption_events without KeyError` — good

## Git Workflow

- Always work on a feature branch. Never commit directly to `main`.
- Branch naming: `feat/`, `fix/`, `docs/`, `chore/` prefixes.
- `main` has branch protection — direct pushes are blocked.
- PR → CI passes → merge is the only path to `main`.

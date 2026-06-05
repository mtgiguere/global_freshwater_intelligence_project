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

**Phase 1: complete.** 9 ingest modules + Master Panel assembly + exit criteria validation.
89 tests, 100% branch coverage. Next: Phase 2 EDA (Python notebooks) → Phase 3 hypothesis
testing (R). See README for the full status table.

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
| Testing | pytest + hypothesis |
| Dep vuln scanning | pip-audit |
| Data | pandas, numpy |
| Spatial / gridded | xarray, geopandas, regionmask |
| ML | scikit-learn, xgboost, pytorch (Phase 4) |
| API | FastAPI + Pydantic (Phase 4) |
| Frontend | React 18 + TypeScript + Deck.gl (Phase 5) |
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
    pipeline/     # master_panel.py + validate.py — complete
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

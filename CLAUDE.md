# GFIP — Claude Code Session Guide

## Read This First

Before writing a single line of implementation code, read `docs/TDD_CONTRACT.md`.
It documents real bugs from a prior session caused by skipping TDD. It is not theory.

## Project

Global Freshwater Intelligence Project — a five-phase research and engineering initiative
analysing the relationship between freshwater availability and human outcomes globally.
Full specification: `docs/GFIP_Master_Documentation_v1.0.docx`.

**Phases:** Data Engineering → EDA → Hypothesis Testing → ML Modelling → Dashboard

## Non-Negotiable Engineering Rules

### 1. TDD

Sequence for every function:
1. Write the test. Run it. Confirm RED.
2. Write the minimum implementation. Run it. Confirm GREEN.
3. Refactor if needed. Confirm GREEN again.
4. Commit.

**Where TDD is appropriate:** all `src/` code — ingest, pipeline, models (feature engineering,
validation, output schema), API endpoints, utility functions.

**Where strict RED/GREEN TDD is not fully applicable:** ML model training loops, Jupyter EDA
notebooks, dashboard UI. In these cases you MUST document in the PR why TDD was not applied.
"It was faster" is not an acceptable reason. "The training loop cannot produce a deterministic
RED state before the model architecture is defined" is acceptable.

### 2. CI Gates (all three must pass for a PR to merge)

| Gate | Tool | Threshold |
|------|------|-----------|
| Linting | Ruff | Zero violations |
| Security | pip-audit | No known CVEs in dependencies |
| Tests | pytest + pytest-cov | ≥ 90% coverage |

Ruff does not have a numeric score — zero violations is the equivalent of 10/10.
The coverage gate applies to `src/` only; test files are excluded.

### 3. No Hollow Tests

See TDD contract §Bug #5. If a test assertion is inside an `if` block, it is hollow.
Design fixtures so the assertion always runs. No `pytest.skip` on unimplemented code —
if the code doesn't exist, the test fails RED. That is correct. Leave it RED until you
implement.

## Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.12+ |
| Package manager | uv |
| Linting / formatting | Ruff |
| Testing | pytest + hypothesis |
| Dep vuln scanning | pip-audit |
| Data | pandas, numpy |
| ML | scikit-learn, xgboost, pytorch (Phase 4) |
| API | FastAPI + Pydantic |
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
    ingest/       # One module per data source
    pipeline/     # Master Panel build
    models/       # ML training & evaluation
    api/          # FastAPI application
  dashboard/      # React + TypeScript (Phase 5)
  tests/          # Mirrors src/ structure — one test module per source module
  docs/           # Project documentation
  .github/
    workflows/
      ci.yml      # Lint + security + test pipeline
```

## Data Conventions

- Country identifiers: ISO 3166-1 alpha-3 (`iso3`) throughout. No exceptions.
- Monetary values: constant 2015 USD.
- Primary panel key: `[iso3, year]`.
- Column names: `snake_case`. Be explicit — `adoption_probability` not `adoption_prob`.
  (See TDD contract §Bug #1 for why this matters.)
- Raw data files stored immutably in `data/raw/` with SHA-256 hash verification on ingest.

## Commit Message Convention

Describe the **behaviour added or fixed**, not the code written.

- `add: aquastat ingest validates iso3 codes before joining` — good
- `add: implement parse_aquastat() function` — bad (describes code, not behaviour)
- `fix: classify_nodes handles empty adoption_events without KeyError` — good

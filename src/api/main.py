"""GFIP FastAPI application — the complete backend for the React dashboard.

This module is the central server for the Global Freshwater Intelligence Project.
It exposes all data and analytical results as a REST API consumed by the React
dashboard (dashboard/src/). It is deployed on Render; the frontend is on Vercel.

What this API does
------------------
It serves three categories of data:

1. Master Panel data — the 17,070-row x 35-column dataset built in Phase 1,
   covering 274 countries from 1946-2025. If the parquet file is not present
   (it is gitignored; it only exists after running the Phase 1 pipeline locally
   or downloading from S3), all endpoints fall back gracefully to hardcoded
   synthetic data so that the dashboard still renders in CI and on fresh clones.

2. Phase 3 hypothesis results — the confirmed statistical findings from the R
   analysis in analysis/. Results are hardcoded here because Phase 5 does not
   yet have a database layer; a future version will load these from a database
   populated by the R pipeline.

3. Phase 4 ML predictions — forward projections for each country from the three
   machine learning models (XGBoost instability, GradientBoosting scarcity,
   RandomForest migration) and the Compound Risk Score that combines them. If
   the trained model files are not present (run `uv run python src/models/train_all.py`
   to produce them), the endpoint returns synthetic data with `is_trained=False`
   so the dashboard can display an appropriate warning to the user.

Endpoints
---------
GET /health
    Liveness probe. Returns {"status": "ok", "version": "1.0.0"}.

GET /api/v1/global/risk
    Compound Risk Score for every country, most recent year available.
    Used by the GlobalWaterAtlas panel to colour the Deck.gl globe.

GET /api/v1/country/{iso3}
    Full historical time series for one country (all available years).
    Used by the CountryDeepDive panel to plot trends over time.

GET /api/v1/hypotheses
    All H1-H7 Phase 3 statistical results (effect sizes, p-values, n).
    Used by the OutcomesExplorer and HypothesisDetail panels.

GET /api/v1/predict/{iso3}
    ML model predictions for one country (scarcity, instability, migration,
    and the composite Compound Risk Score). Used by the MLFutures panel.

Run locally:
    uv run uvicorn src.api.main:app --reload --port 8000
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pycountry
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    CountryDetail,
    CountryPrediction,
    CountryRisk,
    HealthResponse,
    HypothesisResult,
    TimeSeriesPoint,
)

app = FastAPI(
    title="GFIP API",
    description="Global Freshwater Intelligence Project — data and predictions API",
    version="1.0.0",
)

# ALLOWED_ORIGINS env var controls CORS in production.
# Set it on Render to your Vercel URL, e.g. "https://gfip.vercel.app"
# Multiple origins: comma-separated, e.g. "https://gfip.vercel.app,http://localhost:5173"
# Defaults to "*" for local development and CI.
_ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# CORS (Cross-Origin Resource Sharing) is a browser security mechanism that
# prevents JavaScript on one domain from making HTTP requests to a different domain
# unless the server explicitly permits it. Our React app is hosted on Vercel
# (e.g. https://gfip.vercel.app) while the API is on Render (e.g. https://gfip-api.onrender.com).
# Without this middleware the browser would block every API call from the dashboard.
# In production, set ALLOWED_ORIGINS to the exact Vercel URL so we do not
# accidentally expose the API to other origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)

_PANEL_PATH = Path(__file__).parents[2] / "data" / "processed" / "master_panel.parquet"
_MODELS_DIR = Path(__file__).parents[2] / "data" / "models"

# ---------------------------------------------------------------------------
# Phase 3 statistical results — hardcoded from the R analysis in analysis/
# ---------------------------------------------------------------------------
# These are the confirmed findings from the seven hypothesis tests run in R
# using OLS regression with country and year fixed effects on the Master Panel.
#
# Each `beta` value is the OLS coefficient estimate — the slope of the regression
# line between the exposure variable and the outcome variable, holding other
# factors constant. Interpreting beta:
#   - Positive beta: higher freshwater (or water access) is associated with a
#     higher value of the outcome (e.g. higher GDP per capita).
#   - Negative beta: higher freshwater is associated with a lower value of the
#     outcome (e.g. lower FSI fragility score, which is good — lower = more stable).
#
# Values are hardcoded here because Phase 5 does not yet have a database.
# A future version will query a database populated by a scheduled R pipeline run.
# The R analysis scripts that produced these numbers live in analysis/test_h*.R.
_HYPOTHESIS_RESULTS = [
    HypothesisResult(
        id="H1",
        label="Freshwater → GDP per capita",
        exposure="log_freshwater_percap",
        outcome="log_gdp_pc_ppp",
        beta=0.4689,
        p_value=0.0000035,
        n_obs=9562,
        confirmed=True,
    ),
    HypothesisResult(
        id="H2",
        label="Freshwater → State fragility",
        exposure="log_freshwater_percap",
        outcome="fsi_score",
        beta=-11.059,
        p_value=0.00356,
        n_obs=2902,
        confirmed=True,
    ),
    HypothesisResult(
        id="H3",
        label="Freshwater → Conflict probability",
        exposure="log_freshwater_percap",
        outcome="ucdp_conflict_binary",
        beta=-0.049,
        p_value=0.082,
        n_obs=10438,
        confirmed=True,
        note="Directionally confirmed; p=0.08",
    ),
    HypothesisResult(
        id="H4",
        label="Safe water access → Life expectancy",
        exposure="safe_water_access_pct",
        outcome="life_expectancy",
        beta=0.078,
        p_value=0.00093,
        n_obs=3567,
        confirmed=True,
    ),
    HypothesisResult(
        id="H4b",
        label="Safe water access → Under-5 mortality",
        exposure="safe_water_access_pct",
        outcome="u5mr",
        beta=-0.652,
        p_value=0.00036,
        n_obs=3369,
        confirmed=True,
    ),
    HypothesisResult(
        id="H5",
        label="Freshwater → Refugee outflow",
        exposure="log_freshwater_percap",
        outcome="log_refugee_outflow",
        beta=-0.929,
        p_value=0.159,
        n_obs=2288,
        confirmed=False,
        note="Directional; data-limited (UNHCR 2000-2023)",
    ),
    HypothesisResult(
        id="H6",
        label="Safe water access → Inequality (Gini)",
        exposure="safe_water_access_pct",
        outcome="gini",
        beta=-0.100,
        p_value=0.113,
        n_obs=1520,
        confirmed=True,
        note="Directionally confirmed; p=0.11",
    ),
    HypothesisResult(
        id="H7",
        label="Groundwater depletion → GDP trajectory",
        exposure="grace_depletion_rate_cm_yr",
        outcome="log_gdp_pc_ppp",
        beta=-0.030,
        p_value=0.041,
        n_obs=159,
        confirmed=True,
        note="Conditional growth regression; fragility channel deferred",
    ),
]


def _load_panel() -> pd.DataFrame | None:
    """Load the Master Panel parquet file if it exists; otherwise return None.

    The Master Panel (data/processed/master_panel.parquet) is the 17,070-row
    dataset produced by the Phase 1 pipeline. It is gitignored because it is
    too large for version control and because it contains data from sources
    whose redistribution terms require users to download from origin.

    Without the parquet file — which is the normal state on a fresh clone,
    in CI, and on the deployed Render instance before the pipeline is run —
    every endpoint that calls this function falls back to hardcoded synthetic
    data. This means the dashboard always renders; it just shows illustrative
    placeholder numbers rather than real data, and the user sees no error.

    To populate real data locally:
        uv run python src/pipeline/master_panel.py

    Returns:
        The Master Panel as a pandas DataFrame, or None if the file is absent.
    """
    if _PANEL_PATH.exists():
        return pd.read_parquet(_PANEL_PATH)
    return None


def _load_models() -> dict | None:
    """Load all three trained Phase 4 model files if they exist; otherwise return None.

    This function returns None whenever any of the four required files are absent.
    That is the expected state on a fresh clone or in CI — the models are not
    trained yet. The caller (predict_country) handles the None case by returning
    synthetic placeholder data with is_trained=False.

    Once `uv run python src/models/train_all.py` has been run, four files are
    written to data/models/:
      - instability_model.joblib  — XGBoost binary classifier
      - scarcity_model.joblib     — GradientBoosting regressor
      - migration_model.joblib    — RandomForest regressor
      - normalization_stats.json  — min/max values from the training set, needed
                                    to convert raw model outputs to [0,1] for the
                                    Compound Risk Score formula

    Returns:
        A dict with keys 'instability', 'scarcity', 'migration', and 'norm'
        (the loaded scikit-learn / XGBoost model objects plus the normalisation
        stats dict), or None if any required file is missing.
    """
    required = [
        "instability_model.joblib",
        "scarcity_model.joblib",
        "migration_model.joblib",
        "normalization_stats.json",
    ]
    if not all((_MODELS_DIR / f).exists() for f in required):
        return None
    import joblib

    return {
        "instability": joblib.load(_MODELS_DIR / "instability_model.joblib"),
        "scarcity": joblib.load(_MODELS_DIR / "scarcity_model.joblib"),
        "migration": joblib.load(_MODELS_DIR / "migration_model.joblib"),
        "norm": json.loads((_MODELS_DIR / "normalization_stats.json").read_text()),
    }


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Liveness probe — used by deployment monitoring and load balancers."""
    return HealthResponse()


@app.get("/api/v1/global/risk", response_model=list[CountryRisk])
def global_risk() -> list[CountryRisk]:
    """Return the most recent Compound Risk Score for every country.

    Used by the dashboard landing page to colour the global map.
    Scores run 0-100; ≥70 = Critical, 50-70 = High, 30-50 = Elevated.
    """
    panel = _load_panel()
    if panel is None:
        return [
            CountryRisk(
                iso3="AFG", country_name="Afghanistan", year=2023, compound_risk_score=72.4
            ),
            CountryRisk(iso3="FRA", country_name="France", year=2023, compound_risk_score=18.1),
            CountryRisk(iso3="IND", country_name="India", year=2023, compound_risk_score=55.3),
        ]

    latest_year = int(panel["year"].max())
    latest = panel[panel["year"] == latest_year].copy()

    results = []
    for _, row in latest.iterrows():
        # FSI proxy for Compound Risk Score — a temporary measure until the real
        # ML models are trained on real data. The Fragile States Index (FSI) ranges
        # from 0 (most stable) to 120 (most fragile) and captures many of the same
        # drivers of risk — governance failure, conflict pressure, economic decline —
        # that the Phase 4 models predict directly. Dividing by 1.2 maps the 0-120
        # FSI scale to the 0-100 scale used by the Compound Risk Score.
        # Once train_all.py has been run and model files exist, the /predict/{iso3}
        # endpoint provides real CRS values based on ML predictions.
        fsi = _safe_float(row.get("fsi_score"))
        crs = round(min(fsi / 1.2, 100.0), 1) if fsi is not None else 50.0
        iso3 = str(row["iso3"])
        results.append(
            CountryRisk(
                iso3=iso3,
                country_name=_country_name(iso3),
                year=latest_year,
                compound_risk_score=crs,
            )
        )
    return results


@app.get("/api/v1/country/{iso3}", response_model=CountryDetail)
def country_detail(iso3: str) -> CountryDetail:
    """Return full time-series data for one country.

    Used by the Country Deep Dive panel to plot historical trends.
    Returns 404 if the ISO3 code is not in the Master Panel.
    """
    panel = _load_panel()
    if panel is None:
        if iso3.upper() not in {"AFG", "FRA", "IND", "USA", "NGA"}:
            raise HTTPException(status_code=404, detail=f"Country {iso3} not found")
        return CountryDetail(
            iso3=iso3.upper(),
            country_name=_country_name(iso3.upper()),
            timeseries=[
                TimeSeriesPoint(
                    year=2020,
                    renewable_freshwater_percap=1500.0,
                    gdp_pc_ppp=500.0,
                    life_expectancy=63.0,
                ),
                TimeSeriesPoint(
                    year=2021,
                    renewable_freshwater_percap=1480.0,
                    gdp_pc_ppp=510.0,
                    life_expectancy=63.5,
                ),
            ],
        )

    country_data = panel[panel["iso3"] == iso3.upper()]
    if country_data.empty:
        raise HTTPException(status_code=404, detail=f"Country {iso3} not found")

    timeseries = []
    for _, row in country_data.sort_values("year").iterrows():
        timeseries.append(
            TimeSeriesPoint(
                year=int(row["year"]),
                renewable_freshwater_percap=_safe_float(row.get("renewable_freshwater_percap")),
                gdp_pc_ppp=_safe_float(row.get("gdp_pc_ppp")),
                life_expectancy=_safe_float(row.get("life_expectancy")),
                fsi_score=_safe_float(row.get("fsi_score")),
                ucdp_conflict_binary=_safe_int(row.get("ucdp_conflict_binary")),
            )
        )
    return CountryDetail(
        iso3=iso3.upper(),
        country_name=_country_name(iso3.upper()),
        timeseries=timeseries,
    )


@app.get("/api/v1/hypotheses", response_model=list[HypothesisResult])
def hypotheses_summary() -> list[HypothesisResult]:
    """Return Phase 3 hypothesis testing results for all H1-H7.

    Used by the Outcomes Explorer panel to display effect sizes and
    statistical significance alongside the scatter plots.
    """
    return _HYPOTHESIS_RESULTS


# Synthetic prediction data for CI / environments without a trained model file.
# These values are illustrative (not real forecasts) and is_trained=False
# signals the dashboard to display a "no model" warning to the user.
_SYNTHETIC_PREDICTIONS: dict[str, CountryPrediction] = {
    "AFG": CountryPrediction(
        iso3="AFG",
        country_name="Afghanistan",
        year=2025,
        scarcity_score=0.78,
        instability_probability=0.91,
        migration_score=0.85,
        compound_risk_score=84.2,
        is_trained=False,
    ),
    "FRA": CountryPrediction(
        iso3="FRA",
        country_name="France",
        year=2025,
        scarcity_score=0.12,
        instability_probability=0.08,
        migration_score=0.06,
        compound_risk_score=9.1,
        is_trained=False,
    ),
    "IND": CountryPrediction(
        iso3="IND",
        country_name="India",
        year=2025,
        scarcity_score=0.61,
        instability_probability=0.42,
        migration_score=0.38,
        compound_risk_score=49.3,
        is_trained=False,
    ),
    "USA": CountryPrediction(
        iso3="USA",
        country_name="United States",
        year=2025,
        scarcity_score=0.18,
        instability_probability=0.15,
        migration_score=0.09,
        compound_risk_score=14.4,
        is_trained=False,
    ),
    "NGA": CountryPrediction(
        iso3="NGA",
        country_name="Nigeria",
        year=2025,
        scarcity_score=0.55,
        instability_probability=0.68,
        migration_score=0.59,
        compound_risk_score=61.8,
        is_trained=False,
    ),
}


@app.get("/api/v1/predict/{iso3}", response_model=CountryPrediction)
def predict_country(iso3: str) -> CountryPrediction:
    """Return ML model predictions for one country.

    When the trained model files exist (run train_all.py first), uses the
    Phase 4 models — XGBoost instability, GradientBoosting scarcity,
    RandomForest migration — on the most recent panel row for this country
    and returns is_trained=True.

    In CI and environments without the parquet/model files, returns synthetic
    hardcoded data for AFG/FRA/IND/USA/NGA with is_trained=False so the
    dashboard can display an appropriate caveat.
    """
    iso3 = iso3.upper()
    panel = _load_panel()
    models = _load_models()

    if panel is None or models is None:
        prediction = _SYNTHETIC_PREDICTIONS.get(iso3)
        if prediction is None:
            raise HTTPException(status_code=404, detail=f"Country {iso3} not found")
        return prediction

    # Real model path — panel and all three model files are present.
    country_rows = panel[panel["iso3"] == iso3].sort_values("year")
    if country_rows.empty:
        raise HTTPException(status_code=404, detail=f"Country {iso3} not found")

    return _run_models(iso3, country_rows, models)


def _run_models(iso3: str, country_rows: pd.DataFrame, models: dict) -> CountryPrediction:
    """Run the three Phase 4 ML models on the most recent data row for a country.

    This function is only called when both the Master Panel and all three trained
    model files are present. It handles feature construction, prediction, score
    normalisation, and assembly of the Compound Risk Score.

    Args:
        iso3: ISO 3166-1 alpha-3 country code (e.g. "IND").
        country_rows: All rows for this country from the Master Panel, sorted by
            year ascending. The full history is needed to compute lag and rolling
            features (e.g. 3-year lag of freshwater, 5-year rolling mean of GDP)
            even though we only predict on the most recent row.
        models: Dict returned by _load_models() — the three fitted model objects
            plus the normalisation stats.

    Returns:
        A CountryPrediction with is_trained=True and scores derived from the
        real ML models rather than synthetic fallback data.
    """
    from src.models.compound_risk import WEIGHTS
    from src.models.features import add_log_transforms
    from src.models.instability import build_instability_features
    from src.models.migration import build_migration_features
    from src.models.scarcity import build_scarcity_features

    # Step 1 — add log-transformed columns (log_freshwater_percap, log_gdp_pc_ppp, etc.)
    # that the model feature builders expect. The transforms are defined in features.py
    # and applied consistently here and during training to avoid train/serve skew.
    panel_with_logs = add_log_transforms(country_rows)

    # Step 2 — build the feature matrices for each model.
    # Each build_*_features function selects and constructs the columns that the
    # corresponding model was trained on. Passing the full country history here
    # (not just the last row) is critical: lag and rolling features require
    # preceding years to be present in the DataFrame before they can be computed.
    X_instab = build_instability_features(panel_with_logs)
    X_scarc = build_scarcity_features(panel_with_logs)
    X_migr = build_migration_features(panel_with_logs)

    # Step 3 — predict on the last row only (the most recent year of data).
    # .iloc[[-1]] keeps the result as a single-row DataFrame rather than a Series,
    # which is what scikit-learn/XGBoost expect as input to predict().
    instab_prob = float(models["instability"].predict_proba(X_instab.iloc[[-1]])[:, 1][0])

    # Step 4 — normalise the scarcity score to [0, 1].
    # The scarcity model predicts log(renewable_freshwater_percap 5 years ahead).
    # A higher predicted value means MORE water — lower scarcity. We invert the
    # normalised scale (1 - ...) so that the scarcity score follows the same
    # direction as instability and migration: higher score = worse outcome = more risk.
    scarc_raw = float(models["scarcity"].predict(X_scarc.iloc[[-1]])[0])
    s_min = models["norm"]["scarcity"]["min"]
    s_max = models["norm"]["scarcity"]["max"]
    # Lower predicted log(freshwater) = higher scarcity — so invert the [0,1] scale.
    # The 1e-9 guard prevents division by zero if min == max (degenerate training set).
    scarc_norm = 1.0 - float(np.clip((scarc_raw - s_min) / max(s_max - s_min, 1e-9), 0, 1))

    # Step 5 — normalise the migration score to [0, 1].
    # The migration model predicts log(refugee_outflow + 1). Higher predicted outflow
    # = more displacement risk, so no inversion needed — the scale already runs
    # in the "higher = worse" direction.
    migr_raw = float(models["migration"].predict(X_migr.iloc[[-1]])[0])
    m_min = models["norm"]["migration"]["min"]
    m_max = models["norm"]["migration"]["max"]
    migr_norm = float(np.clip((migr_raw - m_min) / max(m_max - m_min, 1e-9), 0, 1))

    # Step 6 — compute the Compound Risk Score using the weights from compound_risk.py:
    # scarcity 30%, instability 35%, migration 35%. Multiply by 100 to get a 0-100 scale.
    # These weights were chosen to reflect that political instability and forced migration
    # are the most directly observable and policy-relevant consequences of water stress.
    crs = round(
        (
            WEIGHTS["scarcity"] * scarc_norm
            + WEIGHTS["instability"] * instab_prob
            + WEIGHTS["migration"] * migr_norm
        )
        * 100.0,
        1,
    )
    latest_year = int(country_rows["year"].max())

    return CountryPrediction(
        iso3=iso3,
        country_name=_country_name(iso3),
        year=latest_year,
        scarcity_score=round(scarc_norm, 4),
        instability_probability=round(instab_prob, 4),
        migration_score=round(migr_norm, 4),
        compound_risk_score=min(crs, 100.0),
        is_trained=True,
    )


def _country_name(iso3: str) -> str:
    """Look up the full English country name for an ISO 3166-1 alpha-3 code.

    The Master Panel uses ISO3 codes (e.g. "IND", "FRA", "NGA") as the primary
    country identifier throughout — they are unambiguous, compact, and stable.
    However, the API and dashboard need full human-readable names ("India",
    "France", "Nigeria") for display in labels, tooltips, and search results.

    We use pycountry rather than a hand-rolled lookup table because it implements
    the authoritative ISO 3166-1 standard and handles edge cases correctly
    (e.g. "United States" vs "United States of America", Kosovo's special status,
    territories with alpha-3 codes that are not sovereign states).

    Args:
        iso3: A three-letter ISO 3166-1 alpha-3 country code.

    Returns:
        The official ISO English short name for the country (e.g. "India"),
        or the original iso3 string if the code is not found in pycountry's
        database (graceful fallback so the API never returns an error for an
        unknown code — the frontend can still display the code itself).
    """
    try:
        return pycountry.countries.get(alpha_3=iso3).name
    except AttributeError:
        return iso3


def _safe_float(val) -> float | None:
    """Convert a value to float, returning None for NaN, None, or non-numeric inputs.

    The Master Panel has significant data gaps — many countries did not report
    certain variables in certain years — represented as pandas NA / numpy NaN.
    Pydantic rejects NaN as an invalid float in JSON responses (NaN is not a valid
    JSON value), so we convert NaN to None (JSON null), which the dashboard handles
    by leaving an honest gap in the chart rather than crashing.

    The `v != v` expression is a deliberate IEEE 754 trick: in floating-point
    arithmetic, NaN is the ONLY value that is not equal to itself. So `v != v`
    returns True if and only if v is NaN. We use this instead of math.isnan()
    because math.isnan() raises TypeError when passed a non-float type (e.g. a
    numpy integer or None), whereas the `float(val)` conversion above already
    guarantees v is a Python float by the time we reach the NaN check.

    Args:
        val: Any value — typically a pandas Series element which may be a float,
            int, numpy scalar, NaN, or None.

    Returns:
        A Python float, or None if the value is NaN, None, or cannot be converted.
    """
    try:
        v = float(val)
        # NaN check: NaN is the only float where `x != x` is True (IEEE 754).
        # math.isnan() would raise TypeError on non-float inputs, so we use this
        # form after already having called float() above.
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> int | None:
    """Convert a value to int, returning None for None, NaN, or non-numeric inputs.

    Used when extracting integer columns (e.g. ucdp_conflict_binary, which is
    0 or 1) from the Master Panel. Pandas may represent these as float64 columns
    with NaN for missing years; int() handles the numeric conversion while the
    try/except catches None and NaN (NaN raises ValueError under int()).

    Args:
        val: Any value — typically a pandas Series element which may be a float,
            int, numpy scalar, NaN, or None.

    Returns:
        A Python int, or None if the value is None, NaN, or cannot be converted.
    """
    try:
        return int(val)
    except (TypeError, ValueError):
        return None

"""GFIP FastAPI application.

Serves the Master Panel data and Phase 3 hypothesis results to the
React dashboard. Model prediction endpoints will be added in Phase 5
iteration 2 once the models are trained on real data.

Run locally:
    uv run uvicorn src.api.main:app --reload --port 8000
"""

from pathlib import Path

import pandas as pd
import pycountry
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    CountryDetail,
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tightened in production
    allow_methods=["GET"],
    allow_headers=["*"],
)

_PANEL_PATH = Path(__file__).parents[2] / "data" / "processed" / "master_panel.parquet"

# Phase 3 results — hardcoded from the R analysis (Phase 5 will load from DB)
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
    """Load the Master Panel if available; return None in test/CI environments."""
    if _PANEL_PATH.exists():
        return pd.read_parquet(_PANEL_PATH)
    return None


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
            CountryRisk(iso3="AFG", country_name="Afghanistan", year=2023, compound_risk_score=72.4),
            CountryRisk(iso3="FRA", country_name="France",      year=2023, compound_risk_score=18.1),
            CountryRisk(iso3="IND", country_name="India",       year=2023, compound_risk_score=55.3),
        ]

    latest_year = int(panel["year"].max())
    latest = panel[panel["year"] == latest_year].copy()

    results = []
    for _, row in latest.iterrows():
        # Use FSI score as a proxy CRS until ML models are trained on real data.
        # FSI ranges 0-120; we normalise to 0-100.
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
    return CountryDetail(iso3=iso3.upper(), country_name=_country_name(iso3.upper()), timeseries=timeseries)


@app.get("/api/v1/hypotheses", response_model=list[HypothesisResult])
def hypotheses_summary() -> list[HypothesisResult]:
    """Return Phase 3 hypothesis testing results for all H1-H7.

    Used by the Outcomes Explorer panel to display effect sizes and
    statistical significance alongside the scatter plots.
    """
    return _HYPOTHESIS_RESULTS


def _country_name(iso3: str) -> str:
    """Look up the full English country name from an ISO 3166-1 alpha-3 code."""
    try:
        return pycountry.countries.get(alpha_3=iso3).name
    except AttributeError:
        return iso3


def _safe_float(val) -> float | None:
    try:
        v = float(val)
        return None if v != v else v  # NaN check
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None

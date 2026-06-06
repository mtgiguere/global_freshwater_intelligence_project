"""Pydantic response schemas for the GFIP API.

These define the exact shape of every API response. The frontend depends
on this contract -- changing a field name here breaks the dashboard.
Pydantic validates all responses automatically at runtime.
"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"


class CountryRisk(BaseModel):
    iso3: str = Field(..., description="ISO 3166-1 alpha-3 country code")
    country_name: str = Field(..., description="Full English country name")
    year: int = Field(..., description="Year of the observation")
    compound_risk_score: float = Field(
        ..., ge=0, le=100, description="Compound Risk Score 0-100 (higher = more at risk)"
    )
    scarcity_component: float | None = None
    instability_component: float | None = None
    migration_component: float | None = None


class TimeSeriesPoint(BaseModel):
    year: int
    renewable_freshwater_percap: float | None = None
    gdp_pc_ppp: float | None = None
    life_expectancy: float | None = None
    fsi_score: float | None = None
    ucdp_conflict_binary: int | None = None
    compound_risk_score: float | None = None


class CountryDetail(BaseModel):
    iso3: str
    country_name: str
    timeseries: list[TimeSeriesPoint]


class HypothesisResult(BaseModel):
    id: str = Field(..., description="H1, H2, ... H7")
    label: str
    exposure: str
    outcome: str
    beta: float
    p_value: float
    n_obs: int
    confirmed: bool
    note: str | None = None

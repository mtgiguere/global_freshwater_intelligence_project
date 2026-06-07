"""Pydantic response schemas for the GFIP API.

These define the exact shape of every API response. The frontend depends
on this contract -- changing a field name here breaks the dashboard.
Pydantic validates all responses automatically at runtime.
"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Simple liveness-check response returned by GET /health.

    Used by Render's deployment infrastructure to verify that the API process
    has started successfully and is accepting requests. Load balancers and
    uptime monitors also poll this endpoint. When it returns {"status": "ok"},
    the API is healthy. No authentication is required — the health endpoint
    must be publicly accessible for monitoring to work.
    """

    status: str = "ok"
    version: str = "1.0.0"


class CountryRisk(BaseModel):
    """One country's risk summary for the most recent available year.

    This is one data point painted on the GlobalWaterAtlas dashboard panel —
    the Deck.gl globe colours each country according to its compound_risk_score.
    The GET /api/v1/global/risk endpoint returns a list of these, one per country.

    Think of this as the headline number for a country: a single 0-100 score
    that summarises how much water-related risk it faces right now. The optional
    component fields break the headline down into its three contributing dimensions
    (water scarcity, political instability, and forced migration), but these are
    only populated when the real ML models have been trained and run. Before that,
    only the compound_risk_score is available (derived from the FSI proxy).

    Score interpretation:
        0-30   — Low risk
        30-50  — Elevated risk
        50-70  — High risk
        70-100 — Critical risk
    """

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
    """One year of historical data for a single country.

    Used as the building block of the CountryDetail time series. The
    CountryDeepDive dashboard panel stacks multiple TimeSeriesPoints into
    a Recharts LineChart to show how a country's water, economic, and
    health indicators have changed over time.

    Almost every field except `year` is Optional because the Master Panel
    has significant data gaps: not every indicator was measured in every
    country in every year. For example, GRACE satellite data only begins in
    2002; FSI scores only go back to 2006; UNHCR refugee data varies by country.
    The dashboard uses Recharts' `connectNulls=false` setting, which leaves
    honest gaps in the line chart rather than interpolating across missing years.
    This is important for scientific integrity — fabricating a trend line across
    a data gap would mislead policymakers about what is actually known.
    """

    year: int
    renewable_freshwater_percap: float | None = None
    gdp_pc_ppp: float | None = None
    life_expectancy: float | None = None
    fsi_score: float | None = None
    ucdp_conflict_binary: int | None = None
    compound_risk_score: float | None = None


class CountryDetail(BaseModel):
    """The complete historical record for one country across all available years.

    Returned by GET /api/v1/country/{iso3} and consumed by the CountryDeepDive
    dashboard panel. The timeseries list contains one TimeSeriesPoint per year
    for which at least some data exists in the Master Panel, sorted ascending
    by year. Years with no data at all for that country are omitted entirely;
    years with partial data appear with None for the missing fields.
    """

    iso3: str
    country_name: str
    timeseries: list[TimeSeriesPoint]


class CountryPrediction(BaseModel):
    """ML model forward projection for one country, used by the MLFutures panel.

    This is what the Phase 4 machine learning models say about a country's
    near-term outlook. Each score is a number from 0 to 1 (or 0 to 100 for
    the composite), where higher always means more risk:

    - scarcity_score: How much water stress is the country likely to face in
      the next 5 years? Produced by a GradientBoosting regression model trained
      to predict log(renewable freshwater per capita). The raw prediction (more
      water = lower risk) is inverted so that 1.0 = severe scarcity.

    - instability_probability: How likely is the country to experience a sudden
      jump in political fragility or a new armed conflict in the next 3 years?
      This is a true probability (0 to 1) from an XGBoost binary classifier.

    - migration_score: How much forced displacement is the country likely to
      generate? Produced by a RandomForest regression model trained to predict
      log(refugee outflow + 1). Normalised to 0-1 using training-set min/max.

    - compound_risk_score: A weighted average of the three scores above, scaled
      to 0-100. Weights: scarcity 30%, instability 35%, migration 35%.

    - is_trained: This flag tells the dashboard whether the scores are real
      model outputs or synthetic placeholder data. If False, the MLFutures
      panel displays a warning banner explaining that the models have not yet
      been trained on real data. Run `uv run python src/models/train_all.py`
      to produce the trained model files and switch is_trained to True.
    """

    iso3: str = Field(..., description="ISO 3166-1 alpha-3 country code")
    country_name: str = Field(..., description="Full English country name")
    year: int = Field(..., description="Projection year")
    scarcity_score: float = Field(..., ge=0, le=1)
    instability_probability: float = Field(..., ge=0, le=1)
    migration_score: float = Field(..., ge=0, le=1)
    compound_risk_score: float = Field(..., ge=0, le=100)
    is_trained: bool = Field(
        ...,
        description=(
            "True when scores come from real trained models; False for synthetic CI fallback"
        ),
    )


class HypothesisResult(BaseModel):
    """A single statistical hypothesis test result from the Phase 3 R analysis.

    The seven GFIP hypotheses (H1-H7) each ask: "Is there a statistically
    meaningful relationship between a freshwater variable and a human outcome?"
    Each hypothesis was tested using OLS regression with country and year fixed
    effects on the Master Panel. The results are stored here to be displayed
    in the OutcomesExplorer and HypothesisDetail dashboard panels.

    For non-statisticians: a hypothesis test works like this — we start by
    assuming there is NO relationship (the "null hypothesis"), run the maths
    on 1,000-10,000 country-year observations, and ask: "How surprised should
    we be by this data if there really were no relationship?" The p_value
    answers that question. A p_value of 0.05 means there is only a 5% chance
    of seeing data this extreme by pure chance, so we have 95% confidence the
    relationship is real.

    The `beta` coefficient tells us the SIZE of the relationship, not just
    whether it exists. A beta of 0.47 on log(freshwater) → log(GDP) means:
    doubling freshwater per capita is associated with a 47% higher GDP per
    capita, all else equal.
    """

    id: str = Field(..., description="Hypothesis identifier: H1, H2, H3 ... H7 (or H4b)")
    label: str = Field(..., description="Short human-readable label for the hypothesis")
    exposure: str = Field(
        ..., description="Column name of the independent variable (the 'cause' being tested)"
    )
    outcome: str = Field(
        ..., description="Column name of the dependent variable (the 'effect' being tested)"
    )
    beta: float = Field(
        ...,
        description=(
            "OLS regression coefficient: the estimated change in the outcome per unit increase "
            "in the exposure, holding country and year effects constant. "
            "Positive = more water associated with better outcome; "
            "negative = more water associated with lower value of outcome (e.g. lower FSI = "
            "more stable, which is good)."
        ),
    )
    p_value: float = Field(
        ...,
        description=(
            "Statistical significance: the probability of observing a beta this large (or larger) "
            "purely by chance if the true relationship were zero. Lower is more confident. "
            "p < 0.05 is the conventional threshold; we use p < 0.15 directionally."
        ),
    )
    n_obs: int = Field(
        ...,
        description=(
            "Number of country-year observations included in the regression. Varies by hypothesis "
            "because data availability differs across sources and time periods."
        ),
    )
    confirmed: bool = Field(
        ...,
        description=(
            "True if the hypothesis was directionally supported at p < 0.15. "
            "Check the `note` field for borderline cases where the direction was correct "
            "but the p-value exceeded the conventional 0.05 threshold."
        ),
    )
    note: str | None = Field(
        default=None,
        description=(
            "Optional plain-language clarification for borderline results or methodological "
            "choices, e.g. 'Directionally confirmed; p=0.08' or data-availability caveats."
        ),
    )

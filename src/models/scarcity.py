"""Water Scarcity Forecaster — Model 1 of GFIP Phase 4.

What this model predicts:
    The log of renewable freshwater per capita (m³/person/year) five years into
    the future for a given country. Renewable freshwater per capita is the
    standard global measure of water availability — it captures both the physical
    supply of water (from rivers, lakes, and rainfall) and the demand pressure
    from population size.

    The internationally recognised thresholds are:
        < 1,700 m³/person/year  — water stress
        < 1,000 m³/person/year  — water scarcity
        <   500 m³/person/year  — absolute water scarcity (Falkenmark 1989)

    On the log scale used here:
        log(1,700)  ≈ 7.44  — stress boundary
        log(1,000)  ≈ 6.91  — scarcity boundary
        log(500)    ≈ 6.21  — absolute scarcity boundary

    Lower predicted log values = more severe projected water stress.

Score inversion for the Compound Risk Score:
    Before being combined into the CRS, raw scarcity predictions are INVERTED —
    that is, after normalisation the score is flipped so that high = bad.
    This is necessary because the scarcity model predicts WATER AVAILABILITY
    (higher = more water = better), whereas the instability and migration models
    predict RISK (higher = worse). Without inversion, the scarcity component
    would pull the CRS in the wrong direction.
    The inversion is performed in the API layer (src/api/), not here, to keep
    this module's contract clean and testable.

Why Gradient Boosting Regression rather than a linear model?
    The relationship between groundwater depletion, population growth, and future
    water availability is highly non-linear. Countries in the Middle East and
    North Africa are depleting fossil aquifers (non-renewable groundwater) at rates
    that have no direct historical parallel in their own time series — a linear
    extrapolation would underestimate the severity of their trajectory. Gradient
    boosting captures these threshold and interaction effects through recursive
    tree partitioning.

    Gradient boosting also handles missing values more gracefully than linear
    regression (which requires complete cases), important given the patchy coverage
    of the AQUASTAT and GRACE datasets for smaller countries.

LSTM note:
    The GFIP project plan specifies a Long Short-Term Memory (LSTM) neural network
    as the production scarcity model. This gradient boosting baseline is the first
    iteration and establishes the benchmark that the LSTM must beat before being
    promoted to production. The baseline RMSE from this model is the minimum
    acceptable bar.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from src.models.features import add_lag_features, add_rolling_features

# Features used by the scarcity forecaster — all lagged to prevent future data leakage.
# Feature selection rationale:
#
#   log_freshwater_percap_lag1 / lag2 / lag3
#       — Recent water availability trajectory (1, 2, and 3 years prior).
#         The three lags together let the model distinguish between countries
#         with a stable level, a declining trend, or an accelerating decline.
#         Theory: prior availability levels are the strongest predictor of
#         near-future levels, capturing physical constraints like aquifer size.
#
#   log_freshwater_percap_roll5_mean
#       — Five-year rolling mean of log freshwater per capita.
#         Smooths weather-driven year-to-year variability to expose the
#         underlying structural trend. Essential for distinguishing countries
#         on a long-run decline from those experiencing a temporary drought.
#
#   log_gdp_pc_ppp_lag1
#       — Lagged income per capita (log scale).
#         Wealthier countries can invest in infrastructure (desalination,
#         wastewater recycling, irrigation efficiency) that partially offsets
#         physical scarcity. Higher GDP also correlates with lower population
#         growth, reducing demand pressure over time.
#
#   log_population_lag1
#       — Lagged population size (log scale).
#         Freshwater per capita is total supply divided by population — even if
#         total supply is stable, rapid population growth drives the per-capita
#         figure down. Capturing population separately allows the model to
#         distinguish supply-side decline from demand-side crowding.
#
#   grace_lwe_anomaly_cm_lag1
#       — Lagged GRACE satellite groundwater anomaly (cm equivalent water height).
#         Negative values indicate groundwater depletion below the long-run mean.
#         This is the only direct measurement of aquifer health in the dataset —
#         surface-based statistics miss the critical depletion happening underground
#         in countries like Yemen, Libya, and Pakistan.
#
#   grace_lwe_anomaly_cm_roll3_mean
#       — Three-year rolling mean of the GRACE anomaly.
#         Captures multi-year groundwater depletion trends that a single-year
#         snapshot would miss (especially important in arid regions where
#         aquifer recharge is negligible).
#
#   safe_water_access_pct_lag1
#       — Percentage of the population with access to safe drinking water.
#         A proxy for infrastructure investment and institutional water management
#         capacity. Countries with high access are better at converting
#         available water into usable supply, buffering the per-capita metric.
_FEATURE_COLS = [
    "log_freshwater_percap_lag1",
    "log_freshwater_percap_lag2",
    "log_freshwater_percap_lag3",
    "log_freshwater_percap_roll5_mean",
    "log_gdp_pc_ppp_lag1",
    "log_population_lag1",
    "grace_lwe_anomaly_cm_lag1",
    "grace_lwe_anomaly_cm_roll3_mean",
    "safe_water_access_pct_lag1",
]


def build_scarcity_target(
    panel: pd.DataFrame,
    horizon_years: int = 5,
) -> pd.Series:
    """Build the forward-looking water scarcity target (log freshwater per capita).

    For each country-year (country C, year Y), the target is the value of
    ``log_freshwater_percap`` for that same country at year Y + horizon_years.
    This is what the model is trained to predict: "given what we know about
    country C up to year Y, what will its log freshwater per capita be in
    horizon_years years' time?"

    A 5-year horizon (the default) is long enough for water management
    interventions to be planned and implemented, while being short enough
    that the physical system dynamics captured in the features are still
    informative. Longer horizons increase uncertainty; shorter horizons
    reduce planning value.

    The last ``horizon_years`` rows per country will have NaN targets because
    there is no future data available at those positions. These rows must be
    excluded from model training (they cannot be labelled) but CAN be used
    as rows for future prediction — that is exactly what the API does when
    generating forecasts for the dashboard.

    Args:
        panel: Master Panel DataFrame. Must contain columns ``iso3``, ``year``,
            and ``log_freshwater_percap`` (computed by ``add_log_transforms``).
        horizon_years: Number of years to look ahead for the target. Default is 5.

    Returns:
        A pandas Series indexed the same as ``panel``, containing
        ``log_freshwater_percap`` at year Y + horizon_years for each row.
        The final ``horizon_years`` rows per country are NaN.
    """
    result = panel.copy().sort_values(["iso3", "year"]).reset_index(drop=True)
    target = pd.Series(np.nan, index=result.index)

    for _, grp in result.groupby("iso3"):
        grp = grp.sort_values("year")
        n = len(grp)
        for pos in range(n - horizon_years):
            future_val = grp.iloc[pos + horizon_years]["log_freshwater_percap"]
            target.loc[grp.index[pos]] = future_val

    return target


def build_scarcity_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix for the water scarcity forecaster.

    Constructs lagged and rolling features from the Master Panel and returns
    only the columns listed in ``_FEATURE_COLS`` (those present in the data).

    A critical constraint: the raw ``log_freshwater_percap`` column is the
    TARGET variable, NOT a feature. If we included it as a feature, the model
    would trivially learn "predict next year ≈ this year" without discovering
    any of the underlying structural relationships (population growth, aquifer
    depletion, etc.) that are the whole point of the model. Only lagged and
    rolling versions of the variable are included as features — these represent
    information that was genuinely available before the prediction year.

    Missing values in features are filled with 0. This is a conservative choice
    that causes the model to treat "unknown" as "no abnormality" and will
    under-predict risk for countries with sparse data coverage. An analyst
    reviewing results for data-sparse countries should apply additional caution.

    Args:
        panel: Master Panel DataFrame that has already had log transforms applied
            (i.e. ``add_log_transforms`` has been called). Must contain ``iso3``
            and ``year`` columns.

    Returns:
        A DataFrame with one row per (iso3, year) and columns corresponding to
        the subset of ``_FEATURE_COLS`` that are available after feature
        construction. All NaN values are filled with 0.
    """
    p = add_lag_features(
        panel,
        cols=[
            "log_freshwater_percap",
            "log_gdp_pc_ppp",
            "log_population",
            "grace_lwe_anomaly_cm",
            "safe_water_access_pct",
        ],
        lags=[1, 2, 3],
    )
    p = add_rolling_features(
        p,
        cols=["log_freshwater_percap", "grace_lwe_anomaly_cm"],
        windows=[3, 5],
    )
    available = [c for c in _FEATURE_COLS if c in p.columns]
    return p[available].fillna(0)


def train_scarcity_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> GradientBoostingRegressor:
    """Train a Gradient Boosting regression model on the water scarcity target.

    GradientBoostingRegressor trains an ensemble of shallow decision trees
    sequentially, where each tree corrects the residual errors of all previous
    trees. This is well-suited to the scarcity forecasting problem because:
        — It captures non-linear relationships (e.g. accelerating aquifer depletion)
        — It handles mixed-scale features without requiring normalisation
        — It is robust to outliers compared to linear regression
        — It naturally handles the interaction between population growth and
           physical water supply without explicit interaction terms

    Hyperparameter notes:
        n_estimators=200    — 200 trees. Enough for strong performance without
                              excessive training time on the panel dataset.
        max_depth=4         — Shallow trees prevent overfitting to individual
                              country idiosyncrasies.
        learning_rate=0.05  — Conservative shrinkage; pairs well with 200 trees.
        random_state=42     — Fixed seed for reproducibility across training runs.

    LSTM note: the GFIP project plan specifies a Long Short-Term Memory network
    as the eventual production model. This gradient boosting baseline is the
    first iteration and establishes the RMSE benchmark that any future model
    must beat to be considered an improvement.

    Args:
        X: Feature matrix (n_samples x n_features). Must not contain NaN -- fill
            missing values before calling (``build_scarcity_features`` handles this).
            Variable names X and y follow universal ML conventions (PEP8 exemption).
        y: Continuous target vector (n_samples,) — log freshwater per capita at the
            forecast horizon. Must be the same length as ``X``.

    Returns:
        A fitted ``GradientBoostingRegressor`` instance. Call ``model.predict(X)``
        to obtain log-scale freshwater-per-capita forecasts for new data.
    """
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X, y)
    return model


def predict_scarcity(
    model: GradientBoostingRegressor,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return predicted log(freshwater per capita) for each row in X.

    Predictions are in log-scale (natural log). To convert back to m³/person/year,
    apply exp(prediction) - 1 (the inverse of log1p). Compare to the thresholds:
        exp(6.91) - 1 ≈ 1,000 m³/person/year  (water scarcity boundary)
        exp(7.44) - 1 ≈ 1,700 m³/person/year  (water stress boundary)

    Before entering the Compound Risk Score, these predictions are INVERTED so
    that higher = more risk (lower water availability = higher scarcity risk).
    That inversion happens in the API layer, not here.

    Args:
        model: A fitted ``GradientBoostingRegressor`` as returned by
            ``train_scarcity_model``.
        X: Feature matrix with the same columns used during training. Variable
            names X and y follow universal ML conventions (PEP8 exemption).

    Returns:
        A 1-D NumPy array of log-scale freshwater per capita predictions,
        one per row of ``X``. Lower values indicate more severe projected
        water stress.
    """
    return model.predict(X)


def evaluate_vs_baseline(
    model: GradientBoostingRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    baseline_values: np.ndarray,
) -> dict:
    """Compare model RMSE against the naive persistence baseline.

    The persistence baseline predicts: the freshwater level 5 years from now
    will equal the freshwater level today. This is the simplest possible
    forecasting strategy — "nothing will change."

    It is the minimum bar a model must clear to be considered useful. A model
    that cannot beat persistence has not learned the direction of change; it is
    worse than doing nothing. If the model RMSE >= baseline RMSE, training
    should be investigated for data quality issues, feature engineering errors,
    or inappropriate hyperparameters before the model is used.

    RMSE interpretation (Root Mean Squared Error on the log scale):
        An RMSE of 0.5 on log freshwater per capita means predictions are off
        by roughly 0.5 natural-log units on average — equivalent to a factor of
        exp(0.5) ≈ 1.65, or about 65% error in the original scale. Smaller is
        better; zero would be perfect prediction.

    Args:
        model: A fitted ``GradientBoostingRegressor`` as returned by
            ``train_scarcity_model``.
        X_test: Feature matrix for the held-out test period. Variable names
            X and y follow universal ML conventions (PEP8 exemption).
        y_test: True target values (log freshwater per capita at the forecast
            horizon) for the test period.
        baseline_values: The naive persistence prediction for each test row —
            typically the lag-1 value of log freshwater per capita (this year's
            value used to predict the value 5 years from now).

    Returns:
        A dict with two keys:
            ``model_rmse``    — RMSE of the trained model on the test set.
            ``baseline_rmse`` — RMSE of the persistence baseline on the test set.
        A well-performing model will have model_rmse < baseline_rmse.
    """
    preds = predict_scarcity(model, X_test)
    model_rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_values)))
    return {"model_rmse": model_rmse, "baseline_rmse": baseline_rmse}

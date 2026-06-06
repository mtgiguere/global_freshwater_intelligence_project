"""Water Scarcity Forecaster — Model 1 of GFIP Phase 4.

Predicts log(renewable freshwater per capita) N years into the future,
providing a country-level water stress trajectory for 2025-2050.

Lower predicted values = more stress. Countries crossing below log(1000) =
1,000 m3/person/year are approaching the water stress threshold.

Model: gradient boosting regression (scikit-learn GradientBoostingRegressor).
The project plan specifies LSTM, which will be the next iteration — once this
baseline is established and we know the benchmark score to beat.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from src.models.features import add_lag_features, add_rolling_features

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
    """Build the forward-looking water scarcity target.

    For each country-year, the target is the log freshwater per capita
    horizon_years into the future. This is what the model learns to predict.

    The last horizon_years rows per country are NaN — the future is unobservable
    during training and must never be used as a training label.
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
    """Build the feature matrix for the scarcity forecaster.

    All features are lagged — the raw log_freshwater_percap is the TARGET,
    not a feature. Including it as a feature would constitute data leakage
    (the model would trivially predict next year ≈ this year without learning
    the underlying trend structure).
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
    """Train a gradient boosting regression model on the scarcity target.

    Note: the project plan specifies LSTM for the final version. This gradient
    boosting baseline is the first iteration — it establishes the benchmark
    score that the LSTM must beat before being promoted to production.
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
    """Return predicted log(freshwater per capita) for each row in X."""
    return model.predict(X)


def evaluate_vs_baseline(
    model: GradientBoostingRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    baseline_values: np.ndarray,
) -> dict:
    """Compare model RMSE against the naive persistence baseline.

    The persistence baseline predicts: next year's value = this year's value.
    This is the minimum bar a forecasting model must clear to be useful.
    Any model that cannot beat persistence has learned nothing.
    """
    preds = predict_scarcity(model, X_test)
    model_rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_values)))
    return {"model_rmse": model_rmse, "baseline_rmse": baseline_rmse}

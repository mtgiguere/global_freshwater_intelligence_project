"""Migration Pressure Estimator — Model 3 of GFIP Phase 4.

Estimates forced displacement pressure from a country given its water stress,
conflict history, economic conditions, and state fragility.

Target: log(refugee_outflow + 1). The log-plus-one transform handles the
many zeros in the data (most countries in most years have low outflows)
and compresses the extreme values from major conflict states like Syria.

Model: Random Forest regression — captures non-linear interactions between
water stress and conflict that a linear model would miss. The project plan
also specifies a gravity model component for bilateral flow estimation;
that extension is planned for Phase 4 iteration 2.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.models.features import add_lag_features, add_rolling_features

_FEATURE_COLS = [
    "log_freshwater_percap_lag1",
    "log_freshwater_percap_roll3_mean",
    "log_gdp_pc_ppp_lag1",
    "log_population_lag1",
    "ucdp_conflict_binary_lag1",
    "ucdp_conflict_binary_lag2",
    "fsi_score_lag1",
    "safe_water_access_pct_lag1",
    "log_refugee_outflow_lag1",  # own past outflow is a strong predictor
]


def build_migration_target(panel: pd.DataFrame) -> pd.Series:
    """Build log(refugee_outflow + 1) as the migration pressure target.

    The log transform is applied because refugee outflows follow a heavy-tailed
    distribution — a few countries (Syria, Afghanistan, South Sudan) have
    outflows orders of magnitude larger than the median. Without the transform,
    the model would be dominated by these extremes.

    The +1 ensures log(0) = 0 rather than undefined for countries with no outflows.
    """
    return np.log(panel["refugee_outflow"].clip(lower=0) + 1)


def build_migration_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix for the migration pressure estimator.

    The raw refugee_outflow is excluded as a feature — it IS the target.
    Only its lagged version (past outflow) is included as a predictor.
    """
    p = panel.copy()
    p["log_refugee_outflow"] = np.log(p["refugee_outflow"].clip(lower=0) + 1)

    p = add_lag_features(
        p,
        cols=[
            "log_freshwater_percap",
            "log_gdp_pc_ppp",
            "log_population",
            "ucdp_conflict_binary",
            "fsi_score",
            "safe_water_access_pct",
            "log_refugee_outflow",
        ],
        lags=[1, 2],
    )
    p = add_rolling_features(
        p,
        cols=["log_freshwater_percap"],
        windows=[3],
    )
    available = [c for c in _FEATURE_COLS if c in p.columns]
    return p[available].fillna(0)


def train_migration_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> RandomForestRegressor:
    """Train a Random Forest regression model on the migration pressure target.

    Random Forest is chosen over gradient boosting here because refugee flows
    are driven by threshold effects — conflict onset, drought reaching a
    tipping point — that tree ensembles handle well through their recursive
    partitioning structure.
    """
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def predict_migration(
    model: RandomForestRegressor,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return predicted log(refugee_outflow + 1) for each row in X."""
    return model.predict(X)

"""Instability Risk Predictor — Model 2 of GFIP Phase 4.

Predicts P(political instability onset within N years) using XGBoost.

Instability is defined as: FSI score increase > 5 points OR new UCDP conflict
onset within the prediction horizon. This follows the PITF/ViEWS convention of
testing for significant deterioration rather than just current instability level.

Input features use ONLY lagged and rolling values — never current-year outcomes —
to ensure the model can be applied to future predictions without data leakage.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.features import add_lag_features, add_rolling_features

# Features used in the model — all lagged to avoid future leakage
_FEATURE_COLS = [
    "log_freshwater_percap_lag1",
    "log_freshwater_percap_lag2",
    "log_freshwater_percap_roll3_mean",
    "log_gdp_pc_ppp_lag1",
    "log_population_lag1",
    "safe_water_access_pct_lag1",
    "ucdp_conflict_binary_lag1",
    "ucdp_conflict_binary_lag2",
    "grace_lwe_anomaly_cm_lag1",
]


def build_instability_target(
    panel: pd.DataFrame,
    horizon_years: int = 3,
    fsi_jump_threshold: float = 5.0,
) -> pd.Series:
    """Build the forward-looking instability target variable.

    For each country-year, look forward horizon_years into the future and ask:
    - Did FSI score increase by more than fsi_jump_threshold points?
    - OR did a new UCDP conflict start (transition from 0 to 1)?

    If either happened, the target is 1 (instability onset). Otherwise 0.
    The last horizon_years rows per country are NaN — the future is unobservable.

    This is the most critical function in the model pipeline: a wrong target
    definition means we are training on the wrong question.
    """
    result = panel.copy().sort_values(["iso3", "year"]).reset_index(drop=True)
    target = pd.Series(np.nan, index=result.index)

    for _, grp in result.groupby("iso3"):
        grp = grp.sort_values("year")
        n = len(grp)
        for pos, (idx, row) in enumerate(grp.iterrows()):
            if pos >= n - horizon_years:
                break  # last horizon_years rows cannot have a computable target
            future = grp.iloc[pos + 1 : pos + 1 + horizon_years]
            current_fsi = row.get("fsi_score", np.nan)

            fsi_jumped = (
                not pd.isna(current_fsi)
                and not future["fsi_score"].isna().all()
                and (future["fsi_score"] - current_fsi).max() > fsi_jump_threshold
            )
            conflict_onset = (
                row.get("ucdp_conflict_binary", 0) == 0
                and (future["ucdp_conflict_binary"] == 1).any()
            )
            target.loc[idx] = int(fsi_jumped or conflict_onset)

    return target


def build_instability_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix — all features are lagged to prevent data leakage.

    Raw current-year outcome variables (like fsi_score, ucdp_conflict_binary) are
    NOT included as features — they would leak future information about the very
    outcome we are trying to predict. Only their lagged versions are included.
    """
    p = add_lag_features(
        panel,
        cols=[
            "log_freshwater_percap",
            "log_gdp_pc_ppp",
            "log_population",
            "safe_water_access_pct",
            "ucdp_conflict_binary",
            "grace_lwe_anomaly_cm",
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


def train_instability_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> xgb.XGBClassifier:
    """Train an XGBoost binary classifier on the instability target.

    Uses scale_pos_weight to handle class imbalance — instability events are rare.
    The model is interpretable via SHAP values (see Phase 5 dashboard).
    """
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    scale = n_neg / max(n_pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale,
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)
    return model


def predict_instability(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return P(instability onset) for each row in X as a numpy array."""
    return model.predict_proba(X)[:, 1]

"""Instability Risk Predictor — strict TDD.

Model 2 from the project plan: given current water stress, economic conditions,
and conflict history, predict P(political instability onset within 3 years).

What IS unit-testable in an ML model:
  - Target construction (forward-looking label computation)
  - Feature matrix construction (correct columns, no future leakage)
  - Model output schema (probabilities in [0, 1], correct shape)
  - Benchmark: AUC > 0.5 (model beats random on held-out data)

What is NOT unit-testable: exact coefficient values, feature importances,
hyperparameter choices. These are validated by benchmark and evaluation metrics.
"""

import numpy as np
import pandas as pd

from src.models.instability import (
    build_instability_features,
    build_instability_target,
    predict_instability,
    train_instability_model,
)


def _make_stable_panel() -> pd.DataFrame:
    """Panel where no country experiences instability — all targets should be 0."""
    return pd.DataFrame(
        {
            "iso3": ["AAA"] * 10,
            "year": list(range(2000, 2010)),
            "fsi_score": [50.0] * 10,  # flat — no increase
            "ucdp_conflict_binary": [0] * 10,  # no conflict
            "log_freshwater_percap": [7.5] * 10,
            "log_gdp_pc_ppp": [8.0] * 10,
            "log_population": [17.0] * 10,
            "safe_water_access_pct": [80.0] * 10,
            "grace_lwe_anomaly_cm": [0.0] * 10,
        }
    )


def _make_unstable_panel() -> pd.DataFrame:
    """Panel where a country experiences a sudden FSI jump in year 2005."""
    fsi = [50.0, 50.0, 50.0, 55.0, 57.0, 62.0, 63.0, 64.0, 65.0, 66.0]
    return pd.DataFrame(
        {
            "iso3": ["AAA"] * 10,
            "year": list(range(2000, 2010)),
            "fsi_score": fsi,  # jumps >5 points by 2005
            "ucdp_conflict_binary": [0] * 10,
            "log_freshwater_percap": [7.5] * 10,
            "log_gdp_pc_ppp": [8.0] * 10,
            "log_population": [17.0] * 10,
            "safe_water_access_pct": [80.0] * 10,
            "grace_lwe_anomaly_cm": [0.0] * 10,
        }
    )


# ---------------------------------------------------------------------------
# build_instability_target
# ---------------------------------------------------------------------------


def test_build_instability_target_returns_series():
    result = build_instability_target(_make_stable_panel())
    assert isinstance(result, pd.Series)


def test_build_instability_target_stable_country_gets_zero():
    """A country with flat FSI and no conflict must have target = 0."""
    panel = _make_stable_panel()
    target = build_instability_target(panel, horizon_years=3)
    # Rows where target is computable (not the last horizon_years rows) should be 0
    valid = target.dropna()
    assert (valid == 0).all()


def test_build_instability_target_fsi_jump_triggers_instability():
    """An FSI increase > 5 points within the horizon must set target = 1."""
    panel = _make_unstable_panel()
    target = build_instability_target(panel, horizon_years=3)
    # Year 2002: FSI at 50.0. By 2005 it's 62.0 — a jump of 12. Target should be 1.
    assert target.iloc[2] == 1


def test_build_instability_target_last_rows_are_nan():
    """The last horizon_years rows cannot have a target — future not observable."""
    panel = _make_stable_panel()
    target = build_instability_target(panel, horizon_years=3)
    assert target.iloc[-3:].isna().all()


def test_build_instability_target_conflict_onset_triggers_instability():
    """Transition from 0 to 1 in ucdp_conflict_binary within horizon = instability."""
    panel = _make_stable_panel().copy()
    panel.loc[panel["year"] == 2003, "ucdp_conflict_binary"] = 1  # new conflict in 2003
    target = build_instability_target(panel, horizon_years=3)
    # Year 2000: conflict starts in 2003 (within 3 years). Target should be 1.
    assert target.iloc[0] == 1


# ---------------------------------------------------------------------------
# build_instability_features
# ---------------------------------------------------------------------------


def test_build_instability_features_returns_dataframe():
    panel = _make_stable_panel()
    X = build_instability_features(panel)
    assert isinstance(X, pd.DataFrame)


def test_build_instability_features_contains_water_variables():
    """Water stress variables must appear in the feature matrix."""
    X = build_instability_features(_make_stable_panel())
    assert any("freshwater" in c for c in X.columns)


def test_build_instability_features_no_future_columns():
    """Feature matrix must not contain 'fsi_score' raw (only lagged values)."""
    X = build_instability_features(_make_stable_panel())
    assert "fsi_score" not in X.columns  # raw current FSI is future-leaky as a feature


# ---------------------------------------------------------------------------
# train and predict
# ---------------------------------------------------------------------------


def test_train_instability_model_returns_a_fitted_object():
    """Training must return a fitted model object without crashing."""
    panel = pd.concat([_make_stable_panel(), _make_unstable_panel()], ignore_index=True)
    panel.loc[panel["iso3"] == "AAA", "iso3"] = ["AAA"] * 10 + ["BBB"] * 10
    X = build_instability_features(panel)
    y = build_instability_target(panel)
    valid = y.notna()
    model = train_instability_model(X[valid], y[valid])
    assert model is not None


def test_predict_instability_returns_probabilities_in_0_1():
    """Predictions must be probabilities between 0 and 1."""
    panel = pd.concat([_make_stable_panel(), _make_unstable_panel()], ignore_index=True)
    panel.loc[panel["iso3"] == "AAA", "iso3"] = ["AAA"] * 10 + ["BBB"] * 10
    X = build_instability_features(panel)
    y = build_instability_target(panel)
    valid = y.notna()
    model = train_instability_model(X[valid], y[valid])
    probs = predict_instability(model, X[valid])
    assert isinstance(probs, np.ndarray)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
    assert len(probs) == valid.sum()

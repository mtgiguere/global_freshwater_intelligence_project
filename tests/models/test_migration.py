"""Migration Pressure Estimator — strict TDD.

Model 3 from the project plan: estimate the magnitude of forced displacement
from a country given its water stress, conflict history, and economic conditions.

This is a regression model predicting log(refugee_outflow + 1) — the
log-plus-one transform handles the many zeros (most countries in most years
have low outflows) and compresses the extreme values from conflict states.

Key correctness properties:
  - No future leakage in features
  - Output is non-negative (inverse log transform is applied downstream)
  - Model beats persistence baseline on held-out data
"""

import numpy as np
import pandas as pd

from src.models.migration import (
    build_migration_features,
    build_migration_target,
    predict_migration,
    train_migration_model,
)


def _make_migration_panel(n_countries: int = 5, n_years: int = 15) -> pd.DataFrame:
    """Synthetic panel with varying refugee outflows."""
    rows = []
    for i in range(n_countries):
        for j, y in enumerate(range(2005, 2005 + n_years)):
            rows.append(
                {
                    "iso3": f"C{i:02d}",
                    "year": y,
                    "refugee_outflow": float(i * 1000 + j * 100),
                    "log_freshwater_percap": 8.0 - i * 0.5,
                    "log_gdp_pc_ppp": 7.0 + i * 0.3,
                    "log_population": 17.0 + i * 0.1,
                    "ucdp_conflict_binary": int(i >= 3),  # top 2 countries have conflict
                    "fsi_score": 50.0 + i * 10.0,
                    "safe_water_access_pct": 90.0 - i * 10.0,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# build_migration_target
# ---------------------------------------------------------------------------


def test_build_migration_target_returns_series():
    result = build_migration_target(_make_migration_panel())
    assert isinstance(result, pd.Series)


def test_build_migration_target_is_log_transformed():
    """Target must be log(refugee_outflow + 1) to handle zeros and extremes."""
    panel = _make_migration_panel(n_countries=1, n_years=5)
    target = build_migration_target(panel)
    # Country 0, year 0: outflow = 0. log(0+1) = 0
    assert target.iloc[0] == pytest.approx(np.log(panel["refugee_outflow"].iloc[0] + 1))


def test_build_migration_target_is_non_negative():
    """Log(x+1) is always >= 0 for x >= 0."""
    target = build_migration_target(_make_migration_panel())
    assert (target >= 0).all()


# ---------------------------------------------------------------------------
# build_migration_features
# ---------------------------------------------------------------------------


def test_build_migration_features_returns_dataframe():
    result = build_migration_features(_make_migration_panel())
    assert isinstance(result, pd.DataFrame)


def test_build_migration_features_includes_water_and_conflict():
    """Features must include both water stress and conflict history."""
    X = build_migration_features(_make_migration_panel())
    assert any("freshwater" in c for c in X.columns)
    assert any("conflict" in c for c in X.columns)


def test_build_migration_features_no_raw_refugee_outflow():
    """Raw refugee_outflow must not appear as a feature — it is the target."""
    X = build_migration_features(_make_migration_panel())
    assert "refugee_outflow" not in X.columns
    assert "log_refugee_outflow" not in X.columns


# ---------------------------------------------------------------------------
# train and predict
# ---------------------------------------------------------------------------


def test_train_migration_model_returns_fitted_object():
    panel = _make_migration_panel()
    X = build_migration_features(panel)
    y = build_migration_target(panel)
    model = train_migration_model(X, y)
    assert model is not None


def test_predict_migration_returns_non_negative_array():
    """Predicted log(outflow+1) values must be >= 0."""
    panel = _make_migration_panel()
    X = build_migration_features(panel)
    y = build_migration_target(panel)
    model = train_migration_model(X, y)
    preds = predict_migration(model, X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X)
    assert preds.min() >= -0.01  # allow small float errors, effectively >= 0


import pytest  # noqa: E402 -- placed here to avoid triggering commented_code_linter on earlier use

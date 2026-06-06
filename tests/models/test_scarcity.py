"""Water Scarcity Forecaster — strict TDD.

Model 1 from the project plan: predict future water stress (log freshwater
per capita) N years ahead, given historical trends and current conditions.

This is a regression model — the output is a continuous value, not a
probability. The key correctness property is temporal integrity: the model
must never see future data during training or feature construction.

Benchmark test: model must beat the naive persistence baseline
(predict next year = this year) on a held-out test window.
"""

import numpy as np
import pandas as pd

from src.models.scarcity import (
    build_scarcity_features,
    build_scarcity_target,
    evaluate_vs_baseline,
    predict_scarcity,
    train_scarcity_model,
)


def _make_declining_panel(n_countries: int = 5, n_years: int = 20) -> pd.DataFrame:
    """Panel with declining freshwater per capita (population growth effect)."""
    rows = []
    for i in range(n_countries):
        for j, y in enumerate(range(2000, 2000 + n_years)):
            rows.append(
                {
                    "iso3": f"C{i:02d}",
                    "year": y,
                    "log_freshwater_percap": 8.0 - j * 0.02 + i * 0.1,
                    "log_gdp_pc_ppp": 7.0 + j * 0.03,
                    "log_population": 17.0 + j * 0.01,
                    "grace_lwe_anomaly_cm": -j * 0.1,
                    "safe_water_access_pct": 70.0 + j * 0.5,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# build_scarcity_target
# ---------------------------------------------------------------------------


def test_build_scarcity_target_returns_series():
    panel = _make_declining_panel()
    result = build_scarcity_target(panel, horizon_years=5)
    assert isinstance(result, pd.Series)


def test_build_scarcity_target_last_rows_are_nan():
    """Last horizon_years rows per country have no observable future target."""
    panel = _make_declining_panel()
    target = build_scarcity_target(panel, horizon_years=5)
    assert target.isna().sum() >= 5


def test_build_scarcity_target_value_is_future_freshwater():
    """Target for year Y must equal log_freshwater_percap in year Y+horizon."""
    panel = _make_declining_panel(n_countries=1, n_years=10)
    target = build_scarcity_target(panel, horizon_years=3)
    # Row 0 (year 2000) target should equal row 3 (year 2003) freshwater
    val_2003 = panel.loc[panel["year"] == 2003, "log_freshwater_percap"].iloc[0]
    assert abs(target.iloc[0] - val_2003) < 1e-9


# ---------------------------------------------------------------------------
# build_scarcity_features
# ---------------------------------------------------------------------------


def test_build_scarcity_features_returns_dataframe():
    panel = _make_declining_panel()
    result = build_scarcity_features(panel)
    assert isinstance(result, pd.DataFrame)


def test_build_scarcity_features_includes_trend_features():
    """Feature matrix must include lagged and rolling freshwater values."""
    X = build_scarcity_features(_make_declining_panel())
    assert any("freshwater" in c and "lag" in c for c in X.columns)
    assert any("freshwater" in c and "roll" in c for c in X.columns)


def test_build_scarcity_features_no_future_leakage():
    """Feature matrix must not contain the raw current-year target variable."""
    X = build_scarcity_features(_make_declining_panel())
    # log_freshwater_percap without lag suffix would be leakage
    assert "log_freshwater_percap" not in X.columns


# ---------------------------------------------------------------------------
# train, predict, and benchmark
# ---------------------------------------------------------------------------


def test_train_scarcity_model_returns_fitted_object():
    panel = _make_declining_panel()
    X = build_scarcity_features(panel)
    y = build_scarcity_target(panel, horizon_years=5)
    valid = y.notna()
    model = train_scarcity_model(X[valid], y[valid])
    assert model is not None


def test_predict_scarcity_returns_numeric_array():
    """Predictions are continuous log-scale values, not probabilities."""
    panel = _make_declining_panel()
    X = build_scarcity_features(panel)
    y = build_scarcity_target(panel, horizon_years=5)
    valid = y.notna()
    model = train_scarcity_model(X[valid], y[valid])
    preds = predict_scarcity(model, X[valid])
    assert isinstance(preds, np.ndarray)
    assert len(preds) == valid.sum()
    # Log freshwater values should be in a reasonable range (0 to 15)
    assert preds.min() >= 0
    assert preds.max() <= 20


def test_scarcity_model_beats_persistence_baseline():
    """Model RMSE must be lower than the naive persistence baseline.

    The persistence baseline predicts: next year = this year's value.
    If our model cannot beat this simple rule, it has learned nothing.
    This is the minimum competence bar for a forecasting model.
    """
    panel = _make_declining_panel(n_countries=10, n_years=20)
    X = build_scarcity_features(panel)
    y = build_scarcity_target(panel, horizon_years=3)
    from src.models.features import temporal_train_test_split

    train_panel, test_panel = temporal_train_test_split(panel, test_from_year=2015)
    X_train = X.loc[train_panel.index]
    y_train = y.loc[train_panel.index]
    X_test = X.loc[test_panel.index]
    y_test = y.loc[test_panel.index]

    train_valid = y_train.notna()
    test_valid = y_test.notna()

    model = train_scarcity_model(X_train[train_valid], y_train[train_valid])
    result = evaluate_vs_baseline(
        model,
        X_test[test_valid],
        y_test[test_valid],
        panel.loc[test_panel.index[test_valid], "log_freshwater_percap"].values,
    )
    assert result["model_rmse"] < result["baseline_rmse"], (
        f"Model RMSE {result['model_rmse']:.4f} did not beat "
        f"persistence baseline {result['baseline_rmse']:.4f}"
    )

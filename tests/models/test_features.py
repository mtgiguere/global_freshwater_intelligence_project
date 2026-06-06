"""Feature engineering for Phase 4 ML models — strict TDD.

Feature engineering is the part of ML that IS fully unit-testable.
A wrong lag, a data leakage bug, or a normalisation error will silently
corrupt every model trained downstream. These tests catch that.
"""

import numpy as np
import pandas as pd

from src.models.features import (
    add_lag_features,
    add_log_transforms,
    add_rolling_features,
    temporal_train_test_split,
)


def _make_panel(n_countries: int = 3, n_years: int = 10) -> pd.DataFrame:
    """Synthetic panel with known values for deterministic assertions."""
    countries = [f"C{i:02d}" for i in range(n_countries)]
    years = list(range(2000, 2000 + n_years))
    rows = [
        {"iso3": c, "year": y, "freshwater": float(i * 10 + j)}
        for i, c in enumerate(countries)
        for j, y in enumerate(years)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# add_lag_features
# ---------------------------------------------------------------------------


def test_add_lag_features_returns_dataframe():
    panel = _make_panel()
    result = add_lag_features(panel, cols=["freshwater"], lags=[1, 2])
    assert isinstance(result, pd.DataFrame)


def test_add_lag_features_creates_correctly_named_columns():
    panel = _make_panel()
    result = add_lag_features(panel, cols=["freshwater"], lags=[1, 3])
    assert "freshwater_lag1" in result.columns
    assert "freshwater_lag3" in result.columns


def test_add_lag_features_lag_is_within_country_not_across():
    """Lag must be computed within each country — never spilling across borders."""
    panel = _make_panel(n_countries=2, n_years=5)
    result = add_lag_features(panel, cols=["freshwater"], lags=[1])
    # The earliest year (min year) for each country must have NaN lag
    min_year = result["year"].min()
    first_rows = result[result["year"] == min_year]
    assert first_rows["freshwater_lag1"].isna().all()


def test_add_lag_features_correct_value():
    """Lag-1 of year 2001 must equal the value from year 2000 for the same country."""
    panel = _make_panel(n_countries=1, n_years=5)
    result = add_lag_features(panel, cols=["freshwater"], lags=[1])
    val_2000 = result.loc[result["year"] == 2000, "freshwater"].iloc[0]
    lag_2001 = result.loc[result["year"] == 2001, "freshwater_lag1"].iloc[0]
    assert val_2000 == lag_2001


# ---------------------------------------------------------------------------
# add_rolling_features
# ---------------------------------------------------------------------------


def test_add_rolling_features_creates_mean_columns():
    panel = _make_panel()
    result = add_rolling_features(panel, cols=["freshwater"], windows=[3])
    assert "freshwater_roll3_mean" in result.columns


def test_add_rolling_features_is_within_country():
    """Rolling mean must be computed within each country."""
    panel = _make_panel(n_countries=2, n_years=5)
    result = add_rolling_features(panel, cols=["freshwater"], windows=[3])
    # First two rows per country must be NaN (window of 3 needs 3 observations)
    for iso3 in panel["iso3"].unique():
        subset = result[result["iso3"] == iso3].sort_values("year")
        assert subset["freshwater_roll3_mean"].iloc[:2].isna().all()


# ---------------------------------------------------------------------------
# temporal_train_test_split
# ---------------------------------------------------------------------------


def test_temporal_train_test_split_returns_two_dataframes():
    panel = _make_panel(n_years=20)
    train, test = temporal_train_test_split(panel, test_from_year=2015)
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)


def test_temporal_train_test_split_no_leakage():
    """Test set must contain no years from the training period."""
    panel = _make_panel(n_years=20)
    train, test = temporal_train_test_split(panel, test_from_year=2015)
    assert train["year"].max() < 2015
    assert test["year"].min() >= 2015


# ---------------------------------------------------------------------------
# add_log_transforms
# ---------------------------------------------------------------------------


def test_add_log_transforms_adds_log_freshwater_percap():
    """Must add log_freshwater_percap derived from renewable_freshwater_percap."""
    df = pd.DataFrame(
        {
            "iso3": ["AFG"],
            "year": [2020],
            "renewable_freshwater_percap": [1000.0],
        }
    )
    result = add_log_transforms(df)
    assert "log_freshwater_percap" in result.columns


def test_add_log_transforms_handles_zero_without_nan():
    """log1p(0) = 0; values of zero must not produce NaN or -inf.

    Arid countries (e.g. Qatar, Kuwait) can have near-zero freshwater per
    capita. Refugee outflows are zero for most countries in most years.
    A NaN here propagates silently through all downstream feature engineering.
    """
    df = pd.DataFrame(
        {
            "iso3": ["TST"],
            "year": [2020],
            "renewable_freshwater_percap": [0.0],
            "gdp_pc_ppp": [0.0],
            "population": [0.0],
        }
    )
    result = add_log_transforms(df)
    assert not result["log_freshwater_percap"].isna().any()
    assert result["log_freshwater_percap"].iloc[0] == 0.0


def test_add_log_transforms_adds_all_three_columns():
    """Must add log_freshwater_percap, log_gdp_pc_ppp, and log_population."""
    df = pd.DataFrame(
        {
            "iso3": ["AFG"],
            "year": [2020],
            "renewable_freshwater_percap": [1000.0],
            "gdp_pc_ppp": [500.0],
            "population": [38_000_000.0],
        }
    )
    result = add_log_transforms(df)
    assert "log_gdp_pc_ppp" in result.columns
    assert "log_population" in result.columns


def test_add_log_transforms_preserves_original_columns():
    """Raw source columns must remain in the output — they are used elsewhere."""
    df = pd.DataFrame(
        {
            "iso3": ["AFG"],
            "year": [2020],
            "renewable_freshwater_percap": [1000.0],
            "gdp_pc_ppp": [500.0],
            "population": [38_000_000.0],
        }
    )
    result = add_log_transforms(df)
    assert "renewable_freshwater_percap" in result.columns
    assert "gdp_pc_ppp" in result.columns
    assert "population" in result.columns


def test_add_log_transforms_values_are_log1p():
    """Log values must equal numpy log1p of the clipped source value."""
    df = pd.DataFrame(
        {
            "iso3": ["AFG"],
            "year": [2020],
            "renewable_freshwater_percap": [1000.0],
            "gdp_pc_ppp": [500.0],
            "population": [38_000_000.0],
        }
    )
    result = add_log_transforms(df)
    assert abs(result["log_freshwater_percap"].iloc[0] - np.log1p(1000.0)) < 1e-9
    assert abs(result["log_gdp_pc_ppp"].iloc[0] - np.log1p(500.0)) < 1e-9


def test_temporal_train_test_split_covers_all_rows():
    """Every row must end up in exactly one split — no rows lost or duplicated."""
    panel = _make_panel(n_years=20)
    train, test = temporal_train_test_split(panel, test_from_year=2015)
    assert len(train) + len(test) == len(panel)

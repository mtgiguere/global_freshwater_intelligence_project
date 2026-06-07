"""Feature engineering for GFIP Phase 4 ML models.

All functions operate on the Master Panel DataFrame (iso3, year, ...variables).
The critical invariant: every transformation is computed WITHIN each country.
Cross-country leakage (e.g. taking a lag across country boundaries) produces
silently wrong features that corrupt every model trained downstream.

This module provides four building-block functions used by all three Phase 4 models:

  add_log_transforms       — compress wide-range variables (GDP, freshwater) onto
                             a log scale so that poorer/drier countries are treated
                             fairly by the gradient-descent optimisers.

  add_lag_features         — give models access to a country's own past values,
                             e.g. "what was this country's freshwater per capita
                             one year ago?" without leaking future data.

  add_rolling_features     — smooth short-term noise to expose multi-year trends,
                             e.g. a 5-year rolling mean of freshwater per capita
                             reveals whether a country is on a declining trajectory.

  temporal_train_test_split — divide the panel at a calendar year so that the
                              model is always trained on the past and evaluated
                              on the future (never the reverse).

Scientific note on temporal splits:
  For this project (data roughly 1960-2020), a test_from_year of 2015 gives five
  years of held-out test data while preserving ~55 years of training history.
  A value of 2018 would give ~3 years of test data (2018-2020), which is also
  reasonable for evaluating short-horizon forecasts. Earlier split years mean
  more test data but less training data — the trade-off should be guided by the
  minimum sample size the model needs rather than maximising test-set size.
"""

import numpy as np
import pandas as pd


def add_log_transforms(panel: pd.DataFrame) -> pd.DataFrame:
    """Add log-transformed versions of the key continuous variables.

    Uses log1p (= log(1 + x)) for numerical stability: handles values near zero
    or exactly zero without producing NaN or -inf. This is safe for freshwater
    per capita (arid countries can be near zero) and refugee outflows (most
    country-years are zero).

    The log transform matters because GDP and freshwater per capita span multiple
    orders of magnitude globally. Without it, model gradients are dominated by
    high-income / high-water countries and the models perform poorly on poorer
    countries — exactly the ones most relevant to the project's mission.

    The `.clip(lower=0)` before each log guards against rare negative values that
    can appear in the source data due to rounding errors or interpolation artefacts
    in the upstream AQUASTAT / World Bank series. A negative freshwater-per-capita
    value is physically impossible, so clipping to zero is the correct fix rather
    than dropping the row. Without this guard, np.log1p would return NaN for any
    negative input, silently propagating missing values through the entire feature
    pipeline and corrupting model training.

    Args:
        panel: Master Panel DataFrame containing at least one of:
            ``renewable_freshwater_percap``, ``gdp_pc_ppp``, ``population``.
            Columns that are absent are silently skipped — the function is safe
            to call on partial panels.

    Returns:
        A copy of ``panel`` with new columns appended:
            ``log_freshwater_percap``  = log1p(renewable_freshwater_percap)
            ``log_gdp_pc_ppp``         = log1p(gdp_pc_ppp)
            ``log_population``         = log1p(population)
        The original columns are preserved unchanged.
    """
    result = panel.copy()
    if "renewable_freshwater_percap" in result.columns:
        result["log_freshwater_percap"] = np.log1p(
            result["renewable_freshwater_percap"].clip(lower=0)
        )
    if "gdp_pc_ppp" in result.columns:
        result["log_gdp_pc_ppp"] = np.log1p(result["gdp_pc_ppp"].clip(lower=0))
    if "population" in result.columns:
        result["log_population"] = np.log1p(result["population"].clip(lower=0))
    return result


def add_lag_features(
    panel: pd.DataFrame,
    cols: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """Add lagged versions of columns, computed within each country.

    A lag-k feature for year Y contains the value from year Y-k for the
    same country. The first k years for each country will be NaN.

    This is the primary way we give models access to temporal history
    without creating data leakage across countries.

    Why lags? Because the current-year value of the outcome variable (conflict,
    refugee outflow, freshwater level) is what we want to predict — we must
    not use it as a feature. Using Y-1, Y-2 gives the model a picture of recent
    history without revealing the answer to the question being asked.

    Args:
        panel: Master Panel DataFrame. Must contain columns named ``iso3``
            (ISO 3166-1 alpha-3 country code) and ``year``, plus every column
            listed in ``cols``.
        cols: Names of the columns to lag. Columns not present in ``panel``
            will raise a KeyError from pandas.
        lags: List of integer lag offsets to create. For example, ``[1, 2]``
            creates both a one-year and a two-year lag for each column.

    Returns:
        A copy of ``panel`` with new columns appended for each (col, lag)
        combination, named ``{col}_lag{lag}``. The first ``max(lags)`` rows
        per country will contain NaN because there is no prior data to shift
        into those positions.
    """
    result = panel.copy()
    for col in cols:
        for lag in lags:
            result[f"{col}_lag{lag}"] = result.groupby("iso3")[col].shift(lag)
    return result


def add_rolling_features(
    panel: pd.DataFrame,
    cols: list[str],
    windows: list[int],
) -> pd.DataFrame:
    """Add rolling mean features, computed within each country.

    A rolling-k mean for year Y averages the k years up to and including Y
    for the same country. The first k-1 years for each country will be NaN.

    Rolling means smooth out year-to-year noise and capture medium-term trends —
    important for slow-moving variables like groundwater depletion, where a
    single anomalous year is less informative than the 5-year trajectory.

    Args:
        panel: Master Panel DataFrame. Must contain columns named ``iso3`` and
            ``year``, plus every column listed in ``cols``.
        cols: Names of the columns to compute rolling means for.
        windows: List of window sizes (in years) to compute. For example,
            ``[3, 5]`` creates a 3-year and a 5-year rolling mean for each column.

    Returns:
        A copy of ``panel`` with new columns appended for each (col, window)
        combination, named ``{col}_roll{window}_mean``. Rows that do not yet
        have a full window of prior data are NaN (``min_periods=k`` enforces this —
        we never return a mean based on fewer years than the window size, which
        would be misleading for trend features).
    """
    result = panel.copy()
    for col in cols:
        for window in windows:
            # Python closures capture variables by reference, not by value.
            # If we wrote `lambda x: x.rolling(window, ...).mean()` directly,
            # all lambdas created in this loop would share the SAME `window`
            # variable, and by the time any of them executes, `window` would
            # hold only the final value from the loop iteration — all rolling
            # features would use the same (wrong) window size. Binding to a
            # local `w` here, or using a default argument `k=w` in the lambda,
            # captures the current value of `window` at the moment the lambda
            # is created, which is the correct behaviour.
            w = window  # bind loop variable to freeze its current value
            result[f"{col}_roll{w}_mean"] = result.groupby("iso3")[col].transform(
                lambda x, k=w: x.rolling(k, min_periods=k).mean()
            )
    return result


def temporal_train_test_split(
    panel: pd.DataFrame,
    test_from_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the panel into train and test sets by year — no data leakage.

    The test set contains all rows from test_from_year onwards.
    The training set contains all rows strictly before test_from_year.

    This enforces the golden rule of time-series ML: the model never sees
    the future during training. A random split would leak future information
    into the training set and produce falsely optimistic evaluation metrics —
    the model would appear to work well but would fail in practice because
    it had already "seen" the test years during training.

    Choosing a good test_from_year for this project:
        The Master Panel runs from 1946 to approximately 2020. A split at 2015
        gives ~55 years of training history and 5 years of test data — enough to
        evaluate medium-term forecasting skill. A split at 2018 gives ~3 years of
        test data (2018-2020), which is useful for evaluating short-horizon
        predictions while preserving the full 1946-2017 training window. Both are
        reasonable choices; earlier splits sacrifice test coverage and later splits
        reduce training data. The current codebase uses 2015 as the default.

    Args:
        panel: Master Panel DataFrame. Must contain a column named ``year``
            (integer or comparable type).
        test_from_year: The first year to include in the test set. All rows
            with ``year >= test_from_year`` go to the test split; all rows
            with ``year < test_from_year`` go to the training split.

    Returns:
        A two-tuple ``(train, test)`` where both are copies of the relevant
        slice of ``panel``. Neither split is modified in place.
    """
    train = panel[panel["year"] < test_from_year].copy()
    test = panel[panel["year"] >= test_from_year].copy()
    return train, test

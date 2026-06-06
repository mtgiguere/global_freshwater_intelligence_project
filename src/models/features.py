"""Feature engineering for GFIP Phase 4 ML models.

All functions operate on the Master Panel DataFrame (iso3, year, ...variables).
The critical invariant: every transformation is computed WITHIN each country.
Cross-country leakage (e.g. taking a lag across country boundaries) produces
silently wrong features that corrupt every model trained downstream.
"""

import pandas as pd


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
    important for slow-moving variables like groundwater depletion.
    """
    result = panel.copy()
    for col in cols:
        for window in windows:
            w = window  # bind loop variable before lambda captures it
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
    into the training set and produce falsely optimistic evaluation metrics.
    """
    train = panel[panel["year"] < test_from_year].copy()
    test = panel[panel["year"] >= test_from_year].copy()
    return train, test

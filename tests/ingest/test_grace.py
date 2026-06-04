"""GRACE groundwater ingest — strict TDD, one test at a time.

GRACE is a 3D raster (lat x lon x time). The pipeline:
  1. Load netCDF4 → xarray Dataset
  2. Resample monthly → annual mean
  3. Area-weighted spatial aggregation per country polygon
  4. Produce tidy country-year panel

Synthetic fixtures are used throughout — no real NASA files required.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from geopandas import GeoDataFrame
from shapely.geometry import box

from src.ingest.grace import load_grace

# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------


def _make_dataset(value: float = 1.0, years: list[int] | None = None) -> xr.Dataset:
    """Synthetic GRACE-like monthly dataset on a 1° global grid."""
    if years is None:
        years = [2010]
    lats = np.arange(-89.5, 90.5, 1.0)
    lons = np.arange(-179.5, 180.5, 1.0)
    times = pd.date_range(f"{years[0]}-01", periods=12 * len(years), freq="MS")
    data = np.full((len(times), len(lats), len(lons)), value)
    return xr.Dataset(
        {"lwe_thickness": (["time", "lat", "lon"], data)},
        coords={"lat": lats, "lon": lons, "time": times},
    )


def _make_shapes(*iso3_boxes: tuple[str, float, float, float, float]) -> GeoDataFrame:
    """Synthetic country boundaries as simple rectangles.

    Each entry: (iso3, min_lon, min_lat, max_lon, max_lat)
    """
    return GeoDataFrame(
        {
            "iso3": [r[0] for r in iso3_boxes],
            "geometry": [box(r[1], r[2], r[3], r[4]) for r in iso3_boxes],
        },
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_grace_returns_dataframe():
    """The consumer gets a DataFrame back."""
    ds = _make_dataset()
    shapes = _make_shapes(("AFG", 60, 30, 75, 38))
    result = load_grace(ds, shapes)
    assert isinstance(result, pd.DataFrame)


def test_load_grace_has_iso3_and_year_columns():
    """Output must have iso3 and year as panel identifiers."""
    ds = _make_dataset()
    shapes = _make_shapes(("AFG", 60, 30, 75, 38))
    result = load_grace(ds, shapes)
    assert "iso3" in result.columns
    assert "year" in result.columns


def test_load_grace_is_one_row_per_country_year():
    """Output must have exactly one row per (iso3, year) — no duplicates."""
    ds = _make_dataset(years=[2010, 2011])
    shapes = _make_shapes(("AFG", 60, 30, 75, 38), ("FRA", -5, 42, 8, 51))
    result = load_grace(ds, shapes)
    assert result.duplicated(subset=["iso3", "year"]).sum() == 0
    assert len(result) == 4  # 2 countries x 2 years


def test_load_grace_produces_lwe_anomaly_column():
    """Output must contain grace_lwe_anomaly_cm column."""
    ds = _make_dataset(value=1.0)
    shapes = _make_shapes(("AFG", 60, 30, 75, 38))
    result = load_grace(ds, shapes)
    assert "grace_lwe_anomaly_cm" in result.columns


def test_load_grace_area_weighted_mean_of_constant_equals_that_constant():
    """Property: area-weighted mean of a spatially uniform field must equal the field value.

    This holds regardless of the country shape or latitude — it is the fundamental
    correctness test for the spatial aggregation. If this fails, the weights are wrong.
    """
    for value in [-3.5, 0.0, 2.8]:
        ds = _make_dataset(value=value)
        shapes = _make_shapes(("TST", -10, -10, 10, 10))  # equatorial box
        result = load_grace(ds, shapes)
        actual = result.iloc[0]["grace_lwe_anomaly_cm"]
        assert abs(actual - value) < 1e-6, f"Expected {value}, got {actual}"


def test_load_grace_area_weighted_mean_of_constant_holds_at_high_latitude():
    """Property must hold at high latitude where cos(lat) weighting matters most."""
    for value in [-1.0, 5.0]:
        ds = _make_dataset(value=value)
        shapes = _make_shapes(("TST", 10, 60, 30, 70))  # 60-70 degrees N box
        result = load_grace(ds, shapes)
        actual = result.iloc[0]["grace_lwe_anomaly_cm"]
        assert abs(actual - value) < 1e-6, f"Expected {value}, got {actual}"


def test_load_grace_tiny_country_with_no_grid_cells_gets_nan():
    """A country smaller than one grid cell must get NaN, not crash."""
    ds = _make_dataset(value=1.0)
    # 0.1 x 0.1 degree box -- smaller than the 1 degree grid cells in the synthetic dataset
    shapes = _make_shapes(("TST", 10.0, 10.0, 10.1, 10.1))
    result = load_grace(ds, shapes)
    assert len(result) == 1
    assert pd.isna(result.iloc[0]["grace_lwe_anomaly_cm"])


def test_load_grace_raises_if_lwe_thickness_variable_missing():
    """Dataset without lwe_thickness variable must raise ValueError."""
    import xarray as xr

    bad_ds = xr.Dataset({"wrong_var": (["time", "lat", "lon"], [[[1.0]]])})
    shapes = _make_shapes(("AFG", 60, 30, 75, 38))
    with pytest.raises(ValueError, match="lwe_thickness"):
        load_grace(bad_ds, shapes)

"""NASA GRACE / GRACE-FO satellite groundwater anomaly ingest.

Background: How GRACE works
-----------------------------
GRACE (Gravity Recovery and Climate Experiment, 2002-2017) and its successor
GRACE-FO (GRACE Follow-On, 2018-present) are twin satellites that fly in formation
roughly 220 km apart. By measuring tiny changes in the distance between the two
satellites — caused by variations in Earth's gravitational field below them — the
mission detects changes in the distribution of mass on and beneath the surface.

Because water is heavy, large-scale shifts in groundwater, soil moisture, ice sheets,
and lake levels all appear as gravitational anomalies. GRACE is the only system
that can measure groundwater changes globally from space — there is no network of
wells or sensors that could match its coverage.

The key measurement: Liquid Water Equivalent (LWE) anomaly
------------------------------------------------------------
The primary variable is `lwe_thickness` (liquid water equivalent thickness, in
centimetres). It represents the deviation from the 2004-2009 baseline mean at each
grid point. A value of -5 cm means there is 5 cm less water (integrated from the
surface down to the deepest aquifer) than the long-term average at that location.
A value of +3 cm means 3 cm more than average — this could be a wet year, a
recovering aquifer, or increased soil moisture after heavy rainfall.

What GRACE cannot distinguish
-------------------------------
GRACE measures total terrestrial water storage change — it cannot separate
groundwater from soil moisture or surface water. Researchers use land surface
models (e.g. GLDAS) to subtract estimated soil moisture and surface water, leaving
an estimated groundwater anomaly. The processed mascon products we use have already
applied this separation.

Why GRACE matters for GFIP
----------------------------
GRACE is the primary data source for H7 (groundwater depletion → GDP trajectory).
It provides the only systematic, globally consistent measure of aquifer depletion —
critical because many major aquifers (the High Plains/Ogallala in the US, the
North China Plain, the Arabian Aquifer) are being depleted at rates that threaten
food security and economic stability for hundreds of millions of people.

Data format
-----------
GRACE data is distributed as a global gridded dataset in netCDF4 format. Each file
contains a time series of 2D rasters (latitude x longitude grid) at roughly 0.5 deg
or 1 deg spatial resolution (~55 km at the equator). This module processes that spatial
data into per-country annual means using area-weighted spatial averaging.

Coverage: 2002-present with a gap between GRACE (2002-2017) and GRACE-FO (2018-).
The gap produces NaN values in the annual panel for 2017-2018 for many countries.

Source: https://grace.jpl.nasa.gov/
Download: src/ingest/download/grace_process.py
"""

import numpy as np
import pandas as pd
from regionmask import Regions


def _area_weighted_mean(da, mask):
    """Compute the area-weighted mean of a 2D (lat, lon) DataArray within a boolean mask.

    Why area weighting is necessary
    ---------------------------------
    A regular latitude/longitude grid (equal degrees spacing) does NOT have equal-area
    grid cells. Near the equator a 1 deg x 1 deg cell is roughly 111 km x 111 km (~12,000 km2).
    Near the poles the same 1 deg x 1 deg cell is much narrower east-to-west: at 70 deg latitude,
    a 1 deg longitude step is only about 38 km wide, giving a cell of ~4,200 km2.

    If we simply averaged all grid cells within a country's boundary without weighting,
    a country that spans high latitudes (e.g. Canada, Russia) would have its groundwater
    signal distorted — polar grid cells that represent very little actual land area would
    count the same as large equatorial cells.

    The cosine(latitude) correction accounts for this: cos(0°) = 1.0 at the equator
    (full weight), cos(70°) ≈ 0.34 at high latitudes (one-third the weight), and
    cos(90°) = 0 at the poles (zero weight). This is the standard correction used in
    climate science for area-weighted global means.

    Args:
        da: An xarray DataArray with dimensions "lat" and "lon" representing a single
            time-slice of the GRACE LWE anomaly field (units: cm). Cells outside
            the country boundary should be left as NaN or zeroed by the mask.
        mask: A boolean array of the same shape as da (lat x lon). True where the
            grid cell falls within the country's boundary polygon, False outside.
            Produced by regionmask using the country's GeoJSON geometry.

    Returns:
        A single float: the area-weighted mean LWE anomaly in centimetres for all
        grid cells within the mask. Returns NaN if no valid (unmasked) cells exist
        for this country in this time-slice, which can happen for very small countries
        that fall between grid points at 1° resolution.
    """
    # Compute cosine-of-latitude weights — one weight per latitude band, broadcast
    # across all longitudes automatically by xarray's weighted averaging engine.
    weights = np.cos(np.deg2rad(da.lat))
    weighted = da.where(mask).weighted(weights)
    return float(weighted.mean(("lat", "lon")).values)


def load_grace(ds, shapes) -> pd.DataFrame:
    """Aggregate GRACE gridded LWE anomalies to a country-year panel.

    Takes a global gridded GRACE/GRACE-FO dataset (an xarray Dataset covering the
    full satellite period) and a GeoDataFrame of country boundaries, and returns a
    tidy panel with one row per (country, year).

    The aggregation process for each country-year:
        1. Resample the monthly GRACE time series to annual means using calendar-year
           end ("YE") resampling — this reduces noise and matches the annual resolution
           of the other GFIP data sources.
        2. For each country, use regionmask to create a boolean mask identifying which
           grid cells fall within that country's boundary.
        3. Compute the area-weighted mean LWE anomaly across all cells in the mask
           (see _area_weighted_mean for why weighting matters).
        4. Store the result as one row in the output panel.

    Args:
        ds: An xarray Dataset containing at minimum the variable "lwe_thickness"
            (liquid water equivalent anomaly, in cm), with dimensions "lat", "lon",
            and "time". Expected to be the NASA GRACE Mascon product (JPL RL06 or
            equivalent), already in cm units. The "time" dimension should be a
            cftime or datetime64 array at monthly or sub-monthly frequency.
        shapes: A geopandas GeoDataFrame where each row represents one country.
            Must have columns:
            - "iso3": ISO 3166-1 alpha-3 country code (str)
            - "geometry": Polygon or MultiPolygon of the country boundary (EPSG:4326)

    Returns:
        A tidy pandas DataFrame with one row per (iso3, year) combination, and columns:
        - iso3 (str): ISO 3166-1 alpha-3 country code
        - year (int): Calendar year
        - grace_lwe_anomaly_cm (float): Area-weighted mean LWE anomaly in centimetres
          relative to the 2004-2009 baseline. Negative values indicate water depletion
          (less total water storage than average); positive values indicate accumulation.
          Will be NaN for country-years where no valid GRACE data exists (e.g. the
          2017-2018 gap between GRACE and GRACE-FO, or very small island nations).

    Raises:
        ValueError: If "lwe_thickness" is not present in the dataset.
            Message: "Dataset missing required variable 'lwe_thickness'"
    """
    # Validate the dataset has the variable we need before doing any expensive computation.
    if "lwe_thickness" not in ds:
        raise ValueError("Dataset missing required variable 'lwe_thickness'")

    # Resample from monthly to annual means. "YE" (year-end) groups all months in a
    # calendar year and averages them. This reduces short-term noise (e.g. seasonal
    # snowmelt affecting the gravity signal) and gives us one value per year —
    # consistent with the annual resolution of AQUASTAT, World Bank, FSI, etc.
    annual = ds["lwe_thickness"].resample(time="YE").mean()

    rows = []
    for year_idx, year in enumerate(annual.time.dt.year.values):
        # Extract the single time-slice (annual mean) for this year.
        da = annual.isel(time=year_idx)

        for _, row in shapes.iterrows():
            # Build a regionmask Regions object from this country's boundary polygon.
            # regionmask.mask() returns an array where cells inside the region are
            # labeled 0 (the region index) and cells outside are NaN. We convert
            # this to a boolean mask (True = inside country).
            region = Regions([row["geometry"]])
            mask = region.mask(da.lon, da.lat) == 0

            # Compute the area-weighted mean for this country in this year.
            value = _area_weighted_mean(da, mask)

            rows.append(
                {
                    "iso3": row["iso3"],
                    "year": int(year),
                    "grace_lwe_anomaly_cm": value,
                }
            )

    return pd.DataFrame(rows)

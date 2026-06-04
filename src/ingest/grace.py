import numpy as np
import pandas as pd
from regionmask import Regions


def _area_weighted_mean(da, mask):
    """Area-weighted mean of a 2D (lat, lon) DataArray within a boolean mask.

    Grid cells shrink toward the poles, so weights = cos(lat).
    """
    weights = np.cos(np.deg2rad(da.lat))
    weighted = da.where(mask).weighted(weights)
    return float(weighted.mean(("lat", "lon")).values)


def load_grace(ds, shapes):
    if "lwe_thickness" not in ds:
        raise ValueError("Dataset missing required variable 'lwe_thickness'")
    annual = ds["lwe_thickness"].resample(time="YE").mean()
    rows = []
    for year_idx, year in enumerate(annual.time.dt.year.values):
        da = annual.isel(time=year_idx)
        for _, row in shapes.iterrows():
            region = Regions([row["geometry"]])
            mask = region.mask(da.lon, da.lat) == 0
            value = _area_weighted_mean(da, mask)
            rows.append(
                {
                    "iso3": row["iso3"],
                    "year": int(year),
                    "grace_lwe_anomaly_cm": value,
                }
            )
    return pd.DataFrame(rows)

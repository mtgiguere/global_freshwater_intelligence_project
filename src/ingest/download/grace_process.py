"""Process the GRACE mascon netCDF4 into a country-year panel.

Uses regionmask's built-in Natural Earth country boundaries -- no shapefile
download required. Aggregates monthly LWE anomaly to annual mean per country.
"""

from pathlib import Path

import geopandas as gpd
import regionmask
import xarray as xr


def process_grace(grace_path: Path, dest_dir: Path) -> Path:
    """Aggregate GRACE LWE anomaly to country-year panel."""
    dest = dest_dir / "grace_country_year.csv"
    if dest.exists():
        print("  GRACE country panel — already exists, skipping")
        return dest

    dest_dir.mkdir(parents=True, exist_ok=True)

    print("  Loading GRACE netCDF4...")
    ds = xr.open_dataset(grace_path)

    # Resample monthly -> annual mean
    annual = ds["lwe_thickness"].resample(time="YE").mean()
    yr_min = int(annual.time.dt.year.min())
    yr_max = int(annual.time.dt.year.max())
    print(f"  Annual means: {len(annual.time)} years ({yr_min}-{yr_max})")

    # Build country GeoDataFrame from regionmask Natural Earth (no download needed).
    # regionmask abbrevs are not reliable ISO3 -- map country names via pycountry.
    print("  Building country masks from Natural Earth...")
    import pycountry

    ne = regionmask.defined_regions.natural_earth_v5_0_0.countries_50

    def _to_iso3(name: str) -> str | None:
        try:
            return pycountry.countries.lookup(name).alpha_3
        except LookupError:
            return None

    iso3_list = [_to_iso3(n) for n in ne.names]
    shapes = gpd.GeoDataFrame(
        {"iso3": iso3_list, "geometry": list(ne.polygons)},
        crs="EPSG:4326",
    )
    # Drop regions that didn't map to a valid ISO3 (territories, disputed areas)
    shapes = shapes[shapes["iso3"].notna()].reset_index(drop=True)

    # Load our ingest module
    from src.ingest.grace import load_grace

    print("  Running spatial aggregation (this takes a few minutes)...")
    result = load_grace(ds, shapes)

    result.to_csv(dest, index=False)
    print(f"  GRACE country panel saved: {len(result):,} rows -> {dest}")
    return dest


if __name__ == "__main__":
    raw_dir = Path(__file__).parents[3] / "data" / "raw"
    grace_files = list((raw_dir / "grace").glob("*.nc"))
    if not grace_files:
        raise FileNotFoundError("No GRACE .nc file found in data/raw/grace/")

    process_grace(
        grace_path=grace_files[0],
        dest_dir=raw_dir / "grace",
    )

"""Download UNHCR refugee statistics via the public API.

No API key required. Fetches per-country aggregate (all destinations combined)
for each country of origin to produce country-year displacement totals.
API docs: https://api.unhcr.org/docs/refugee-statistics.html
"""

from pathlib import Path

import pandas as pd
import pycountry
import requests

_VALID_ISO3 = {c.alpha_3 for c in pycountry.countries}
_API_BASE = "https://api.unhcr.org/population/v1/population/"
_YEAR_START = 2000
_YEAR_END = 2023


def _fetch_country(iso3: str) -> list[dict]:
    """Fetch all years for one country of origin — returns one row per year."""
    r = requests.get(
        _API_BASE,
        params={
            "coo": iso3,
            "yearFrom": _YEAR_START,
            "yearTo": _YEAR_END,
            "limit": 100,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["items"]


def download_unhcr(dest_dir: Path) -> Path:
    """Download UNHCR population stats and save as load_unhcr()-compatible CSV."""
    dest = dest_dir / "unhcr_displacement.csv"
    if dest.exists():
        print("  UNHCR — already exists, skipping")
        return dest

    dest_dir.mkdir(parents=True, exist_ok=True)
    countries = sorted(_VALID_ISO3)
    all_records: list[dict] = []

    for i, iso3 in enumerate(countries):
        if i % 50 == 0:
            print(f"  {i}/{len(countries)} countries...")
        records = _fetch_country(iso3)
        for row in records:
            all_records.append(
                {
                    "Country": iso3,
                    "Year": row["year"],
                    "Refugees": row.get("refugees") or 0,
                    "IDPs": row.get("idps") or 0,
                    "AsylumSeekers": row.get("asylum_seekers") or 0,
                }
            )

    df = pd.DataFrame(all_records).sort_values(["Country", "Year"]).reset_index(drop=True)
    df.to_csv(dest, index=False)
    print(f"  UNHCR — saved ({dest.stat().st_size:,} bytes, {len(df):,} rows)")
    return dest


if __name__ == "__main__":
    raw_dir = Path(__file__).parents[3] / "data" / "raw" / "unhcr"
    print("Downloading UNHCR displacement data...")
    result = download_unhcr(raw_dir)
    print(f"\nDone. File at {result}")

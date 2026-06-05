"""Download UN DESA population data via the World Bank API.

UN DESA World Population Prospects data is co-published by the World Bank.
Fetches total, urban, and rural population and saves in the format that
load_undesa() expects: Country, Year, PopTotal, PopUrban, PopRural (in thousands).
"""

from pathlib import Path

import pandas as pd
import pycountry
import requests

_VALID_ISO3 = {c.alpha_3 for c in pycountry.countries}

_WB_API = "https://api.worldbank.org/v2/country/all/indicator/{indicator}"

_INDICATORS = {
    "SP.POP.TOTL": "PopTotal",
    "SP.URB.TOTL": "PopUrban",
    "SP.RUR.TOTL": "PopRural",
}


def _fetch_indicator(indicator: str) -> dict[tuple[str, int], float]:
    """Fetch all country-years for one WB indicator. Returns {(iso3, year): value}."""
    values: dict[tuple[str, int], float] = {}
    page = 1

    while True:
        r = requests.get(
            _WB_API.format(indicator=indicator),
            params={"format": "json", "per_page": 1000, "page": page},
            timeout=120,
        )
        r.raise_for_status()
        meta, data = r.json()

        for row in data:
            iso3 = row.get("countryiso3code", "")
            if iso3 not in _VALID_ISO3:
                continue
            if row["value"] is None:
                continue
            values[(iso3, int(row["date"]))] = row["value"]

        if page >= meta["pages"]:
            break
        page += 1

    return values


def download_undesa(dest_dir: Path) -> Path:
    """Download population data to dest_dir as load_undesa()-compatible CSV."""
    dest = dest_dir / "undesa_population.csv"
    if dest.exists():
        print("  UN DESA — already exists, skipping")
        return dest

    dest_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, dict] = {}

    for indicator, col_name in _INDICATORS.items():
        print(f"  {col_name}... downloading")
        values = _fetch_indicator(indicator)
        for (iso3, year), value in values.items():
            key = f"{iso3}_{year}"
            if key not in data:
                data[key] = {"Country": iso3, "Year": year}
            # WB publishes in absolute numbers; UN DESA format is thousands
            data[key][col_name] = round(value / 1000, 1)

    df = pd.DataFrame(data.values()).sort_values(["Country", "Year"]).reset_index(drop=True)
    df.to_csv(dest, index=False)
    print(f"  UN DESA — saved ({dest.stat().st_size:,} bytes, {len(df):,} rows)")
    return dest


if __name__ == "__main__":
    raw_dir = Path(__file__).parents[3] / "data" / "raw" / "undesa"
    print("Downloading UN DESA population data (via World Bank API)...")
    result = download_undesa(raw_dir)
    print(f"\nDone. File at {result}")

"""Download WHO health indicators via the World Bank API.

Life expectancy and U5MR are co-published by the World Bank from WHO data.
Diarrhoeal DALY burden (GBD metric) is not available via WB — DiarrhoeaDALY
column will be NaN until a direct GBD/GHDx download is implemented.

Saves in the format load_who() expects: Country, Year, LifeExpectancy, U5MR, DiarrhoeaDALY.
"""

from pathlib import Path

import pandas as pd
import pycountry
import requests

_VALID_ISO3 = {c.alpha_3 for c in pycountry.countries}

_WB_API = "https://api.worldbank.org/v2/country/all/indicator/{indicator}"

_INDICATORS = {
    "SP.DYN.LE00.IN": "LifeExpectancy",
    "SH.DYN.MORT": "U5MR",
}


def _fetch_indicator(indicator: str) -> dict[tuple[str, int], float]:
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
            if iso3 not in _VALID_ISO3 or row["value"] is None:
                continue
            values[(iso3, int(row["date"]))] = row["value"]

        if page >= meta["pages"]:
            break
        page += 1

    return values


def download_who(dest_dir: Path) -> Path:
    """Download WHO health indicators via WB API in load_who()-compatible CSV."""
    dest = dest_dir / "who_health.csv"
    if dest.exists():
        print("  WHO — already exists, skipping")
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
            data[key][col_name] = value

    df = pd.DataFrame(data.values()).sort_values(["Country", "Year"]).reset_index(drop=True)
    # DiarrhoeaDALY not available via WB — leave as NaN for future GBD integration
    df["DiarrhoeaDALY"] = float("nan")
    df.to_csv(dest, index=False)
    print(f"  WHO — saved ({dest.stat().st_size:,} bytes, {len(df):,} rows)")
    return dest


if __name__ == "__main__":
    raw_dir = Path(__file__).parents[3] / "data" / "raw" / "GHDx"
    print("Downloading WHO health indicators (via World Bank API)...")
    result = download_who(raw_dir)
    print(f"\nDone. File at {result}")

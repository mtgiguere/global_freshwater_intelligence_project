"""Download World Bank WDI indicators via the JSON API.

Fetches all country-year observations for each indicator, converts to the
wide CSV format that src/ingest/worldbank.py expects, and stores in data/raw/.
No API key required.
"""

from pathlib import Path

import pandas as pd
import requests

# True WB API indicator codes (HDI is UNDP — handled separately)
WB_INDICATORS = {
    "NY.GDP.PCAP.KD": "gdp_pc_ppp",
    "SI.POV.GINI": "gini",
    "NV.AGR.TOTL.ZS": "agri_value_added_pct_gdp",
    "SH.H2O.SMDW.ZS": "safe_water_access_pct",
}

_WB_API = "https://api.worldbank.org/v2/country/all/indicator/{indicator}"


def _fetch_indicator(indicator: str) -> pd.DataFrame:
    """Fetch all country-years for one indicator, return wide-format DataFrame."""
    records = []
    page = 1

    while True:
        r = requests.get(
            _WB_API.format(indicator=indicator),
            params={"format": "json", "per_page": 1000, "mrv": 65, "page": page},
            timeout=60,
        )
        r.raise_for_status()
        meta, data = r.json()

        for row in data:
            iso3 = row.get("countryiso3code", "")
            if len(iso3) != 3:  # skip regional aggregates
                continue
            records.append(
                {
                    "Country Name": row["country"]["value"],
                    "Country Code": iso3,
                    "Indicator Name": row["indicator"]["value"],
                    "Indicator Code": row["indicator"]["id"],
                    "Year": row["date"],
                    "Value": row["value"],
                }
            )

        if page >= meta["pages"]:
            break
        page += 1

    df = pd.DataFrame(records)
    if df.empty:
        return df

    wide = df.pivot_table(
        index=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
        columns="Year",
        values="Value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def download_worldbank(dest_dir: Path) -> dict[str, Path]:
    """Download all WB indicators to dest_dir. Returns {indicator_code: csv_path}."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for indicator in WB_INDICATORS:
        dest = dest_dir / f"{indicator}.csv"
        if dest.exists():
            print(f"  {indicator} — already exists, skipping")
            paths[indicator] = dest
            continue

        print(f"  {indicator} — downloading (may take a moment)...")
        df = _fetch_indicator(indicator)
        df.to_csv(dest, index=False)
        print(f"  {indicator} — saved ({dest.stat().st_size:,} bytes, {len(df)} rows)")
        paths[indicator] = dest

    return paths


if __name__ == "__main__":
    raw_dir = Path(__file__).parents[3] / "data" / "raw" / "worldbank"
    print("Downloading World Bank indicators...")
    result = download_worldbank(raw_dir)
    print(f"\nDone. {len(result)} file(s) in {raw_dir}")

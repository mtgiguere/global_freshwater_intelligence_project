"""Download AQUASTAT freshwater variables via the World Bank API.

AQUASTAT data is co-published by the World Bank (same values, different access).
This script fetches the three primary freshwater variables and converts them to
AQUASTAT's long format (Area, Variable Name, Year, Value) so load_aquastat()
works without modification.

When the AQUASTAT REST API endpoint is confirmed, swap this download script.
load_aquastat() will not need to change.
"""

from pathlib import Path

import pandas as pd
import pycountry
import requests

# Pre-built set of valid ISO 3166-1 alpha-3 codes — filters out WB regional aggregates
_VALID_ISO3 = {c.alpha_3 for c in pycountry.countries}

# World Bank indicator codes for the AQUASTAT freshwater variables
_WB_TO_AQUASTAT: dict[str, str] = {
    "ER.H2O.INTR.PC": "Renewable internal freshwater resources per capita",
    "ER.H2O.FWTL.K3": "Total freshwater withdrawal",
    "ER.H2O.FWAG.ZS": "Agricultural water withdrawal as % of total freshwater withdrawal",
}

_WB_API = "https://api.worldbank.org/v2/country/all/indicator/{indicator}"


def _fetch_wb_indicator(indicator: str) -> list[dict]:
    records, page = [], 1
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
            if iso3 not in _VALID_ISO3:  # exclude WB regional aggregates
                continue
            records.append(
                {
                    "Area": iso3,  # use ISO3 code directly — avoids WB name quirks
                    "Variable Name": _WB_TO_AQUASTAT[indicator],
                    "Year": int(row["date"]),
                    "Value": row["value"],
                    "Symbol": "",
                    "Md": "",
                }
            )
        if page >= meta["pages"]:
            break
        page += 1
    return records


def download_aquastat(dest_dir: Path) -> Path:
    """Download AQUASTAT freshwater variables via WB API in AQUASTAT long format."""
    dest = dest_dir / "aquastat_all.csv"
    if dest.exists():
        print("  AQUASTAT — already exists, skipping")
        return dest

    dest_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []

    for indicator, variable_name in _WB_TO_AQUASTAT.items():
        print(f"  {variable_name[:50]}... downloading")
        all_records.extend(_fetch_wb_indicator(indicator))

    df = pd.DataFrame(all_records)
    df.to_csv(dest, index=False)
    print(f"  AQUASTAT — saved ({dest.stat().st_size:,} bytes, {len(df):,} rows)")
    return dest


if __name__ == "__main__":
    raw_dir = Path(__file__).parents[3] / "data" / "raw" / "aquastat"
    print("Downloading AQUASTAT (via World Bank API)...")
    result = download_aquastat(raw_dir)
    print(f"\nDone. File at {result}")

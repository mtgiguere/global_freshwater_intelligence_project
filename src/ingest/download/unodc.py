"""Process UNODC homicide Excel file into a CSV for load_unodc().

Download from https://www.unodc.org/unodc/en/data-and-analysis/
Place data_cts_intentional_homicide.xlsx in data/raw/unodc/ then run this.
"""

from pathlib import Path

import pandas as pd
import pycountry

_VALID_ISO3 = {c.alpha_3 for c in pycountry.countries}


def process_unodc_homicide(source_dir: Path, dest_dir: Path) -> Path:
    """Extract homicide rate and count from the UNODC intentional homicide Excel."""
    xl_files = list(source_dir.glob("*intentional_homicide*.xlsx"))
    if not xl_files:
        raise FileNotFoundError(f"No UNODC homicide Excel file found in {source_dir}")

    xl_file = xl_files[0]
    print(f"  Reading {xl_file.name}...")

    df = pd.read_excel(xl_file, sheet_name="data_cts_intentional_homicide", header=2)

    # Filter to total homicide counts and rates (all ages, both sexes, total dimension)
    mask = (
        (df["Indicator"] == "Victims of intentional homicide")
        & (df["Sex"] == "Total")
        & (df["Age"] == "Total")
        & (df["Dimension"] == "Total")
        & df["Unit of measurement"].isin(["Rate per 100,000 population", "Counts"])
    )
    filtered = df[mask][["Iso3_code", "Year", "Unit of measurement", "VALUE"]].copy()

    # Drop sub-national and non-standard codes
    filtered = filtered[filtered["Iso3_code"].isin(_VALID_ISO3)]

    # Pivot to one row per country-year with Rate and Count as columns
    wide = filtered.pivot_table(
        index=["Iso3_code", "Year"],
        columns="Unit of measurement",
        values="VALUE",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    # load_unodc expects: Country, Year, Rate, Count (Country accepts ISO3 codes)
    wide = wide.rename(
        columns={
            "Iso3_code": "Country",
            "Rate per 100,000 population": "Rate",
            "Counts": "Count",
        }
    )[["Country", "Year", "Rate", "Count"]]

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "unodc_homicide.csv"
    wide.to_csv(dest, index=False)
    print(f"  Saved {len(wide)} rows -> {dest}")
    return dest


if __name__ == "__main__":
    raw_dir = Path(__file__).parents[3] / "data" / "raw"
    process_unodc_homicide(
        source_dir=raw_dir / "unodc",
        dest_dir=raw_dir / "unodc",
    )

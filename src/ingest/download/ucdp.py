"""Process UCDP/PRIO Armed Conflict Dataset into a CSV for load_ucdp().

Download UcdpPrioConflict_v*.csv from https://ucdp.uu.se/downloads/
Place in data/raw/ucdp/ then run this script.

Multi-country conflict locations (e.g. "Egypt, Israel") are expanded to one
row per country so load_ucdp() can aggregate to country-year binary indicators.
"""

from pathlib import Path

import pandas as pd

# UCDP uses parenthetical aliases and historical state names
_UCDP_NAME_MAP: dict[str, str | None] = {
    "Bosnia-Herzegovina": "BIH",
    "Brunei": "BRN",
    "Cambodia (Kampuchea)": "KHM",
    "DR Congo (Zaire)": "COD",
    "Hyderabad": None,  # historical princely state — not a modern country
    "Ivory Coast": "CIV",
    "Madagascar (Malagasy)": "MDG",
    "Myanmar (Burma)": "MMR",
    "Russia (Soviet Union)": "RUS",
    "Serbia (Yugoslavia)": "SRB",
    "South Vietnam": "VNM",
    "South Yemen": "YEM",
    "Turkey": "TUR",
    "Vietnam (North Vietnam)": "VNM",
    "Yemen (North Yemen)": "YEM",
    "Zimbabwe (Rhodesia)": "ZWE",
}


def process_ucdp(source_dir: Path, dest_dir: Path) -> Path:
    """Expand multi-country locations and clean names for load_ucdp()."""
    csv_files = list(source_dir.glob("UcdpPrioConflict_v*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No UcdpPrioConflict CSV found in {source_dir}")

    df = pd.read_csv(csv_files[0])
    print(f"  Read {len(df)} conflict-year rows from {csv_files[0].name}")

    rows = []
    for _, row in df.iterrows():
        # Expand compound locations: "Egypt, Israel" -> ["Egypt", "Israel"]
        countries = [c.strip() for c in str(row["location"]).split(",")]
        for country in countries:
            # Apply UCDP name map for aliases/historical states
            if country in _UCDP_NAME_MAP:
                mapped = _UCDP_NAME_MAP[country]
                if mapped is None:
                    continue  # skip historical non-country entities
                location = mapped  # use ISO3 directly — pycountry can look it up
            else:
                location = country

            rows.append(
                {
                    "location": location,
                    "year": row["year"],
                    "type_of_conflict": row["type_of_conflict"],
                    "intensity_level": row["intensity_level"],
                }
            )

    result = pd.DataFrame(rows)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "ucdp_conflicts.csv"
    result.to_csv(dest, index=False)
    print(f"  Saved {len(result)} rows -> {dest}")
    return dest


if __name__ == "__main__":
    raw_dir = Path(__file__).parents[3] / "data" / "raw" / "ucdp"
    process_ucdp(source_dir=raw_dir, dest_dir=raw_dir)

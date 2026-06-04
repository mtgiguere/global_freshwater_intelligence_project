import pandas as pd
import pycountry

_REQUIRED_COLUMNS = ["Country", "Year", "Refugees"]

COLUMN_NAMES: dict[str, str] = {
    "Year": "year",
    "Refugees": "refugee_outflow",
    "IDPs": "idp_count",
    "AsylumSeekers": "asylum_applications_origin",
}


def _to_iso3(name: str) -> str | None:
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


def load_unhcr(path):
    df = pd.read_csv(path)

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["iso3"] = df["Country"].map(_to_iso3)

    unmapped = df.loc[df["iso3"].isna(), "Country"].unique().tolist()
    if unmapped:
        raise ValueError(f"unmapped countries — no ISO3 code found: {unmapped}")

    return df.drop(columns=["Country"]).rename(columns=COLUMN_NAMES)

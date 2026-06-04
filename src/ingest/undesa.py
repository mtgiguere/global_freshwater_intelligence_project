import pandas as pd
import pycountry

_REQUIRED_COLUMNS = ["Country", "Year", "PopTotal"]

COLUMN_NAMES: dict[str, str] = {
    "Year": "year",
    "PopTotal": "population",
    "PopUrban": "population_urban",
    "PopRural": "population_rural",
}

# UN DESA publishes population in thousands
_THOUSANDS = 1_000


def _to_iso3(name: str) -> str | None:
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


def load_undesa(path):
    df = pd.read_csv(path)

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["iso3"] = df["Country"].map(_to_iso3)

    unmapped = df.loc[df["iso3"].isna(), "Country"].unique().tolist()
    if unmapped:
        raise ValueError(f"unmapped countries — no ISO3 code found: {unmapped}")

    df = df.drop(columns=["Country"]).rename(columns=COLUMN_NAMES)
    for col in ["population", "population_urban", "population_rural"]:
        if col in df.columns:
            df[col] = df[col] * _THOUSANDS
    return df

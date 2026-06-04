import pandas as pd
import pycountry

VARIABLE_NAMES: dict[str, str] = {
    "Renewable internal freshwater resources per capita": "renewable_freshwater_percap",
    "Total freshwater withdrawal": "total_withdrawal_km3",
    "Agricultural water withdrawal as % of total freshwater withdrawal": "agri_withdrawal_pct",
}

_REQUIRED_COLUMNS = ["Area", "Variable Name", "Year", "Value"]


def _to_iso3(name: str) -> str | None:
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


def load_aquastat(path):
    df = pd.read_csv(path)

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    present = [v for v in VARIABLE_NAMES if v in df["Variable Name"].values]
    if not present:
        raise ValueError(f"No recognised variables found. Expected one of: {list(VARIABLE_NAMES)}")

    df["iso3"] = df["Area"].map(_to_iso3)
    unmapped = df.loc[df["iso3"].isna(), "Area"].unique().tolist()
    if unmapped:
        raise ValueError(f"unmapped countries — no ISO3 code found: {unmapped}")
    df = df.drop(columns=["Area"])
    df = df.rename(columns={"Year": "year"})
    df = df.pivot_table(
        index=["iso3", "year"],
        columns="Variable Name",
        values="Value",
        aggfunc="first",
    ).reset_index()
    df.columns.name = None
    df = df.rename(columns=VARIABLE_NAMES)
    return df

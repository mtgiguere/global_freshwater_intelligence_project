import pandas as pd
import pycountry

_REQUIRED_COLUMNS = ["country", "event_date", "fatalities"]


def _to_iso3(name: str) -> str | None:
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


def load_acled(path):
    df = pd.read_csv(path)

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["year"] = pd.to_datetime(df["event_date"]).dt.year
    df["iso3"] = df["country"].map(_to_iso3)

    unmapped = df.loc[df["iso3"].isna(), "country"].unique().tolist()
    if unmapped:
        raise ValueError(f"unmapped countries — no ISO3 code found: {unmapped}")

    return df.groupby(["iso3", "year"], as_index=False).agg(
        acled_events_count=("fatalities", "size"), acled_fatalities=("fatalities", "sum")
    )

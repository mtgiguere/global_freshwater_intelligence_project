import pandas as pd
import pycountry

_REQUIRED_COLUMNS = ["location", "year"]


def _to_iso3(name: str) -> str | None:
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


def load_ucdp(path):
    df = pd.read_csv(path)

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["iso3"] = df["location"].map(_to_iso3)

    unmapped = df.loc[df["iso3"].isna(), "location"].unique().tolist()
    if unmapped:
        raise ValueError(f"unmapped countries — no ISO3 code found: {unmapped}")

    return (
        df.groupby(["iso3", "year"], as_index=False)
        .agg(ucdp_conflict_count=("location", "size"))
        .assign(ucdp_conflict_binary=1)
    )

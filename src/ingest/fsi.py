"""FSI (Fragile States Index) ingest.

Column names verified against Fund for Peace annual Excel files (2006-2023).
"""

import pandas as pd
import pycountry

COLUMN_NAMES: dict[str, str] = {
    "Total": "fsi_score",
    "C1: Security Apparatus": "fsi_c1_security",
    "C2: Factionalized Elites": "fsi_c2_factions",
    "C3: Group Grievance": "fsi_c3_group_grievance",
    "E1: Economy": "fsi_e1_economy",
    "E2: Economic Inequality": "fsi_e2_inequality",
    "E3: Human Flight and Brain Drain": "fsi_e3_human_flight",
    "P1: State Legitimacy": "fsi_p1_legitimacy",
    "P2: Public Services": "fsi_p2_public_services",
    "P3: Human Rights": "fsi_p3_human_rights",
    "S1: Demographic Pressures": "fsi_s1_demographics",
    "S2: Refugees and IDPs": "fsi_s2_refugees",
    "X1: External Intervention": "fsi_x1_intervention",
}

_REQUIRED_COLUMNS = ["Country", "Year", "Total"]

# Non-standard FSI country names → ISO3
_FSI_NAME_MAP: dict[str, str] = {
    "Cape Verde": "CPV",
    "Congo Democratic Republic": "COD",
    "Congo Republic": "COG",
    "Cote d'Ivoire": "CIV",
    "Guinea Bissau": "GNB",
    "Macedonia": "MKD",
    "Micronesia": "FSM",
    "Palestine": "PSE",
    "Russia": "RUS",
    "Swaziland": "SWZ",
    "Turkey": "TUR",
}

# Compound political entries that are not single countries — dropped silently
_COMPOUND_ENTRIES = {"Israel and West Bank"}


def _to_iso3(name: str) -> str | None:
    name = name.strip()
    if name in _FSI_NAME_MAP:
        return _FSI_NAME_MAP[name]
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


def load_fsi(path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["_name"] = df["Country"].str.strip()

    # Drop compound entries silently before the unmapped check
    df = df[~df["_name"].isin(_COMPOUND_ENTRIES)].copy()

    df["iso3"] = df["_name"].map(_to_iso3)

    unmapped = df.loc[df["iso3"].isna(), "_name"].unique().tolist()
    if unmapped:
        raise ValueError(f"unmapped countries — no ISO3 code found: {unmapped}")

    # Drop administrative/derived columns that are not analytical variables
    to_drop = ["Country", "_name", "Rank", "Change from Previous Year"]
    drop = [c for c in to_drop if c in df.columns]
    df = df.drop(columns=drop)
    df = df.rename(columns={"Year": "year", **COLUMN_NAMES})

    # Extract integer year — older FSI files store Year as a datetime object
    df["year"] = df["year"].apply(lambda v: int(str(v)[:4]))

    return df.reset_index(drop=True)

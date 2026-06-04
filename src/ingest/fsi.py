import pandas as pd
import pycountry

COLUMN_NAMES: dict[str, str] = {
    "Total": "fsi_score",
    "P1: State Legitimacy": "fsi_p1_legitimacy",
    "C3: Human Rights": "fsi_c3_human_rights",
    "P2: Public Services": "fsi_p2_public_services",
    "P3: Human Rights": "fsi_p3_human_rights",
    "E1: Economy": "fsi_e1_economy",
    "E2: Economic Inequality": "fsi_e2_inequality",
    "E3: Human Flight": "fsi_e3_human_flight",
    "C1: Security Apparatus": "fsi_c1_security",
    "C2: Factionalized Elites": "fsi_c2_factions",
    "S1: Demographic Pressures": "fsi_s1_demographics",
    "S2: Refugees & IDPs": "fsi_s2_refugees",
    "X1: External Intervention": "fsi_x1_intervention",
}

_REQUIRED_COLUMNS = ["Country", "Year", "Total"]


def _to_iso3(name: str) -> str | None:
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


def load_fsi(path):
    df = pd.read_csv(path)

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["iso3"] = df["Country"].map(_to_iso3)

    unmapped = df.loc[df["iso3"].isna(), "Country"].unique().tolist()
    if unmapped:
        raise ValueError(f"unmapped countries — no ISO3 code found: {unmapped}")

    df = df.drop(columns=["Country"])
    df = df.rename(columns={"Year": "year", **COLUMN_NAMES})
    return df

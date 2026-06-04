import pandas as pd

INDICATOR_NAMES: dict[str, str] = {
    "NY.GDP.PCAP.KD": "gdp_pc_ppp",
    "SI.POV.GINI": "gini",
    "HDI": "hdi",
    "NV.AGR.TOTL.ZS": "agri_value_added_pct_gdp",
    "SH.H2O.SMDW.ZS": "safe_water_access_pct",
}

_REQUIRED_COLUMNS = ["Country Code", "Indicator Code"]


def load_worldbank(path):
    df = pd.read_csv(path)

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    present = [c for c in df["Indicator Code"].unique() if c in INDICATOR_NAMES]
    if not present:
        known = list(INDICATOR_NAMES)
        raise ValueError(f"No recognised indicators found. Expected one of: {known}")

    df = df.rename(columns={"Country Code": "iso3"})
    df = df.drop(columns=["Country Name"])
    year_cols = [c for c in df.columns if c.isdigit()]
    df = df.melt(
        id_vars=["iso3", "Indicator Name", "Indicator Code"],
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )
    df["year"] = df["year"].astype("int64")
    df = df[df["Indicator Code"].isin(INDICATOR_NAMES)].copy()
    df["Indicator Code"] = df["Indicator Code"].map(INDICATOR_NAMES)
    df = df.pivot_table(
        index=["iso3", "year"],
        columns="Indicator Code",
        values="value",
        aggfunc="first",
    ).reset_index()
    df.columns.name = None
    return df

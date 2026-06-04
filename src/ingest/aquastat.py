"""AQUASTAT data ingest module.

Parses the FAO AQUASTAT bulk CSV (long format) and produces a validated
wide country-year panel ready for joining to the Master Panel.

Pipeline:
    parse_raw_csv → filter_variables → pivot_to_wide → map_country_codes → validate_schema
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import IO

import pandas as pd
import pycountry

# FAO AQUASTAT variable name → canonical snake_case column name
AQUASTAT_VARIABLES: dict[str, str] = {
    "Renewable internal freshwater resources per capita": "renewable_freshwater_percap",
    "Total freshwater withdrawal": "total_withdrawal_km3",
    "Agricultural water withdrawal as % of total freshwater withdrawal": "agri_withdrawal_pct",
}

_REQUIRED_RAW_COLUMNS: list[str] = ["Area", "Variable Name", "Year", "Value"]
_REQUIRED_WIDE_COLUMNS: list[str] = ["iso3", "year", "renewable_freshwater_percap"]
_ISO3_RE = re.compile(r"^[A-Z]{3}$")


def parse_raw_csv(path: Path | IO) -> pd.DataFrame:
    """Load the raw AQUASTAT long-format CSV and enforce column contract."""
    df = pd.read_csv(path)
    missing = [c for c in _REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["Year"] = df["Year"].astype("int64")
    return df


def filter_variables(df: pd.DataFrame, variables: dict[str, str]) -> pd.DataFrame:
    """Keep only rows whose Variable Name appears in the mapping keys."""
    mask = df["Variable Name"].isin(variables.keys())
    result = df[mask].copy()
    if result.empty:
        sample = df["Variable Name"].unique()[:10].tolist()
        raise ValueError(f"No requested variables found in data. Sample of available: {sample}")
    return result


def pivot_to_wide(df: pd.DataFrame, variables: dict[str, str]) -> pd.DataFrame:
    """Pivot long-format rows to one row per (Area, Year) with canonical column names."""
    pivoted = df.pivot_table(
        index=["Area", "Year"],
        columns="Variable Name",
        values="Value",
        aggfunc="first",
    ).reset_index()
    pivoted.columns.name = None
    rename_map = {k: v for k, v in variables.items() if k in pivoted.columns}
    return pivoted.rename(columns=rename_map)


def map_country_codes(df: pd.DataFrame, country_col: str = "Area") -> pd.DataFrame:
    """Add iso3 column by mapping country names to ISO 3166-1 alpha-3 codes.

    Unknown countries get NaN. validate_schema will catch them downstream.
    """

    def _lookup(name: str) -> str | None:
        try:
            return pycountry.countries.lookup(name).alpha_3
        except LookupError:
            return None

    result = df.copy()
    result["iso3"] = result[country_col].map(_lookup)
    return result


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if the wide panel DataFrame violates the schema contract."""
    for col in _REQUIRED_WIDE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    if df["iso3"].isna().any():
        bad = df.loc[df["iso3"].isna(), "Area"].tolist() if "Area" in df.columns else ["(unknown)"]
        raise ValueError(f"iso3 is null for {len(bad)} row(s) — unmapped countries: {bad[:5]}")

    invalid_iso3 = df[~df["iso3"].str.match(r"^[A-Z]{3}$")]
    if not invalid_iso3.empty:
        bad_values = invalid_iso3["iso3"].tolist()[:5]
        raise ValueError(f"iso3 values must be 3 uppercase letters. Invalid: {bad_values}")

    if "renewable_freshwater_percap" in df.columns:
        n_negative = (df["renewable_freshwater_percap"] < 0).sum()
        if n_negative > 0:
            raise ValueError(f"renewable_freshwater_percap contains {n_negative} negative value(s)")

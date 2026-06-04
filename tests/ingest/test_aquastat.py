"""Tests for the AQUASTAT ingest module.

Written before implementation — TDD.
Run pytest to confirm all RED before touching src/ingest/aquastat.py.
"""

import io
import textwrap

import pandas as pd
import pytest

from src.ingest.aquastat import (
    AQUASTAT_VARIABLES,
    filter_variables,
    map_country_codes,
    parse_raw_csv,
    pivot_to_wide,
    validate_schema,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

_VALID_CSV = textwrap.dedent("""\
    Area,Variable Name,Year,Value,Symbol,Md
    Afghanistan,Renewable internal freshwater resources per capita,2010,1.5,,
    Afghanistan,Total freshwater withdrawal,2010,20.3,,
    France,Renewable internal freshwater resources per capita,2010,3.2,,
    France,Total freshwater withdrawal,2010,31.1,,
""")

_MISSING_COLUMNS_CSV = "Country,Var,Yr\nAfghanistan,foo,2010\n"


def _make_long_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal long-format AQUASTAT DataFrame."""
    defaults = {
        "Area": "Afghanistan",
        "Variable Name": "Renewable internal freshwater resources per capita",
        "Year": 2010,
        "Value": 1.5,
        "Symbol": "",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _make_wide_df() -> pd.DataFrame:
    """Build a minimal valid post-pivot wide panel DataFrame."""
    return pd.DataFrame(
        {
            "iso3": ["AFG", "FRA"],
            "year": [2010, 2010],
            "renewable_freshwater_percap": [1.5, 3.2],
            "total_withdrawal_km3": [20.3, 31.1],
            "agri_withdrawal_pct": [95.2, 72.1],
        }
    )


# ---------------------------------------------------------------------------
# parse_raw_csv
# ---------------------------------------------------------------------------


def test_parse_raw_csv_returns_dataframe():
    """parse_raw_csv must return a pandas DataFrame."""
    result = parse_raw_csv(io.StringIO(_VALID_CSV))
    assert isinstance(result, pd.DataFrame)


def test_parse_raw_csv_preserves_required_columns():
    """All four required raw columns must be present in the result."""
    result = parse_raw_csv(io.StringIO(_VALID_CSV))
    for col in ["Area", "Variable Name", "Year", "Value"]:
        assert col in result.columns, f"Required column missing: {col}"


def test_parse_raw_csv_raises_on_missing_required_columns():
    """parse_raw_csv raises ValueError naming the missing columns."""
    with pytest.raises(ValueError, match="Missing required columns"):
        parse_raw_csv(io.StringIO(_MISSING_COLUMNS_CSV))


def test_parse_raw_csv_year_is_integer_dtype():
    """Year column must be an integer dtype, not float or object."""
    result = parse_raw_csv(io.StringIO(_VALID_CSV))
    assert pd.api.types.is_integer_dtype(result["Year"])


# ---------------------------------------------------------------------------
# filter_variables
# ---------------------------------------------------------------------------


def test_filter_variables_keeps_only_requested_rows():
    """filter_variables returns only rows matching the variable mapping keys."""
    df = _make_long_df(
        [
            {"Variable Name": "Renewable internal freshwater resources per capita", "Value": 1.5},
            {"Variable Name": "Some irrelevant variable", "Value": 99.0},
        ]
    )
    result = filter_variables(df, AQUASTAT_VARIABLES)
    assert len(result) == 1
    assert result.iloc[0]["Variable Name"] == "Renewable internal freshwater resources per capita"


def test_filter_variables_raises_if_none_of_requested_variables_exist():
    """filter_variables raises ValueError if none of the keys appear in the data."""
    df = _make_long_df([{"Variable Name": "Some other variable"}])
    with pytest.raises(ValueError, match="No requested variables found"):
        filter_variables(df, {"Nonexistent variable": "col_name"})


# ---------------------------------------------------------------------------
# pivot_to_wide
# ---------------------------------------------------------------------------


def test_pivot_to_wide_produces_one_row_per_area_year():
    """pivot_to_wide must produce exactly one row per (Area, Year) pair."""
    df = pd.DataFrame(
        {
            "Area": ["Afghanistan", "Afghanistan", "France"],
            "Variable Name": [
                "Renewable internal freshwater resources per capita",
                "Total freshwater withdrawal",
                "Renewable internal freshwater resources per capita",
            ],
            "Year": [2010, 2010, 2010],
            "Value": [1.5, 20.3, 3.2],
            "Symbol": ["", "", ""],
        }
    )
    result = pivot_to_wide(df, AQUASTAT_VARIABLES)
    assert len(result) == 2


def test_pivot_to_wide_renames_columns_using_variable_mapping():
    """Variable columns must use canonical snake_case names from the mapping."""
    df = pd.DataFrame(
        {
            "Area": ["Afghanistan"],
            "Variable Name": ["Renewable internal freshwater resources per capita"],
            "Year": [2010],
            "Value": [1.5],
            "Symbol": [""],
        }
    )
    result = pivot_to_wide(df, AQUASTAT_VARIABLES)
    assert "renewable_freshwater_percap" in result.columns
    assert "Renewable internal freshwater resources per capita" not in result.columns


# ---------------------------------------------------------------------------
# map_country_codes
# ---------------------------------------------------------------------------


def test_map_country_codes_adds_iso3_column():
    """map_country_codes must add an iso3 column."""
    df = pd.DataFrame({"Area": ["Afghanistan", "France"], "Year": [2010, 2010]})
    result = map_country_codes(df)
    assert "iso3" in result.columns


def test_map_country_codes_known_countries_produce_valid_iso3():
    """iso3 values for known countries must be exactly 3 uppercase ASCII letters."""
    df = pd.DataFrame({"Area": ["Afghanistan", "France"], "Year": [2010, 2010]})
    result = map_country_codes(df)
    known = result.dropna(subset=["iso3"])
    assert known["iso3"].str.match(r"^[A-Z]{3}$").all()


def test_map_country_codes_unknown_country_gets_null_not_crash():
    """Countries with no ISO3 mapping produce NaN iso3; the function must not raise."""
    df = pd.DataFrame({"Area": ["NotARealCountry"], "Year": [2010]})
    result = map_country_codes(df)
    assert result["iso3"].isna().all()


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------


def test_validate_schema_passes_for_valid_dataframe():
    """validate_schema must not raise for a correctly structured wide panel."""
    validate_schema(_make_wide_df())  # no exception = pass


def test_validate_schema_raises_if_iso3_column_absent():
    """validate_schema raises ValueError mentioning 'iso3' when column is missing."""
    df = _make_wide_df().drop(columns=["iso3"])
    with pytest.raises(ValueError, match="iso3"):
        validate_schema(df)


def test_validate_schema_raises_if_year_column_absent():
    """validate_schema raises ValueError mentioning 'year' when column is missing."""
    df = _make_wide_df().drop(columns=["year"])
    with pytest.raises(ValueError, match="year"):
        validate_schema(df)


def test_validate_schema_raises_for_null_iso3_values():
    """validate_schema raises ValueError if any iso3 value is null."""
    df = _make_wide_df().copy()
    df.loc[0, "iso3"] = None
    with pytest.raises(ValueError, match="iso3"):
        validate_schema(df)


def test_validate_schema_raises_for_invalid_iso3_format():
    """validate_schema raises ValueError if any iso3 is not 3 uppercase letters."""
    df = _make_wide_df().copy()
    df.loc[0, "iso3"] = "afghanistan"
    with pytest.raises(ValueError, match="iso3"):
        validate_schema(df)


def test_validate_schema_raises_for_negative_freshwater_values():
    """validate_schema raises ValueError if renewable_freshwater_percap is negative."""
    df = _make_wide_df().copy()
    df.loc[0, "renewable_freshwater_percap"] = -1.0
    with pytest.raises(ValueError, match="renewable_freshwater_percap"):
        validate_schema(df)

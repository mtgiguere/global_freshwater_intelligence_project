"""AQUASTAT ingest — strict TDD, one test at a time."""

import io

import pandas as pd
import pytest

from src.ingest.aquastat import load_aquastat

_VALID_CSV = (
    "Area,Variable Name,Year,Value,Symbol,Md\n"
    "Afghanistan,Renewable internal freshwater resources per capita,2010,1.5,,\n"
    "Afghanistan,Total freshwater withdrawal,2010,20.3,,\n"
    "France,Renewable internal freshwater resources per capita,2010,3.2,,\n"
    "France,Total freshwater withdrawal,2010,31.1,,\n"
)


def test_load_aquastat_returns_dataframe():
    """The consumer gets a DataFrame back — nothing more, nothing less."""
    result = load_aquastat(io.StringIO(_VALID_CSV))
    assert isinstance(result, pd.DataFrame)


def test_load_aquastat_identifies_countries_by_iso3():
    """Countries must be identified by ISO3 code, not by name."""
    result = load_aquastat(io.StringIO(_VALID_CSV))
    assert "iso3" in result.columns
    assert "Area" not in result.columns


def test_load_aquastat_is_one_row_per_country_year():
    """Output must be wide format — one row per (iso3, year), not one per variable."""
    result = load_aquastat(io.StringIO(_VALID_CSV))
    assert result.duplicated(subset=["iso3", "year"]).sum() == 0


def test_load_aquastat_water_columns_use_canonical_names():
    """Variable columns must be snake_case canonical names, not raw AQUASTAT strings."""
    result = load_aquastat(io.StringIO(_VALID_CSV))
    assert "renewable_freshwater_percap" in result.columns
    assert "total_withdrawal_km3" in result.columns
    assert "Renewable internal freshwater resources per capita" not in result.columns


def test_load_aquastat_raises_on_missing_required_columns():
    """A CSV missing Area, Variable Name, Year, or Value must fail with a clear error."""
    bad_csv = "Country,Var,Yr\nAfghanistan,foo,2010\n"
    with pytest.raises(ValueError, match="Missing required columns"):
        load_aquastat(io.StringIO(bad_csv))


def test_load_aquastat_year_is_integer_dtype():
    """Year must be an integer dtype — float years break panel joins downstream."""
    result = load_aquastat(io.StringIO(_VALID_CSV))
    assert pd.api.types.is_integer_dtype(result["year"])


def test_load_aquastat_raises_if_no_known_variables_present():
    """A CSV with no recognised AQUASTAT variables must fail, not return an empty panel."""
    csv = "Area,Variable Name,Year,Value,Symbol,Md\nAfghanistan,Some unknown metric,2010,1.5,,\n"
    with pytest.raises(ValueError, match="No recognised variables"):
        load_aquastat(io.StringIO(csv))


def test_load_aquastat_raises_if_any_country_cannot_be_mapped():
    """Unmapped countries must cause a loud failure, not silently produce NaN iso3 rows."""
    csv = (
        "Area,Variable Name,Year,Value,Symbol,Md\n"
        "Afghanistan,Renewable internal freshwater resources per capita,2010,1.5,,\n"
        "NotARealCountry,Renewable internal freshwater resources per capita,2010,0.5,,\n"
    )
    with pytest.raises(ValueError, match="unmapped"):
        load_aquastat(io.StringIO(csv))

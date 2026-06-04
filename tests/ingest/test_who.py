"""WHO/GHDx health data ingest — strict TDD, one test at a time.

WHO publishes life expectancy, under-5 mortality, and disease burden
as a country-year CSV. Already in the right shape — map to iso3, rename.
"""

import io

import pandas as pd
import pytest

from src.ingest.who import load_who

_VALID_CSV = (
    "Country,Year,LifeExpectancy,U5MR,DiarrhoeaDALY\n"
    "Afghanistan,2015,63.2,70.1,4821.3\n"
    "France,2015,82.4,4.2,89.7\n"
    "Afghanistan,2016,63.8,67.4,4710.5\n"
)


def test_load_who_returns_dataframe():
    """The consumer gets a DataFrame back."""
    result = load_who(io.StringIO(_VALID_CSV))
    assert isinstance(result, pd.DataFrame)


def test_load_who_has_iso3_and_year_columns():
    """Output must use iso3 not country name; year must be present."""
    result = load_who(io.StringIO(_VALID_CSV))
    assert "iso3" in result.columns
    assert "year" in result.columns
    assert "Country" not in result.columns


def test_load_who_is_one_row_per_country_year():
    """WHO data is already country-year format — no duplicates allowed."""
    result = load_who(io.StringIO(_VALID_CSV))
    assert result.duplicated(subset=["iso3", "year"]).sum() == 0


def test_load_who_columns_use_canonical_names():
    """Health indicators must use snake_case canonical names."""
    result = load_who(io.StringIO(_VALID_CSV))
    assert "life_expectancy" in result.columns
    assert "u5mr" in result.columns
    assert "diarrhoeal_daly" in result.columns
    assert "LifeExpectancy" not in result.columns


def test_load_who_values_are_preserved_correctly():
    """Health indicator values must survive the rename without modification."""
    result = load_who(io.StringIO(_VALID_CSV))
    afg = result[(result["iso3"] == "AFG") & (result["year"] == 2015)].iloc[0]
    assert abs(afg["life_expectancy"] - 63.2) < 1e-6
    assert abs(afg["u5mr"] - 70.1) < 1e-6
    assert abs(afg["diarrhoeal_daly"] - 4821.3) < 1e-6


def test_load_who_raises_on_missing_required_columns():
    """A CSV without Country, Year, or LifeExpectancy must fail with a clear error."""
    bad_csv = "Nation,Yr,LE\nAfghanistan,2015,63.2\n"
    with pytest.raises(ValueError, match="Missing required columns"):
        load_who(io.StringIO(bad_csv))


def test_load_who_raises_if_any_country_cannot_be_mapped():
    """Unmapped country names must raise loudly."""
    csv = (
        "Country,Year,LifeExpectancy,U5MR,DiarrhoeaDALY\n"
        "Afghanistan,2015,63.2,70.1,4821.3\n"
        "NotARealCountry,2015,55.0,50.0,1000.0\n"
    )
    with pytest.raises(ValueError, match="unmapped"):
        load_who(io.StringIO(csv))

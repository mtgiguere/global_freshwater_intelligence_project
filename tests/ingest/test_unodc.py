"""UNODC homicide ingest — strict TDD, one test at a time.

UNODC publishes homicide rates and counts as a country-year CSV.
Already in the right shape — map to iso3, rename, validate.
"""

import io

import pandas as pd
import pytest

from src.ingest.unodc import load_unodc

_VALID_CSV = (
    "Country,Year,Rate,Count\n"
    "Afghanistan,2015,6.7,1820\n"
    "France,2015,1.2,776\n"
    "Afghanistan,2016,7.1,1950\n"
)


def test_load_unodc_returns_dataframe():
    """The consumer gets a DataFrame back."""
    result = load_unodc(io.StringIO(_VALID_CSV))
    assert isinstance(result, pd.DataFrame)


def test_load_unodc_has_iso3_and_year_columns():
    """Output must use iso3 not country name; year must be present."""
    result = load_unodc(io.StringIO(_VALID_CSV))
    assert "iso3" in result.columns
    assert "year" in result.columns
    assert "Country" not in result.columns


def test_load_unodc_is_one_row_per_country_year():
    """UNODC is already country-year format — no duplicates allowed."""
    result = load_unodc(io.StringIO(_VALID_CSV))
    assert result.duplicated(subset=["iso3", "year"]).sum() == 0


def test_load_unodc_columns_use_canonical_names():
    """Rate and Count must use snake_case canonical names."""
    result = load_unodc(io.StringIO(_VALID_CSV))
    assert "homicide_rate" in result.columns
    assert "homicide_count" in result.columns
    assert "Rate" not in result.columns
    assert "Count" not in result.columns


def test_load_unodc_values_are_preserved_correctly():
    """Rate and count values must survive the rename without modification."""
    result = load_unodc(io.StringIO(_VALID_CSV))
    afg = result[(result["iso3"] == "AFG") & (result["year"] == 2015)].iloc[0]
    assert abs(afg["homicide_rate"] - 6.7) < 1e-6
    assert afg["homicide_count"] == 1820


def test_load_unodc_raises_on_missing_required_columns():
    """A CSV without Country, Year, or Rate must fail with a clear error."""
    bad_csv = "Nation,Yr,Homicides\nAfghanistan,2015,1820\n"
    with pytest.raises(ValueError, match="Missing required columns"):
        load_unodc(io.StringIO(bad_csv))


def test_load_unodc_raises_if_any_country_cannot_be_mapped():
    """Unmapped country names must raise loudly."""
    csv = "Country,Year,Rate,Count\nAfghanistan,2015,6.7,1820\nNotARealCountry,2015,3.2,100\n"
    with pytest.raises(ValueError, match="unmapped"):
        load_unodc(io.StringIO(csv))

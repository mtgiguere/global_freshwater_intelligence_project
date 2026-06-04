"""UN DESA World Population Prospects ingest — strict TDD, one test at a time.

UN DESA publishes population totals and urban/rural split as a country-year CSV.
Population values are published in thousands — we convert to absolute counts.
"""

import io

import pandas as pd
import pytest

from src.ingest.undesa import load_undesa

# UN DESA publishes population in thousands (e.g. 32527 = 32,527,000)
_VALID_CSV = (
    "Country,Year,PopTotal,PopUrban,PopRural\n"
    "Afghanistan,2015,32527,8661,23866\n"
    "France,2015,64457,52428,12029\n"
    "Afghanistan,2016,33370,9019,24351\n"
)


def test_load_undesa_returns_dataframe():
    """The consumer gets a DataFrame back."""
    result = load_undesa(io.StringIO(_VALID_CSV))
    assert isinstance(result, pd.DataFrame)


def test_load_undesa_has_iso3_and_year_columns():
    """Output must use iso3 not country name; year must be present."""
    result = load_undesa(io.StringIO(_VALID_CSV))
    assert "iso3" in result.columns
    assert "year" in result.columns
    assert "Country" not in result.columns


def test_load_undesa_is_one_row_per_country_year():
    """UN DESA is already country-year format — no duplicates allowed."""
    result = load_undesa(io.StringIO(_VALID_CSV))
    assert result.duplicated(subset=["iso3", "year"]).sum() == 0


def test_load_undesa_columns_use_canonical_names():
    """Population columns must use snake_case canonical names."""
    result = load_undesa(io.StringIO(_VALID_CSV))
    assert "population" in result.columns
    assert "population_urban" in result.columns
    assert "population_rural" in result.columns
    assert "PopTotal" not in result.columns


def test_load_undesa_converts_thousands_to_absolute():
    """UN DESA publishes in thousands — output must be absolute population counts."""
    result = load_undesa(io.StringIO(_VALID_CSV))
    afg = result[(result["iso3"] == "AFG") & (result["year"] == 2015)].iloc[0]
    assert afg["population"] == 32_527_000
    assert afg["population_urban"] == 8_661_000
    assert afg["population_rural"] == 23_866_000


def test_load_undesa_raises_on_missing_required_columns():
    """A CSV without Country, Year, or PopTotal must fail with a clear error."""
    bad_csv = "Nation,Yr,Pop\nAfghanistan,2015,32527\n"
    with pytest.raises(ValueError, match="Missing required columns"):
        load_undesa(io.StringIO(bad_csv))


def test_load_undesa_converts_thousands_when_urban_rural_absent():
    """Conversion must work when only PopTotal is present — urban/rural are optional."""
    csv = "Country,Year,PopTotal\nAfghanistan,2015,32527\n"
    result = load_undesa(io.StringIO(csv))
    afg = result[(result["iso3"] == "AFG") & (result["year"] == 2015)].iloc[0]
    assert afg["population"] == 32_527_000
    assert "population_urban" not in result.columns


def test_load_undesa_raises_if_any_country_cannot_be_mapped():
    """Unmapped country names must raise loudly."""
    csv = (
        "Country,Year,PopTotal,PopUrban,PopRural\n"
        "Afghanistan,2015,32527,8661,23866\n"
        "NotARealCountry,2015,5000,2000,3000\n"
    )
    with pytest.raises(ValueError, match="unmapped"):
        load_undesa(io.StringIO(csv))

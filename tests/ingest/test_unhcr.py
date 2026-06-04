"""UNHCR displacement ingest — strict TDD, one test at a time.

UNHCR publishes refugee outflows, IDP counts, and asylum applications
by country of origin as a country-year CSV.
"""

import io

import pandas as pd
import pytest

from src.ingest.unhcr import load_unhcr

_VALID_CSV = (
    "Country,Year,Refugees,IDPs,AsylumSeekers\n"
    "Afghanistan,2015,2593368,1200000,185328\n"
    "France,2015,6542,0,12400\n"
    "Afghanistan,2016,2502606,1200000,196278\n"
)


def test_load_unhcr_returns_dataframe():
    """The consumer gets a DataFrame back."""
    result = load_unhcr(io.StringIO(_VALID_CSV))
    assert isinstance(result, pd.DataFrame)


def test_load_unhcr_has_iso3_and_year_columns():
    """Output must use iso3 not country name; year must be present."""
    result = load_unhcr(io.StringIO(_VALID_CSV))
    assert "iso3" in result.columns
    assert "year" in result.columns
    assert "Country" not in result.columns


def test_load_unhcr_is_one_row_per_country_year():
    """UNHCR is already country-year format — no duplicates allowed."""
    result = load_unhcr(io.StringIO(_VALID_CSV))
    assert result.duplicated(subset=["iso3", "year"]).sum() == 0


def test_load_unhcr_columns_use_canonical_names():
    """Displacement indicators must use snake_case canonical names."""
    result = load_unhcr(io.StringIO(_VALID_CSV))
    assert "refugee_outflow" in result.columns
    assert "idp_count" in result.columns
    assert "asylum_applications_origin" in result.columns
    assert "Refugees" not in result.columns


def test_load_unhcr_values_are_preserved_correctly():
    """Displacement values must survive the rename without modification."""
    result = load_unhcr(io.StringIO(_VALID_CSV))
    afg = result[(result["iso3"] == "AFG") & (result["year"] == 2015)].iloc[0]
    assert afg["refugee_outflow"] == 2593368
    assert afg["idp_count"] == 1200000
    assert afg["asylum_applications_origin"] == 185328


def test_load_unhcr_raises_on_missing_required_columns():
    """A CSV without Country, Year, or Refugees must fail with a clear error."""
    bad_csv = "Nation,Yr,Displaced\nAfghanistan,2015,2593368\n"
    with pytest.raises(ValueError, match="Missing required columns"):
        load_unhcr(io.StringIO(bad_csv))


def test_load_unhcr_raises_if_any_country_cannot_be_mapped():
    """Unmapped country names must raise loudly."""
    csv = (
        "Country,Year,Refugees,IDPs,AsylumSeekers\n"
        "Afghanistan,2015,2593368,1200000,185328\n"
        "NotARealCountry,2015,1000,500,200\n"
    )
    with pytest.raises(ValueError, match="unmapped"):
        load_unhcr(io.StringIO(csv))

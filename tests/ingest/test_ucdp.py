"""UCDP armed conflict ingest — strict TDD, one test at a time.

UCDP/PRIO Armed Conflict Dataset: one row per active conflict per year.
Aggregates to a country-year panel with a binary conflict indicator and count.

From _VALID_CSV:
  AFG 2010: 2 active conflicts  -> binary=1, count=2
  AFG 2011: 1 active conflict   -> binary=1, count=1
  FRA 2010: 1 active conflict   -> binary=1, count=1
"""

import io

import pandas as pd
import pytest

from src.ingest.ucdp import load_ucdp

# One row per active conflict-year (UCDP/PRIO format)
_VALID_CSV = (
    "location,year,type_of_conflict,intensity_level\n"
    "Afghanistan,2010,3,2\n"
    "Afghanistan,2010,3,1\n"
    "Afghanistan,2011,3,2\n"
    "France,2010,1,1\n"
)


def test_load_ucdp_returns_dataframe():
    """The consumer gets a DataFrame back."""
    result = load_ucdp(io.StringIO(_VALID_CSV))
    assert isinstance(result, pd.DataFrame)


def test_load_ucdp_has_iso3_and_year_columns():
    """Output must identify rows by iso3 and year, not by location name."""
    result = load_ucdp(io.StringIO(_VALID_CSV))
    assert "iso3" in result.columns
    assert "year" in result.columns
    assert "location" not in result.columns


def test_load_ucdp_is_one_row_per_country_year():
    """Multiple conflict rows for the same country-year must collapse into one."""
    result = load_ucdp(io.StringIO(_VALID_CSV))
    assert result.duplicated(subset=["iso3", "year"]).sum() == 0


def test_load_ucdp_aggregates_binary_and_count_correctly():
    """Binary flag must be 1 when any conflict exists; count must equal active conflicts.

    From _VALID_CSV:
      AFG 2010: 2 conflicts -> binary=1, count=2
      AFG 2011: 1 conflict  -> binary=1, count=1
      FRA 2010: 1 conflict  -> binary=1, count=1
    """
    result = load_ucdp(io.StringIO(_VALID_CSV))

    afg_2010 = result[(result["iso3"] == "AFG") & (result["year"] == 2010)].iloc[0]
    assert afg_2010["ucdp_conflict_binary"] == 1
    assert afg_2010["ucdp_conflict_count"] == 2

    afg_2011 = result[(result["iso3"] == "AFG") & (result["year"] == 2011)].iloc[0]
    assert afg_2011["ucdp_conflict_binary"] == 1
    assert afg_2011["ucdp_conflict_count"] == 1

    fra_2010 = result[(result["iso3"] == "FRA") & (result["year"] == 2010)].iloc[0]
    assert fra_2010["ucdp_conflict_binary"] == 1
    assert fra_2010["ucdp_conflict_count"] == 1


def test_load_ucdp_raises_on_missing_required_columns():
    """A CSV without location or year must fail with a clear error."""
    bad_csv = "country,date,conflict_type\nAfghanistan,2010,3\n"
    with pytest.raises(ValueError, match="Missing required columns"):
        load_ucdp(io.StringIO(bad_csv))


def test_load_ucdp_raises_if_any_country_cannot_be_mapped():
    """Unmapped location names must raise loudly, not produce NaN rows."""
    csv = (
        "location,year,type_of_conflict,intensity_level\n"
        "Afghanistan,2010,3,2\n"
        "NotARealCountry,2010,3,1\n"
    )
    with pytest.raises(ValueError, match="unmapped"):
        load_ucdp(io.StringIO(csv))

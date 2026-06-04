"""ACLED political violence ingest — strict TDD, one test at a time.

ACLED is event-level: one row per violent event with date and fatality count.
The pipeline aggregates to a country-year panel of event counts and fatalities.
"""

import io

import pandas as pd
import pytest

from src.ingest.acled import load_acled

# One row per event — two Afghan events in 2010, one in 2011, one French in 2010
_VALID_CSV = (
    "country,event_date,event_type,fatalities\n"
    "Afghanistan,2010-03-15,Battles,5\n"
    "Afghanistan,2010-07-22,Battles,3\n"
    "Afghanistan,2011-01-10,Violence against civilians,2\n"
    "France,2010-05-01,Protests,0\n"
)


def test_load_acled_returns_dataframe():
    """The consumer gets a DataFrame back."""
    result = load_acled(io.StringIO(_VALID_CSV))
    assert isinstance(result, pd.DataFrame)


def test_load_acled_has_iso3_and_year_columns():
    """Output must identify rows by iso3 and year, not by country name or date."""
    result = load_acled(io.StringIO(_VALID_CSV))
    assert "iso3" in result.columns
    assert "year" in result.columns
    assert "country" not in result.columns
    assert "event_date" not in result.columns


def test_load_acled_is_one_row_per_country_year():
    """Multiple events for the same country-year must be collapsed into one row."""
    result = load_acled(io.StringIO(_VALID_CSV))
    assert result.duplicated(subset=["iso3", "year"]).sum() == 0


def test_load_acled_aggregates_counts_and_fatalities_correctly():
    """Event counts and fatalities must be summed correctly per country-year.

    From _VALID_CSV:
      AFG 2010: 2 events, 5+3=8 fatalities
      AFG 2011: 1 event,  2   fatalities
      FRA 2010: 1 event,  0   fatalities
    """
    result = load_acled(io.StringIO(_VALID_CSV))

    afg_2010 = result[(result["iso3"] == "AFG") & (result["year"] == 2010)].iloc[0]
    assert afg_2010["acled_events_count"] == 2
    assert afg_2010["acled_fatalities"] == 8

    afg_2011 = result[(result["iso3"] == "AFG") & (result["year"] == 2011)].iloc[0]
    assert afg_2011["acled_events_count"] == 1
    assert afg_2011["acled_fatalities"] == 2

    fra_2010 = result[(result["iso3"] == "FRA") & (result["year"] == 2010)].iloc[0]
    assert fra_2010["acled_events_count"] == 1
    assert fra_2010["acled_fatalities"] == 0


def test_load_acled_raises_on_missing_required_columns():
    """A CSV without country, event_date, or fatalities must fail with a clear error."""
    bad_csv = "location,date,deaths\nKabul,2010-01-01,3\n"
    with pytest.raises(ValueError, match="Missing required columns"):
        load_acled(io.StringIO(bad_csv))


def test_load_acled_raises_if_any_country_cannot_be_mapped():
    """Unmapped country names must raise loudly, not silently produce NaN rows."""
    csv = (
        "country,event_date,event_type,fatalities\n"
        "Afghanistan,2010-03-15,Battles,5\n"
        "NotARealCountry,2010-04-01,Protests,0\n"
    )
    with pytest.raises(ValueError, match="unmapped"):
        load_acled(io.StringIO(csv))


def test_load_acled_year_is_integer_dtype():
    """Year column must be integer — float years break panel joins downstream."""
    result = load_acled(io.StringIO(_VALID_CSV))
    assert pd.api.types.is_integer_dtype(result["year"])

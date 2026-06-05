"""FSI (Fragile States Index) ingest — strict TDD, one test at a time.

Fund for Peace publishes one row per country per year with a composite
FSI score and 12 sub-indicator scores. Already country-year format —
no pivot or aggregation needed, just map to iso3 and rename columns.

Column names verified against real FSI Excel files (2006-2023).
"""

import io

import pandas as pd
import pytest

from src.ingest.fsi import load_fsi

# Fixture uses real FSI column names as found in the annual Excel files
_VALID_CSV = (
    "Country,Year,Rank,Total,"
    "C1: Security Apparatus,C2: Factionalized Elites,C3: Group Grievance,"
    "E1: Economy,E2: Economic Inequality,E3: Human Flight and Brain Drain,"
    "P1: State Legitimacy,P2: Public Services,P3: Human Rights,"
    "S1: Demographic Pressures,S2: Refugees and IDPs,X1: External Intervention\n"
    "Afghanistan,2015,7th,105.3,9.2,8.5,8.7,8.2,7.9,8.1,9.0,9.1,8.8,9.3,9.2,9.1\n"
    "France,2015,161st,26.1,2.1,2.3,3.2,2.8,4.1,2.9,2.0,1.8,1.9,1.5,1.3,1.9\n"
    "Afghanistan,2016,7th,103.8,9.0,8.3,8.5,8.0,7.7,7.9,8.8,8.9,8.6,9.1,9.0,9.0\n"
)


def test_load_fsi_returns_dataframe():
    """The consumer gets a DataFrame back."""
    result = load_fsi(io.StringIO(_VALID_CSV))
    assert isinstance(result, pd.DataFrame)


def test_load_fsi_has_iso3_and_year_columns():
    """Output must use iso3 not country name; year must be present."""
    result = load_fsi(io.StringIO(_VALID_CSV))
    assert "iso3" in result.columns
    assert "year" in result.columns
    assert "Country" not in result.columns


def test_load_fsi_drops_rank_column():
    """Rank is a derived label, not a variable — must not appear in output."""
    result = load_fsi(io.StringIO(_VALID_CSV))
    assert "Rank" not in result.columns


def test_load_fsi_is_one_row_per_country_year():
    """FSI is already country-year format — no duplicates allowed."""
    result = load_fsi(io.StringIO(_VALID_CSV))
    assert result.duplicated(subset=["iso3", "year"]).sum() == 0


def test_load_fsi_columns_use_canonical_names():
    """Composite score and sub-indicators must use snake_case canonical names."""
    result = load_fsi(io.StringIO(_VALID_CSV))
    assert "fsi_score" in result.columns
    assert "fsi_p1_legitimacy" in result.columns
    assert "fsi_c3_group_grievance" in result.columns
    assert "fsi_s2_refugees" in result.columns
    assert "fsi_e3_human_flight" in result.columns
    assert "Total" not in result.columns
    assert "P1: State Legitimacy" not in result.columns


def test_load_fsi_scores_are_preserved_correctly():
    """Score values must survive the rename without modification."""
    result = load_fsi(io.StringIO(_VALID_CSV))
    afg = result[(result["iso3"] == "AFG") & (result["year"] == 2015)].iloc[0]
    assert abs(afg["fsi_score"] - 105.3) < 1e-6
    assert abs(afg["fsi_p1_legitimacy"] - 9.0) < 1e-6


def test_load_fsi_raises_on_missing_required_columns():
    """A CSV without Country, Year, or Total must fail with a clear error."""
    bad_csv = "Nation,Yr,Score\nAfghanistan,2015,105.3\n"
    with pytest.raises(ValueError, match="Missing required columns"):
        load_fsi(io.StringIO(bad_csv))


def test_load_fsi_year_is_integer_even_when_source_stores_dates():
    """Older FSI Excel files store Year as a date object — must be extracted as int."""
    csv = (
        "Country,Year,Total\n"
        "Afghanistan,2006-01-01,105.3\n"  # date stored as datetime string
        "France,2015,26.1\n"  # year stored as integer
    )
    result = load_fsi(io.StringIO(csv))
    assert pd.api.types.is_integer_dtype(result["year"])
    assert result.iloc[0]["year"] == 2006
    assert result.iloc[1]["year"] == 2015


def test_load_fsi_drops_extra_derived_columns():
    """Non-indicator columns like 'Change from Previous Year' must not appear in output."""
    csv = (
        "Country,Year,Rank,Total,Change from Previous Year,P1: State Legitimacy\n"
        "Afghanistan,2015,7th,105.3,+1.2,9.0\n"
    )
    result = load_fsi(io.StringIO(csv))
    assert "Change from Previous Year" not in result.columns


def test_load_fsi_strips_trailing_whitespace_from_country_names():
    """FSI files contain trailing spaces in country names — must be stripped."""
    csv = (
        "Country,Year,Rank,Total,P1: State Legitimacy\n"
        "Afghanistan ,2015,7th,105.3,9.0\n"  # trailing space
        "France ,2015,161st,26.1,2.1\n"
    )
    result = load_fsi(io.StringIO(csv))
    assert "AFG" in result.iso3.values
    assert "FRA" in result.iso3.values


def test_load_fsi_handles_fsi_specific_country_names():
    """FSI uses non-standard names for some countries — must map to valid ISO3."""
    csv = (
        "Country,Year,Rank,Total\n"
        "Congo Democratic Republic,2015,4th,110.0\n"
        "Swaziland,2015,20th,80.0\n"
        "Cape Verde,2015,100th,44.0\n"
    )
    result = load_fsi(io.StringIO(csv))
    assert "COD" in result.iso3.values
    assert "SWZ" in result.iso3.values
    assert "CPV" in result.iso3.values


def test_load_fsi_skips_compound_political_entries():
    """Entries like 'Israel and West Bank' are not single countries — must be dropped."""
    csv = (
        "Country,Year,Rank,Total\nAfghanistan,2015,7th,105.3\nIsrael and West Bank,2015,50th,60.0\n"
    )
    result = load_fsi(io.StringIO(csv))
    assert len(result) == 1
    assert "AFG" in result.iso3.values


def test_load_fsi_raises_if_any_country_cannot_be_mapped():
    """Unmapped country names must raise loudly."""
    csv = (
        "Country,Year,Rank,Total,P1: State Legitimacy\n"
        "Afghanistan,2015,7th,105.3,9.0\n"
        "NotARealCountry,2015,99th,50.0,5.0\n"
    )
    with pytest.raises(ValueError, match="unmapped"):
        load_fsi(io.StringIO(csv))

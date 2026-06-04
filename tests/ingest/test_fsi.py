"""FSI (Fragile States Index) ingest — strict TDD, one test at a time.

Fund for Peace publishes one row per country per year with a composite
FSI score and 12 sub-indicator scores. Already country-year format —
no pivot or aggregation needed, just map to iso3 and rename columns.
"""

import io

import pandas as pd
import pytest

from src.ingest.fsi import load_fsi

_VALID_CSV = (
    "Country,Year,Total,P1: State Legitimacy,C3: Human Rights\n"
    "Afghanistan,2015,105.3,9.2,8.7\n"
    "France,2015,26.1,2.1,1.8\n"
    "Afghanistan,2016,103.8,9.0,8.5\n"
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


def test_load_fsi_is_one_row_per_country_year():
    """FSI is already country-year format — no duplicates allowed."""
    result = load_fsi(io.StringIO(_VALID_CSV))
    assert result.duplicated(subset=["iso3", "year"]).sum() == 0


def test_load_fsi_columns_use_canonical_names():
    """Composite score and sub-indicators must use snake_case canonical names."""
    result = load_fsi(io.StringIO(_VALID_CSV))
    assert "fsi_score" in result.columns
    assert "fsi_p1_legitimacy" in result.columns
    assert "Total" not in result.columns
    assert "P1: State Legitimacy" not in result.columns


def test_load_fsi_scores_are_preserved_correctly():
    """Score values must survive the rename without modification."""
    result = load_fsi(io.StringIO(_VALID_CSV))
    afg = result[(result["iso3"] == "AFG") & (result["year"] == 2015)].iloc[0]
    assert abs(afg["fsi_score"] - 105.3) < 1e-6
    assert abs(afg["fsi_p1_legitimacy"] - 9.2) < 1e-6


def test_load_fsi_raises_on_missing_required_columns():
    """A CSV without Country, Year, or Total must fail with a clear error."""
    bad_csv = "Nation,Yr,Score\nAfghanistan,2015,105.3\n"
    with pytest.raises(ValueError, match="Missing required columns"):
        load_fsi(io.StringIO(bad_csv))


def test_load_fsi_raises_if_any_country_cannot_be_mapped():
    """Unmapped country names must raise loudly."""
    csv = (
        "Country,Year,Total,P1: State Legitimacy,C3: Human Rights\n"
        "Afghanistan,2015,105.3,9.2,8.7\n"
        "NotARealCountry,2015,50.0,5.0,5.0\n"
    )
    with pytest.raises(ValueError, match="unmapped"):
        load_fsi(io.StringIO(csv))

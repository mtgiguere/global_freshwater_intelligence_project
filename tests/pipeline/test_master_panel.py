"""Master Panel assembly — strict TDD, one test at a time.

Joins all Phase 1 ingest outputs on [iso3, year] into a single
validated country-year panel. Lives in src/pipeline/.
"""

import pandas as pd
import pytest

from src.pipeline.master_panel import build_master_panel


def _df(iso3s, years, **cols):
    """Build a minimal panel-format DataFrame for testing."""
    data = {"iso3": iso3s, "year": years}
    data.update(cols)
    return pd.DataFrame(data)


def test_build_master_panel_returns_dataframe():
    """The consumer gets a DataFrame back."""
    a = _df(["AFG"], [2010], val_a=[1.0])
    result = build_master_panel([a])
    assert isinstance(result, pd.DataFrame)


def test_build_master_panel_has_iso3_and_year_columns():
    """Output must have iso3 and year as panel identifiers."""
    a = _df(["AFG"], [2010], val_a=[1.0])
    result = build_master_panel([a])
    assert "iso3" in result.columns
    assert "year" in result.columns


def test_build_master_panel_is_one_row_per_country_year():
    """Output must have exactly one row per (iso3, year)."""
    a = _df(["AFG", "FRA"], [2010, 2010], val_a=[1.0, 2.0])
    b = _df(["AFG", "FRA"], [2010, 2010], val_b=[3.0, 4.0])
    result = build_master_panel([a, b])
    assert result.duplicated(subset=["iso3", "year"]).sum() == 0


def test_build_master_panel_includes_columns_from_all_sources():
    """All columns from every source must appear in the output."""
    a = _df(["AFG"], [2010], val_a=[1.0])
    b = _df(["AFG"], [2010], val_b=[2.0])
    c = _df(["AFG"], [2010], val_c=[3.0])
    result = build_master_panel([a, b, c])
    assert "val_a" in result.columns
    assert "val_b" in result.columns
    assert "val_c" in result.columns


def test_build_master_panel_missing_country_year_in_source_gets_nan():
    """A country-year present in one source but absent in another gets NaN, not dropped.

    AFG 2010 exists in both; FRA 2010 only in source b.
    The panel must contain FRA 2010 with NaN for val_a.
    """
    a = _df(["AFG"], [2010], val_a=[1.0])
    b = _df(["AFG", "FRA"], [2010, 2010], val_b=[2.0, 3.0])
    result = build_master_panel([a, b])
    fra = result[(result["iso3"] == "FRA") & (result["year"] == 2010)]
    assert len(fra) == 1
    assert pd.isna(fra.iloc[0]["val_a"])
    assert fra.iloc[0]["val_b"] == 3.0


def test_build_master_panel_raises_if_source_missing_iso3_or_year():
    """Any source without iso3 or year columns must raise with a clear error."""
    good = _df(["AFG"], [2010], val_a=[1.0])
    bad = pd.DataFrame({"country": ["AFG"], "yr": [2010], "val_b": [2.0]})
    with pytest.raises(ValueError, match="iso3"):
        build_master_panel([good, bad])


def test_build_master_panel_raises_if_given_empty_list():
    """An empty source list must raise — there is nothing to build."""
    with pytest.raises(ValueError, match="at least one"):
        build_master_panel([])

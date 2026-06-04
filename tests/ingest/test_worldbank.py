"""World Bank ingest — strict TDD, one test at a time."""

import io

import pandas as pd
import pytest

from src.ingest.worldbank import load_worldbank

# World Bank bulk CSV has a wide format: one row per country-indicator,
# one column per year. The header block looks like:
#
# "Country Name","Country Code","Indicator Name","Indicator Code","1960","1961",...,"2023",
#
# followed by data rows like:
# "Afghanistan","AFG","GDP per capita (constant 2015 US$)","NY.GDP.PCAP.KD","","",..."500.1",

_VALID_CSV = (
    '"Country Name","Country Code","Indicator Name","Indicator Code","2010","2015","2020"\n'
    '"Afghanistan","AFG","GDP per capita (constant 2015 US$)","NY.GDP.PCAP.KD","500.1","530.2","480.3"\n'  # noqa: E501
    '"Afghanistan","AFG","Human development index (HDI)","HDI","0.45","0.48","0.51"\n'
    '"France","FRA","GDP per capita (constant 2015 US$)","NY.GDP.PCAP.KD","38000.1","39500.4","37200.8"\n'  # noqa: E501
    '"France","FRA","Human development index (HDI)","HDI","0.89","0.90","0.90"\n'
)


def test_load_worldbank_returns_dataframe():
    """The consumer gets a DataFrame back."""
    result = load_worldbank(io.StringIO(_VALID_CSV))
    assert isinstance(result, pd.DataFrame)


def test_load_worldbank_identifies_countries_by_iso3():
    """Countries must be identified by an iso3 column."""
    result = load_worldbank(io.StringIO(_VALID_CSV))
    assert "iso3" in result.columns
    assert "Country Name" not in result.columns
    assert "Country Code" not in result.columns


def test_load_worldbank_is_one_row_per_country_year():
    """Output must be wide format — one row per (iso3, year), indicators as columns."""
    result = load_worldbank(io.StringIO(_VALID_CSV))
    assert result.duplicated(subset=["iso3", "year"]).sum() == 0
    assert "year" in result.columns


def test_load_worldbank_indicators_use_canonical_names():
    """Indicator columns must use snake_case canonical names, not raw World Bank strings."""
    result = load_worldbank(io.StringIO(_VALID_CSV))
    assert "gdp_pc_ppp" in result.columns
    assert "hdi" in result.columns
    assert "GDP per capita (constant 2015 US$)" not in result.columns


def test_load_worldbank_raises_on_missing_required_columns():
    """A CSV missing Country Code or Indicator Code must fail with a clear error."""
    bad_csv = "Country,Indicator,Value\nAfghanistan,GDP,500\n"
    with pytest.raises(ValueError, match="Missing required columns"):
        load_worldbank(io.StringIO(bad_csv))


def test_load_worldbank_raises_if_no_known_indicators_present():
    """A CSV with no recognised indicator codes must fail, not return an empty panel."""
    csv = (
        '"Country Name","Country Code","Indicator Name","Indicator Code","2010"\n'
        '"Afghanistan","AFG","Some unknown metric","XX.UNKNOWN","1.5"\n'
    )
    with pytest.raises(ValueError, match="No recognised indicators"):
        load_worldbank(io.StringIO(csv))


def test_load_worldbank_drops_rows_where_all_indicator_values_are_null():
    """Country-year rows with no non-null indicator values must be dropped silently."""
    csv = (
        '"Country Name","Country Code","Indicator Name","Indicator Code","2010","2015"\n'
        '"Afghanistan","AFG","GDP per capita (constant 2015 US$)","NY.GDP.PCAP.KD","500.1",""\n'
        '"France","FRA","GDP per capita (constant 2015 US$)","NY.GDP.PCAP.KD","38000.1","39500.4"\n'
    )
    result = load_worldbank(io.StringIO(csv))
    afg_2015 = result[(result["iso3"] == "AFG") & (result["year"] == 2015)]
    assert len(afg_2015) == 0

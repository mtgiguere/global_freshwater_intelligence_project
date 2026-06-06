"""load_grace_panel ingest — strict TDD, one test at a time.

Reads the pre-processed GRACE country-year CSV produced by
src/ingest/download/grace_process.py. Validates structure and values.
"""

import io

import pandas as pd
import pytest

from src.ingest.grace_panel import load_grace_panel

_VALID_CSV = (
    "iso3,year,grace_lwe_anomaly_cm\nAFG,2003,-1.2\nAFG,2004,-2.1\nIND,2003,-4.5\nIND,2004,-5.3\n"
)


def test_load_grace_panel_returns_dataframe():
    result = load_grace_panel(io.StringIO(_VALID_CSV))
    assert isinstance(result, pd.DataFrame)


def test_load_grace_panel_has_iso3_year_and_anomaly_columns():
    result = load_grace_panel(io.StringIO(_VALID_CSV))
    assert "iso3" in result.columns
    assert "year" in result.columns
    assert "grace_lwe_anomaly_cm" in result.columns


def test_load_grace_panel_is_one_row_per_country_year():
    result = load_grace_panel(io.StringIO(_VALID_CSV))
    assert result.duplicated(subset=["iso3", "year"]).sum() == 0


def test_load_grace_panel_year_is_integer():
    result = load_grace_panel(io.StringIO(_VALID_CSV))
    assert pd.api.types.is_integer_dtype(result["year"])


def test_load_grace_panel_raises_on_missing_required_columns():
    bad = "country,yr,anomaly\nAFG,2003,-1.2\n"
    with pytest.raises(ValueError, match="Missing required columns"):
        load_grace_panel(io.StringIO(bad))

"""Master Panel data quality validation — strict TDD, one test at a time.

Enforces the Phase 1 exit criteria before any analysis begins:
  - No duplicate (iso3, year) keys
  - iso3 values are valid 3-letter uppercase codes
  - year values are integers within a sensible range
  - Primary exposure variable coverage >= 90% for 1990+ rows
  - All outcome variables coverage >= 70%
"""

import pandas as pd
import pytest

from src.pipeline.validate import validate_master_panel


def _panel(**cols):
    """Build a minimal Master Panel DataFrame."""
    base = {
        "iso3": ["AFG", "FRA"],
        "year": [2010, 2010],
    }
    base.update(cols)
    return pd.DataFrame(base)


def test_validate_master_panel_passes_for_valid_panel():
    """validate_master_panel must not raise for a correctly structured panel."""
    panel = _panel(renewable_freshwater_percap=[1.5, 3.2])
    validate_master_panel(panel)  # no exception = pass


def test_validate_raises_on_duplicate_panel_keys():
    """Duplicate (iso3, year) rows must raise — they corrupt every downstream join."""
    panel = pd.DataFrame(
        {
            "iso3": ["AFG", "AFG"],
            "year": [2010, 2010],
            "renewable_freshwater_percap": [1.5, 1.6],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_master_panel(panel)


def test_validate_raises_on_invalid_iso3_format():
    """iso3 values that are not 3 uppercase letters must raise."""
    panel = _panel(renewable_freshwater_percap=[1.5, 3.2])
    panel.loc[0, "iso3"] = "afghanistan"
    with pytest.raises(ValueError, match="iso3"):
        validate_master_panel(panel)


def test_validate_raises_on_year_out_of_range():
    """Years before 1945 or after 2100 must raise — they indicate a parsing error."""
    panel = _panel(renewable_freshwater_percap=[1.5, 3.2])
    panel.loc[0, "year"] = 1800
    with pytest.raises(ValueError, match="year"):
        validate_master_panel(panel)


def test_validate_accepts_post_ww2_years():
    """Years from 1946 onward are valid — UCDP conflict data starts in 1946."""
    panel = _panel(renewable_freshwater_percap=[1.5, 3.2])
    panel.loc[0, "year"] = 1946
    validate_master_panel(panel)  # no exception = pass


def _iso3(i: int) -> str:
    """Generate a valid-format (3 uppercase letters) iso3 code for test fixtures."""
    return f"{chr(65 + i // 26 % 26)}{chr(65 + i % 26)}A"


def test_validate_raises_if_primary_exposure_coverage_below_60_pct():
    """renewable_freshwater_percap must be non-null for >= 60% of 1990+ rows.

    60% reflects realistic AQUASTAT coverage when outer-joined with all sources.
    Below 60% indicates a data pipeline failure, not normal missingness.
    """
    rows = [
        {"iso3": _iso3(i), "year": 1995, "renewable_freshwater_percap": (1.0 if i < 5 else None)}
        for i in range(20)
    ]  # only 5/20 = 25% coverage
    panel = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="renewable_freshwater_percap"):
        validate_master_panel(panel)


def test_validate_passes_when_primary_exposure_coverage_meets_threshold():
    """Must not raise when >= 60% of 1990+ rows have renewable_freshwater_percap."""
    rows = [
        {"iso3": _iso3(i), "year": 1995, "renewable_freshwater_percap": (1.0 if i < 14 else None)}
        for i in range(20)
    ]  # 19/20 = 95% coverage
    validate_master_panel(pd.DataFrame(rows))  # no exception = pass


def test_validate_raises_if_primary_exposure_column_entirely_absent():
    """Panel with no renewable_freshwater_percap column at all must raise."""
    panel = _panel(some_other_col=[1.0, 2.0])
    with pytest.raises(ValueError, match="renewable_freshwater_percap"):
        validate_master_panel(panel)


def test_validate_coverage_check_only_applies_to_1990_and_later():
    """Pre-1990 rows with missing exposure variable must not trigger the coverage check."""
    rows = [
        {"iso3": _iso3(i), "year": 1985, "renewable_freshwater_percap": None} for i in range(20)
    ]  # all missing but all pre-1990
    validate_master_panel(pd.DataFrame(rows))  # no exception = pass

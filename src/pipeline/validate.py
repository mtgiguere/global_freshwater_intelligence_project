"""Structural validation for the GFIP Master Panel.

This module runs quality checks on the assembled Master Panel *before* it is written
to disk. Think of this as a test suite for the data itself, not for the code.

The checks here enforce the Phase 1 exit criteria: the panel must be structurally
sound — correct identifiers, no duplicate rows, sensible year range, and adequate
coverage of the primary freshwater exposure variable — before any downstream phase
(EDA, hypothesis testing, ML models) can consume it.

Why validate before saving rather than trusting the ingest modules?
Each ingest module validates its own slice of data, but the merge step in
master_panel.py can still produce structural problems (e.g. a poorly formatted
helper file could introduce rows with numeric country codes instead of ISO3 strings).
Running a final structural check here gives a single, guaranteed clean input for all
downstream work and makes failures loud and early rather than silent and confusing.
"""

import pandas as pd

_PRIMARY_EXPOSURE = "renewable_freshwater_percap"
_COVERAGE_THRESHOLD = 0.60  # AQUASTAT covers ~186/260 countries; 60% is realistic floor
_COVERAGE_FROM_YEAR = 1990
_YEAR_MIN = 1945  # post-WWII floor; UCDP conflict data starts 1946
_YEAR_MAX = 2100


def validate_master_panel(panel: pd.DataFrame) -> None:
    """Enforce Phase 1 exit criteria on the assembled Master Panel.

    Runs four structural checks in sequence. The first failure raises immediately;
    subsequent checks are not attempted after a failure. Fix errors one at a time
    by re-running the pipeline.

    Args:
        panel: The assembled Master Panel DataFrame produced by build_master_panel().
            Must contain at least the columns ``iso3``, ``year``, and
            ``renewable_freshwater_percap``.

    Returns:
        None. This function is called for its side-effects only — it either passes
        silently or raises.

    Raises:
        ValueError: If any of the following conditions are detected:
            - Duplicate (iso3, year) pairs.
            - iso3 values that are not exactly 3 uppercase ASCII letters.
            - year values outside the range [_YEAR_MIN, _YEAR_MAX].
            - The ``renewable_freshwater_percap`` column is absent.
            - Coverage of ``renewable_freshwater_percap`` for post-1990 rows is
              below the 60% threshold.
    """

    # --- Check 1: No duplicate panel keys ---
    # Downstream consequence if skipped: panel regressions in Phase 3 and ML feature
    # engineering in Phase 4 both assume one row per (country, year). Duplicate rows
    # cause double-counting of observations, which inflates sample sizes, deflates
    # standard errors, and can produce falsely "significant" regression results.
    dupes = panel.duplicated(subset=["iso3", "year"]).sum()
    if dupes > 0:
        raise ValueError(f"Panel has {dupes} duplicate (iso3, year) row(s)")

    # --- Check 2: iso3 must be exactly 3 uppercase letters ---
    # Downstream consequence if skipped: every downstream merge, API lookup, and
    # dashboard query keys on iso3. A row with "Afghanistan" or "afg" instead of
    # "AFG" would silently fail every join and appear as missing data rather than
    # producing an error — extremely hard to diagnose months later.
    invalid_iso3 = panel[~panel["iso3"].str.match(r"^[A-Z]{3}$")]
    if not invalid_iso3.empty:
        raise ValueError(
            f"iso3 values must be 3 uppercase letters. "
            f"Invalid: {invalid_iso3['iso3'].unique().tolist()[:5]}"
        )

    # --- Check 3: year must be within sensible bounds ---
    # Downstream consequence if skipped: a year value of e.g. 20190 (typo) would
    # create a row far outside the panel range that no analysis code would pick up,
    # but which would corrupt rolling-window and lag features in Phase 4 ML models
    # because pandas time-based sorting would place it at the end of the series.
    out_of_range = panel[(panel["year"] < _YEAR_MIN) | (panel["year"] > _YEAR_MAX)]
    if not out_of_range.empty:
        raise ValueError(
            f"year values must be between {_YEAR_MIN} and {_YEAR_MAX}. "
            f"Found: {out_of_range['year'].unique().tolist()[:5]}"
        )

    # --- Check 4: Primary exposure column must be present ---
    # renewable_freshwater_percap is the central exposure variable in all seven
    # Phase 3 hypotheses and the primary input to the scarcity ML model. If it is
    # absent the entire project's causal story collapses. Fail loudly here rather
    # than producing NaN-filled regression tables that look plausible.
    if _PRIMARY_EXPOSURE not in panel.columns:
        raise ValueError("renewable_freshwater_percap column is missing from the panel")

    # --- Check 5: Primary exposure coverage must meet the 60% floor for 1990+ rows ---
    # AQUASTAT (the data source for this column) covers approximately 186 of the
    # 260 countries in the panel — roughly 72% globally, but sparser before 1990.
    # We only enforce the threshold for 1990-onwards rows because pre-1990 data is
    # legitimately sparse. If coverage falls below 60% it almost certainly means
    # the AQUASTAT ingest failed or the merge dropped rows unexpectedly.
    # Downstream consequence if skipped: Phase 3 regressions would run on a sample
    # far too small to support generalisation, and results would be unrepresentative.
    recent = panel[panel["year"] >= _COVERAGE_FROM_YEAR]
    if len(recent) > 0:
        coverage = recent[_PRIMARY_EXPOSURE].notna().mean()
        if coverage < _COVERAGE_THRESHOLD:
            raise ValueError(
                f"renewable_freshwater_percap coverage is {coverage:.1%} "
                f"for {_COVERAGE_FROM_YEAR}+ rows — minimum is {_COVERAGE_THRESHOLD:.0%}"
            )

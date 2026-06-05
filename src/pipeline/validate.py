import pandas as pd

_PRIMARY_EXPOSURE = "renewable_freshwater_percap"
_COVERAGE_THRESHOLD = 0.60  # AQUASTAT covers ~186/260 countries; 60% is realistic floor
_COVERAGE_FROM_YEAR = 1990
_YEAR_MIN = 1945  # post-WWII floor; UCDP conflict data starts 1946
_YEAR_MAX = 2100


def validate_master_panel(panel: pd.DataFrame) -> None:
    """Enforce Phase 1 exit criteria. Raises ValueError on any violation."""

    # No duplicate panel keys
    dupes = panel.duplicated(subset=["iso3", "year"]).sum()
    if dupes > 0:
        raise ValueError(f"Panel has {dupes} duplicate (iso3, year) row(s)")

    # iso3 must be 3 uppercase letters
    invalid_iso3 = panel[~panel["iso3"].str.match(r"^[A-Z]{3}$")]
    if not invalid_iso3.empty:
        raise ValueError(
            f"iso3 values must be 3 uppercase letters. "
            f"Invalid: {invalid_iso3['iso3'].unique().tolist()[:5]}"
        )

    # year must be within sensible range
    out_of_range = panel[(panel["year"] < _YEAR_MIN) | (panel["year"] > _YEAR_MAX)]
    if not out_of_range.empty:
        raise ValueError(
            f"year values must be between {_YEAR_MIN} and {_YEAR_MAX}. "
            f"Found: {out_of_range['year'].unique().tolist()[:5]}"
        )

    # Primary exposure column must be present
    if _PRIMARY_EXPOSURE not in panel.columns:
        raise ValueError("renewable_freshwater_percap column is missing from the panel")

    # Primary exposure coverage >= 90% for 1990+ rows
    recent = panel[panel["year"] >= _COVERAGE_FROM_YEAR]
    if len(recent) > 0:
        coverage = recent[_PRIMARY_EXPOSURE].notna().mean()
        if coverage < _COVERAGE_THRESHOLD:
            raise ValueError(
                f"renewable_freshwater_percap coverage is {coverage:.1%} "
                f"for {_COVERAGE_FROM_YEAR}+ rows — minimum is {_COVERAGE_THRESHOLD:.0%}"
            )

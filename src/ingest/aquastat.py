"""AQUASTAT freshwater and agricultural water-use ingest.

AQUASTAT is the Food and Agriculture Organization's (FAO) Global Information System
on Water and Agriculture. It is the world's primary repository of country-level data
on freshwater resources, water use, and irrigation infrastructure, with records going
back to the 1960s for some indicators.

Why AQUASTAT matters for GFIP
-------------------------------
AQUASTAT provides the core "freshwater availability" exposure variable used in every
Phase 3 hypothesis test (H1-H7). Without reliable, comparable freshwater data across
countries, we cannot establish whether scarcity precedes or follows the human outcomes
we are measuring.

Key variables loaded by this module
-------------------------------------
- renewable_freshwater_percap (m³ / person / year):
    The volume of internal river flow and groundwater recharge divided by population.
    This is the standard international measure of freshwater availability. The
    UN defines <1,700 m³/person/year as "water stress" and <1,000 as "water scarcity".
    Values below 500 m³/person/year are considered "absolute scarcity".

- total_withdrawal_km3 (km³ / year):
    The total volume of freshwater extracted from rivers, lakes, and aquifers for all
    purposes (agriculture, industry, domestic). High withdrawal relative to renewable
    supply signals unsustainable use — countries can withdraw more than they receive
    only by drawing down aquifers (the subject of H7).

- agri_withdrawal_pct (% of total withdrawal):
    The share of water used for agriculture. In many low-income countries, agriculture
    accounts for 70-90 % of all water use. This variable is central to H5 (water
    scarcity → agricultural economic dependence) and H6 (irrigation → HDI).

Data format
-----------
AQUASTAT publishes data in a "long" format with one row per (country, variable, year).
This module pivots it to a "wide" panel with one row per (iso3, year) and one column
per variable — the tidy format expected by the Master Panel assembler.

Source: https://www.fao.org/aquastat/en/
Download: src/ingest/download/download_aquastat.py
"""

import pandas as pd
import pycountry

# Mapping from AQUASTAT's verbose variable labels (as they appear in the "Variable Name"
# column of the raw CSV) to the project's canonical snake_case column names.
# Canonical names are documented in CLAUDE.md under "Phase 1 — Master Panel columns".
# This is declared at module level so tests can import and inspect it directly.
VARIABLE_NAMES: dict[str, str] = {
    "Renewable internal freshwater resources per capita": "renewable_freshwater_percap",
    "Total freshwater withdrawal": "total_withdrawal_km3",
    "Agricultural water withdrawal as % of total freshwater withdrawal": "agri_withdrawal_pct",
}

# These four columns must be present in the raw CSV for the module to function correctly.
# They are validated immediately after the file is read — any missing column triggers a
# fail-fast ValueError rather than allowing the pipeline to continue and silently produce
# wrong output downstream (e.g. nulls in a join key, or aggregations over the wrong column).
_REQUIRED_COLUMNS = ["Area", "Variable Name", "Year", "Value"]


def _to_iso3(name: str) -> str | None:
    """Convert a country name string to its ISO 3166-1 alpha-3 code.

    Uses the pycountry library, which follows the official ISO 3166 register.
    Returns None (rather than raising) so the caller can collect all unmapped
    names in one pass before raising a single, informative error.

    Args:
        name: The country name as it appears in the AQUASTAT "Area" column.
            AQUASTAT uses FAO country names, which sometimes differ from the
            ISO canonical name. Common mismatches include abbreviations (e.g.
            "Lao PDR" instead of "Lao People's Democratic Republic"), former
            country names (e.g. "Swaziland" instead of "Eswatini"), and
            UN-specific variants (e.g. "Bolivia (Plurinational State of)").
            When a new mismatch is discovered, add a manual override in the
            calling load function or add an alias to pycountry's fuzzy search.

    Returns:
        The three-letter ISO 3166-1 alpha-3 code (e.g. "USA", "CHN", "BRA"),
        or None if pycountry cannot match the name.

    Note:
        The load_aquastat function will raise a ValueError listing all
        unmapped country names if any are found — this ensures we never
        silently drop data for a country we failed to identify.
    """
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


def load_aquastat(path) -> pd.DataFrame:
    """Load, validate, and reshape an AQUASTAT CSV into a tidy country-year panel.

    AQUASTAT distributes its data as a long-format CSV where each row represents
    a single observation: one country, one variable, one year, one value. This
    function validates the file, maps country names to ISO3 codes, and pivots the
    data into the wide (one row per country-year) format expected by the Master Panel.

    Processing steps:
        1. Read CSV and fail-fast if any required columns are missing.
        2. Check that at least one of our three target variables is present; raise
           a descriptive error if none are found so the caller knows what to fix.
        3. Map country names ("Area" column) to ISO 3166-1 alpha-3 codes via pycountry.
           Fail-fast if any country name cannot be resolved — we never silently drop data.
        4. Pivot from long format (one row per country/variable/year) to wide format
           (one row per country/year, one column per variable).
        5. Rename columns from AQUASTAT's verbose labels to canonical snake_case names.

    Args:
        path: File path (str or pathlib.Path) to the AQUASTAT CSV, or any file-like
            object (e.g. io.StringIO) that pandas can read. Accepting file-like objects
            allows tests to pass in-memory CSV strings without touching the filesystem.

    Returns:
        A tidy pandas DataFrame with:
        - One row per (iso3, year) combination.
        - Columns: iso3 (str), year (int), and any subset of
          [renewable_freshwater_percap, total_withdrawal_km3, agri_withdrawal_pct]
          for which data is present in the source file. Columns for variables not
          present in the source will simply be absent from the output — the Master Panel
          assembler handles missing columns gracefully via outer joins.
        - No duplicate (iso3, year) rows (pivot_table uses aggfunc="first" to resolve
          the rare case where AQUASTAT reports multiple values for the same cell,
          e.g. revised estimates appearing alongside original estimates).

    Raises:
        ValueError: If any of _REQUIRED_COLUMNS are absent from the CSV.
            Message: "Missing required columns: [<list of missing column names>]"
        ValueError: If none of the recognised AQUASTAT variable names are present
            in the "Variable Name" column.
            Message: "No recognised variables found. Expected one of: [<list>]"
        ValueError: If any country name in the "Area" column cannot be mapped to an
            ISO3 code via pycountry.
            Message: "unmapped countries — no ISO3 code found: [<list of names>]"
    """
    df = pd.read_csv(path)

    # Validate required columns immediately — fail-fast rather than producing
    # confusing KeyError or NaN-filled output further down the pipeline.
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check that at least one of our three target variables appears in the file.
    # AQUASTAT exports can be filtered by variable at download time, so it is
    # entirely possible to receive a valid CSV that happens not to contain any
    # variable we care about. A clear error is better than a silently empty output.
    present = [v for v in VARIABLE_NAMES if v in df["Variable Name"].values]
    if not present:
        raise ValueError(f"No recognised variables found. Expected one of: {list(VARIABLE_NAMES)}")

    # Map the "Area" column (FAO country names) to ISO3 codes. We map first, then
    # check for failures in a single pass so the error message lists ALL unmapped
    # names at once — this is much friendlier than discovering them one-by-one.
    df["iso3"] = df["Area"].map(_to_iso3)
    unmapped = df.loc[df["iso3"].isna(), "Area"].unique().tolist()
    if unmapped:
        raise ValueError(f"unmapped countries — no ISO3 code found: {unmapped}")

    # Drop the original name column now that we have the canonical iso3 identifier.
    df = df.drop(columns=["Area"])
    df = df.rename(columns={"Year": "year"})

    # Pivot from long format to wide format:
    # Before:  iso3 | year | Variable Name                          | Value
    #          AFG  | 2010 | Renewable internal freshwater...       | 1503.0
    #          AFG  | 2010 | Total freshwater withdrawal            | 20.3
    # After:   iso3 | year | renewable_freshwater_percap | total_withdrawal_km3
    #          AFG  | 2010 | 1503.0                      | 20.3
    #
    # aggfunc="first" resolves any duplicate (iso3, year, variable) combinations
    # (e.g. where AQUASTAT provides both a preliminary and revised estimate for the
    # same cell). Using "first" is safe here because AQUASTAT's own quality flags
    # are not included in standard exports — all values are treated equally.
    df = df.pivot_table(
        index=["iso3", "year"],
        columns="Variable Name",
        values="Value",
        aggfunc="first",
    ).reset_index()

    # pivot_table sets the columns index name to "Variable Name"; clear it so the
    # DataFrame has clean, unnamed column axes before we apply the rename mapping.
    df.columns.name = None

    # Rename from AQUASTAT's verbose labels to the project's canonical snake_case names.
    df = df.rename(columns=VARIABLE_NAMES)
    return df

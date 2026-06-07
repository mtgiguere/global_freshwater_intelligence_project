"""Fragile States Index (FSI) ingest.

The Fragile States Index is published annually by the Fund for Peace, an independent
non-governmental organisation. The FSI assesses the vulnerability of states to
collapse or conflict by scoring 178 countries on 12 indicators that together capture
political, economic, social, and security dimensions of state fragility.

Column names verified against Fund for Peace annual Excel files (2006-2023).

Understanding the FSI score
-----------------------------
The total FSI score ranges from 0 (least fragile — most stable) to 120 (most fragile).
It is the sum of 12 sub-indicator scores, each ranging from 0 to 10. A score above
90 is generally considered "high alert" (extreme fragility). A score below 30 indicates
a very stable, high-functioning state.

The 12 sub-indicators belong to three groups:
    Cohesion indicators (C) — measure the capacity of the state to function:
        C1: Security Apparatus — whether the state monopoly on force is intact
        C2: Factionalized Elites — political fragmentation and elite competition
        C3: Group Grievance — communal tensions, historical injustices

    Economic indicators (E) — measure economic health and equity:
        E1: Economy — GDP trends, inflation, debt, unemployment
        E2: Economic Inequality — wealth and income disparities across groups
        E3: Human Flight and Brain Drain — emigration of skilled workers

    Political/Social indicators (P and S and X):
        P1: State Legitimacy — representation, corruption, democratic accountability
        P2: Public Services — infrastructure, health, education, sanitation
        P3: Human Rights — respect for civil and political rights
        S1: Demographic Pressures — population growth, food security, disease burden
        S2: Refugees and IDPs — displacement driven by or causing state stress
        X1: External Intervention — economic, political, or military interference

Why FSI matters for GFIP
--------------------------
FSI is the primary "state fragility" outcome variable in Phase 3 H2:
    H2: Countries with lower freshwater availability show higher state fragility scores.
    Expected sign: negative (more water → lower FSI → more stability).

FSI is also a feature in the Phase 4 instability model (instability.py), where
P1 (State Legitimacy) and the total FSI score are used as predictors of future
conflict onset and sudden regime deterioration.

Data limitations
-----------------
The FSI series begins in 2006. This limits the sample size for H2 relative to
hypotheses that can use AQUASTAT data from the 1960s onward — we have at most
~18 years of overlap rather than ~60. This is acknowledged in the H2 analysis and
in the "limitations" section of the GFIP final report.

Additionally, FSI is an index constructed from secondary sources and editorial
judgement, not raw administrative data — the Fund for Peace's methodology has
evolved over time. Researchers should be cautious about interpreting small
year-to-year changes as meaningful signals.

Source: https://fragilestatesindex.org/
Download: src/ingest/download/download_fsi.py
"""

import pandas as pd
import pycountry

# Mapping from FSI's column headers (as they appear in the annual Excel/CSV files)
# to the project's canonical snake_case names.
# The FSI uses a consistent naming convention across years (C1-C3, E1-E3, P1-P3,
# S1-S2, X1), which makes this mapping stable. The "Total" column is the overall
# FSI score — the sum of all 12 sub-indicator scores.
# Canonical names are documented in CLAUDE.md under "Phase 1 — Master Panel columns".
COLUMN_NAMES: dict[str, str] = {
    "Total": "fsi_score",
    "C1: Security Apparatus": "fsi_c1_security",
    "C2: Factionalized Elites": "fsi_c2_factions",
    "C3: Group Grievance": "fsi_c3_group_grievance",
    "E1: Economy": "fsi_e1_economy",
    "E2: Economic Inequality": "fsi_e2_inequality",
    "E3: Human Flight and Brain Drain": "fsi_e3_human_flight",
    "P1: State Legitimacy": "fsi_p1_legitimacy",
    "P2: Public Services": "fsi_p2_public_services",
    "P3: Human Rights": "fsi_p3_human_rights",
    "S1: Demographic Pressures": "fsi_s1_demographics",
    "S2: Refugees and IDPs": "fsi_s2_refugees",
    "X1: External Intervention": "fsi_x1_intervention",
}

# At minimum, we need a country identifier, a year, and the total fragility score.
# Sub-indicator columns are optional -- earlier FSI releases (2006-2008) occasionally
# used slightly different column headings. The pivot/rename step handles missing
# sub-indicator columns gracefully. Validated immediately after file read — any
# missing required column triggers a fail-fast error.
_REQUIRED_COLUMNS = ["Country", "Year", "Total"]

# FSI uses non-standard country names that do not match pycountry's ISO register.
# Rather than relying on fuzzy matching (which can silently produce wrong mappings),
# we maintain an explicit override table. These are the names that appear in the
# Fund for Peace raw files and whose correct ISO3 code we have verified manually.
# When a new mismatch is discovered (e.g. after a FSI methodology revision), add
# the override here rather than adjusting the data — this keeps the data immutable
# and the logic explicit.
_FSI_NAME_MAP: dict[str, str] = {
    "Cape Verde": "CPV",
    "Congo Democratic Republic": "COD",
    "Congo Republic": "COG",
    "Cote d'Ivoire": "CIV",
    "Guinea Bissau": "GNB",
    "Macedonia": "MKD",
    "Micronesia": "FSM",
    "Palestine": "PSE",
    "Russia": "RUS",
    "Swaziland": "SWZ",
    "Turkey": "TUR",
}

# Some FSI rows represent compound political entities that span multiple sovereign states
# or contested territories that lack a single ISO3 code. These rows cannot be meaningfully
# joined to a country-level panel, so we drop them silently before the unmapped-country
# check — otherwise they would trigger a spurious error.
_COMPOUND_ENTRIES = {"Israel and West Bank"}


def _to_iso3(name: str) -> str | None:
    """Convert an FSI country name to its ISO 3166-1 alpha-3 code.

    Checks the manual override table (_FSI_NAME_MAP) first, then falls back to
    pycountry's lookup. Returns None if neither method succeeds, so the caller
    can collect all unmapped names in a single pass before raising an error.

    The two-step lookup is necessary because FSI uses a mix of:
    - Informal short names: "Russia" (ISO: "Russian Federation"), "Turkey" (ISO: "Türkiye")
    - Outdated names: "Swaziland" (renamed to "Eswatini" in 2018), "Macedonia"
    - Variant spellings: "Cote d'Ivoire" (ISO uses "Côte d'Ivoire" with accent)
    - Unofficial entities: "Palestine" (no universally recognised ISO entry)

    Args:
        name: A country name as it appears in the FSI "Country" column, after
            stripping leading/trailing whitespace.

    Returns:
        The three-letter ISO 3166-1 alpha-3 code (e.g. "RUS", "TUR", "COD"),
        or None if the name cannot be resolved via either lookup method.
    """
    # Strip whitespace defensively — FSI files occasionally have trailing spaces.
    name = name.strip()

    # Check the manual override table first. These are known mismatches that
    # pycountry's lookup would either fail on or (worse) map incorrectly.
    if name in _FSI_NAME_MAP:
        return _FSI_NAME_MAP[name]

    # Fall back to pycountry for all standard country names. pycountry uses the
    # official ISO 3166 register and handles common aliases (e.g. "United States"
    # → "USA", "South Korea" → "KOR") via its fuzzy search internals.
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


def load_fsi(path) -> pd.DataFrame:
    """Load, validate, and clean a Fund for Peace FSI CSV into a tidy country-year panel.

    The FSI is distributed as an annual Excel file (one file per year) or as a
    multi-year CSV. This function handles both formats — in both cases the "Year"
    column identifies the observation year.

    Processing steps:
        1. Read CSV and fail-fast if required columns are missing.
        2. Strip whitespace from country names and drop compound entries that cannot
           be mapped to a single sovereign state.
        3. Map country names to ISO3 codes via _to_iso3 (manual overrides + pycountry).
           Fail-fast if any country name cannot be resolved.
        4. Drop administrative columns (Rank, Change from Previous Year) that are
           not analytical variables and would clutter the Master Panel.
        5. Rename columns from FSI labels to canonical snake_case names.
        6. Extract the integer year from the Year column — older FSI files store
           "Year" as a full datetime string (e.g. "2006-01-01"), not just "2006".

    Data coverage note: FSI data begins in 2006. Analyses combining FSI with
    AQUASTAT (which covers 1960-present) are limited to the 2006-present window.
    This limits statistical power in H2 relative to hypotheses that use longer series.

    Args:
        path: File path (str or pathlib.Path) to the FSI CSV, or any file-like
            object (e.g. io.StringIO) that pandas can read. Accepting file-like objects
            allows tests to pass in-memory CSV strings without touching the filesystem.

    Returns:
        A tidy pandas DataFrame with:
        - One row per (iso3, year) combination.
        - Columns: iso3 (str), year (int), fsi_score (float), and up to 12 sub-indicator
          columns (fsi_c1_security through fsi_x1_intervention) depending on which are
          present in the source file.
        - No duplicate (iso3, year) rows.
        - Index reset to 0-based integer index.

    Raises:
        ValueError: If any of _REQUIRED_COLUMNS are absent from the CSV.
            Message: "Missing required columns: [<list of missing column names>]"
        ValueError: If any country name (after dropping compound entries) cannot be
            mapped to an ISO3 code via either _FSI_NAME_MAP or pycountry.
            Message: "unmapped countries — no ISO3 code found: [<list of names>]"
    """
    df = pd.read_csv(path)

    # Validate required columns immediately — fail-fast rather than producing
    # confusing KeyError or NaN-filled output further down the pipeline.
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Strip whitespace from country names. FSI Excel exports sometimes pad cells
    # with trailing spaces, which would cause lookups to fail silently.
    df["_name"] = df["Country"].str.strip()

    # Drop compound political entries silently before the unmapped-country check.
    # Entries like "Israel and West Bank" represent contested or composite entities
    # that span multiple internationally recognised states. They have no single ISO3
    # code and would trigger a spurious unmapped-country error if not removed here.
    df = df[~df["_name"].isin(_COMPOUND_ENTRIES)].copy()

    # Map all remaining country names to ISO3 codes, collecting all failures at once.
    df["iso3"] = df["_name"].map(_to_iso3)

    # Fail-fast if any country names could not be resolved. Listing all failures in
    # one error message is much friendlier than discovering them one-by-one through
    # repeated runs. When this error occurs, check whether the name should be added
    # to _FSI_NAME_MAP (for FSI-specific variants) or if pycountry needs updating.
    unmapped = df.loc[df["iso3"].isna(), "_name"].unique().tolist()
    if unmapped:
        raise ValueError(f"unmapped countries — no ISO3 code found: {unmapped}")

    # Drop administrative and derived columns that are not analytical variables:
    # - "Country": replaced by "iso3"
    # - "_name": our temporary stripped working copy
    # - "Rank": an ordinal ranking derived from the total score; it adds no new
    #   information to the panel since the score itself is included
    # - "Change from Previous Year": a derived delta column; if researchers need
    #   year-over-year change they should compute it from the score series directly
    #   to ensure consistency with their analytical window
    to_drop = ["Country", "_name", "Rank", "Change from Previous Year"]
    drop = [c for c in to_drop if c in df.columns]
    df = df.drop(columns=drop)
    df = df.rename(columns={"Year": "year", **COLUMN_NAMES})

    # Extract integer year from whatever format FSI uses in this file.
    # Older FSI Excel files store the Year column as a full datetime string like
    # "2006-01-01 00:00:00" (a pandas Timestamp or datetime object), while newer
    # files store it as a plain integer (2006). Converting to string and slicing
    # the first four characters handles both formats safely.
    # Note: FSI data begins in 2006. The limited historical window (2006-present)
    # is a known constraint on analyses that join FSI with longer-running data series
    # like AQUASTAT (1960-present) or World Bank (1960-present).
    df["year"] = df["year"].apply(lambda v: int(str(v)[:4]))

    return df.reset_index(drop=True)

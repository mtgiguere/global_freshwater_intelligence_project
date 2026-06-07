"""World Bank Open Data ingest for development and welfare indicators.

The World Bank Open Data portal (data.worldbank.org) provides comparable
development statistics for nearly every country in the world, covering the period
from approximately 1960 to present. Data is collected from national statistical
offices, international surveys, and administrative records. The World Bank assigns
each indicator a standardised code (e.g. "NY.GDP.PCAP.KD") that is stable across
years and API versions.

Why World Bank data matters for GFIP
--------------------------------------
The World Bank indicators are the primary source for human outcome variables in the
Phase 3 hypothesis tests. They measure whether freshwater availability (the exposure)
actually translates into better or worse welfare for people.

Key indicators loaded by this module
--------------------------------------
- gdp_pc_ppp (NY.GDP.PCAP.KD — GDP per capita, PPP, constant 2015 USD):
    Purchasing Power Parity-adjusted GDP per capita, held in real 2015 dollars.
    PPP adjustment allows fair comparison across countries with different price levels
    (a dollar buys more in Ethiopia than in Switzerland). This is the economic outcome
    variable in H7 (groundwater depletion → lower subsequent GDP growth).

- gini (SI.POV.GINI — Gini index):
    Measures income inequality from 0 (perfect equality, everyone earns the same)
    to 100 (perfect inequality, one person earns everything). Higher Gini scores
    indicate more unequal societies. Note: World Bank Gini data has significant
    gaps — many country-years are missing because nationally representative income
    surveys are expensive and infrequent, particularly in low-income countries.

- hdi (HDI — Human Development Index):
    A composite index combining life expectancy, education, and income. Ranges from
    0 to 1. Published by UNDP but distributed via the World Bank API for convenience.
    Used in H6 (irrigation infrastructure → broader human development).

- agri_value_added_pct_gdp (NV.AGR.TOTL.ZS — agriculture, value added, % of GDP):
    The economic contribution of farming, forestry, and fishing as a share of total
    GDP. In water-stressed countries, agriculture often dominates the economy precisely
    because irrigated farming is one of the few viable livelihoods — this is the
    outcome variable in H5 (water scarcity → agrarian economic lock-in).

- safe_water_access_pct (SH.H2O.SMDW.ZS — people using safely managed drinking water):
    The percentage of the population with access to an improved water source that is
    accessible on premises, available when needed, and free from contamination. This is
    the primary outcome variable for H4 (freshwater availability → water access →
    child mortality and life expectancy outcomes).

Data format
-----------
The World Bank API and bulk downloads provide data in a "wide" format with one column
per year (e.g. columns "1960", "1961", … "2023"). This module melts those year columns
into a long format, then re-pivots to the tidy (iso3, year) panel the Master Panel
assembler expects.

Country identifiers: The World Bank already uses ISO 3166-1 alpha-3 codes in the
"Country Code" column — no name-to-ISO3 translation is needed here, unlike in the
AQUASTAT and FSI ingest modules.

Source: https://data.worldbank.org/
Download: src/ingest/download/download_worldbank.py
"""

import pandas as pd

# Mapping from World Bank indicator codes (stable, version-independent identifiers)
# to the project's canonical snake_case column names.
# The indicator codes are used as the join key when filtering the raw download,
# so they must exactly match what the World Bank API returns.
# Canonical names are documented in CLAUDE.md under "Phase 1 — Master Panel columns".
INDICATOR_NAMES: dict[str, str] = {
    "NY.GDP.PCAP.KD": "gdp_pc_ppp",
    "SI.POV.GINI": "gini",
    "HDI": "hdi",
    "NV.AGR.TOTL.ZS": "agri_value_added_pct_gdp",
    "SH.H2O.SMDW.ZS": "safe_water_access_pct",
}

# These two columns must be present for the module to function. Unlike AQUASTAT,
# World Bank exports already use ISO3 codes ("Country Code"), so we only need
# the code and the indicator identifier — we do not need the country name for
# our pipeline (it is dropped during processing).
# Validated immediately after file read — any missing column triggers a fail-fast error.
_REQUIRED_COLUMNS = ["Country Code", "Indicator Code"]


def load_worldbank(path) -> pd.DataFrame:
    """Load, validate, and reshape a World Bank CSV into a tidy country-year panel.

    World Bank bulk exports use a "wide" format where each year is a separate column
    (e.g. "1990", "1991", … "2023"). This function validates the file, filters to the
    indicators GFIP needs, melts the year columns into rows, and pivots to the wide
    (one row per country-year) format expected by the Master Panel assembler.

    Processing steps:
        1. Read CSV and fail-fast if required columns are missing.
        2. Check that at least one of our five target indicators is present; raise
           a descriptive error if none are found.
        3. Rename "Country Code" to "iso3" — World Bank already provides ISO3 codes,
           so no name-to-ISO3 translation is necessary.
        4. Identify year columns (any column whose name is an all-digit string like
           "1990") and melt them into a long format with one row per
           (iso3, indicator, year, value).
        5. Filter to only the indicators listed in INDICATOR_NAMES; map indicator codes
           to canonical column names.
        6. Pivot to wide format: one row per (iso3, year), one column per indicator.

    Args:
        path: File path (str or pathlib.Path) to the World Bank CSV, or any file-like
            object (e.g. io.StringIO) that pandas can read. Accepting file-like objects
            allows tests to pass in-memory CSV strings without touching the filesystem.

    Returns:
        A tidy pandas DataFrame with:
        - One row per (iso3, year) combination.
        - Columns: iso3 (str), year (int), and any subset of
          [gdp_pc_ppp, gini, hdi, agri_value_added_pct_gdp, safe_water_access_pct]
          for which data is present in the source file.
        - No duplicate (iso3, year) rows (pivot_table uses aggfunc="first").
        - Note: many country-year cells will be NaN — this is expected. The World Bank
          has genuine data gaps, particularly for Gini (infrequent surveys) and HDI
          (available only from UNDP for select years). NaNs propagate to the Master Panel
          where they are handled by the hypothesis tests.

    Raises:
        ValueError: If any of _REQUIRED_COLUMNS are absent from the CSV.
            Message: "Missing required columns: [<list of missing column names>]"
        ValueError: If none of the recognised World Bank indicator codes are present
            in the "Indicator Code" column.
            Message: "No recognised indicators found. Expected one of: [<list>]"
    """
    df = pd.read_csv(path)

    # Validate required columns immediately — fail-fast rather than producing
    # confusing KeyError or NaN-filled output further down the pipeline.
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check that at least one of our five target indicators is present. World Bank
    # bulk exports can contain hundreds of indicators; it's possible to download a
    # file that is perfectly valid but contains none of the indicators GFIP needs.
    # A clear error message is more useful than silently returning an empty DataFrame.
    present = [c for c in df["Indicator Code"].unique() if c in INDICATOR_NAMES]
    if not present:
        known = list(INDICATOR_NAMES)
        raise ValueError(f"No recognised indicators found. Expected one of: {known}")

    # World Bank already uses ISO3 codes — just rename the column to match our convention.
    df = df.rename(columns={"Country Code": "iso3"})

    # "Country Name" is a human-readable label included by the World Bank for convenience.
    # We drop it here because: (1) our pipeline uses iso3 as the join key throughout,
    # and (2) keeping it would cause it to be treated as an id_vars in the melt below,
    # creating a wider DataFrame than necessary.
    df = df.drop(columns=["Country Name"])

    # Identify year columns: World Bank exports use bare four-digit strings like "1990".
    # We detect them by checking whether the column name consists entirely of digits.
    # Non-year metadata columns (e.g. "Indicator Name", "Indicator Code") are not
    # all-digit strings and will correctly be excluded.
    year_cols = [c for c in df.columns if c.isdigit()]

    # Melt from wide (one column per year) to long (one row per country/indicator/year).
    # Before melt: iso3 | Indicator Code | 1990 | 1991 | ... | 2023
    # After melt:  iso3 | Indicator Code | year | value
    df = df.melt(
        id_vars=["iso3", "Indicator Name", "Indicator Code"],
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )

    # Convert year from string (e.g. "1990") to integer for consistent joins with
    # other modules that store year as int64.
    df["year"] = df["year"].astype("int64")

    # Filter to only our five indicators of interest, then map codes to canonical names.
    df = df[df["Indicator Code"].isin(INDICATOR_NAMES)].copy()
    df["Indicator Code"] = df["Indicator Code"].map(INDICATOR_NAMES)

    # Pivot from long back to wide: one row per (iso3, year), one column per indicator.
    # aggfunc="first" handles the rare case of duplicate (iso3, year, indicator) rows,
    # which can occur when the World Bank file contains both a preliminary and a revised
    # estimate for the same cell — we take whichever appears first in the file.
    df = df.pivot_table(
        index=["iso3", "year"],
        columns="Indicator Code",
        values="value",
        aggfunc="first",
    ).reset_index()

    # Clear the column axis name ("Indicator Code") that pivot_table sets automatically,
    # so the output DataFrame has a clean, unnamed column index.
    df.columns.name = None
    return df

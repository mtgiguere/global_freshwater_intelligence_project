"""Ingest module for the UN Department of Economic and Social Affairs (UNDESA) population data.

The UNDESA Population Division publishes the World Population Prospects (WPP) — the most
authoritative and widely used global dataset of population estimates and projections. The
WPP covers every country and territory in the world from 1950 to 2100 (the latter being
a projection), updated every two years. It is the source used by the UN system, the World
Bank, and virtually all international research that requires consistent cross-national
population figures.

Why population data is foundational for GFIP
---------------------------------------------
Population appears throughout the GFIP master panel in two distinct roles:

  1. *Denominator for per-capita calculations*: Many GFIP variables are only meaningful
     when normalised by population. Renewable freshwater availability in cubic kilometres
     is not comparable across countries of vastly different sizes; dividing by population
     to get cubic metres per person makes it comparable. The same logic applies to water
     withdrawal, GDP, homicide counts, and more. The ``population`` column from this module
     is the denominator used whenever a count variable is converted to a per-capita rate.

  2. *Control variable in regressions*: Population size, density, and urbanisation are
     important confounders in many GFIP hypotheses. For example, a country with a large
     urban population faces different water management challenges than a rural one, even
     at the same total freshwater availability level. Including population and urbanisation
     controls prevents these structural differences from being misattributed to freshwater
     stress in the regression models.

The urban/rural split
----------------------
The WPP also disaggregates population into urban and rural components:

  - ``population_urban``: People living in areas classified as urban by their national
    government. Urban areas typically have piped water infrastructure, centralised sewage,
    and regulated water utilities — though quality and reliability vary enormously.

  - ``population_rural``: People living outside urban areas. Rural populations are more
    likely to rely on boreholes, rivers, rainwater harvesting, or other decentralised
    water sources that are more vulnerable to drought, groundwater depletion, and
    contamination.

The urban/rural split matters for water access analysis because the risk profile is very
different between urban and rural water insecurity. A country moving from majority-rural
to majority-urban (as many developing countries are) may show improving average water
access while still having large numbers of people in peri-urban slums with poor water
quality — a nuance that aggregate figures obscure.

Units: thousands vs. absolute counts
--------------------------------------
UNDESA publishes population figures in *thousands* — a longstanding convention in the
demography literature. A WPP row showing "PopTotal = 67,000" means 67 million people,
not 67 thousand. This convention is confusing and a frequent source of data errors.

GFIP standardises to *absolute counts* (whole persons) throughout. This module multiplies
all three population columns by 1,000 immediately after loading. This means:
  - ``population``       → total persons (not thousands)
  - ``population_urban`` → urban persons (not thousands)
  - ``population_rural`` → rural persons (not thousands)

Any downstream code that reads the master panel should expect absolute counts for all
three population columns.

Coverage: global, 1950-2100. For GFIP we use historical estimates (1950-present);
projections are available but not currently loaded. Data quality is highest for countries
with strong civil registration systems; for countries with limited vital registration,
UNDESA uses statistical modelling from surveys and censuses.

Source: https://population.un.org/wpp/

Canonical GFIP column names produced:
  - ``iso3``             — ISO 3166-1 alpha-3 country code
  - ``year``             — calendar year (integer)
  - ``population``       — total population (absolute persons, converted from thousands)
  - ``population_urban`` — urban population (absolute persons, converted from thousands)
  - ``population_rural`` — rural population (absolute persons, converted from thousands)
"""

import pandas as pd
import pycountry

# The minimum set of columns that must be present in the raw UNDESA CSV.
# "Country"  — country name (UNDESA's own naming convention)
# "Year"     — calendar year
# "PopTotal" — total population in thousands (UNDESA convention; see _THOUSANDS below)
# PopUrban and PopRural are optional columns; their absence does not raise an error.
_REQUIRED_COLUMNS = ["Country", "Year", "PopTotal"]

# Mapping from raw UNDESA column names to GFIP canonical snake_case names.
# All downstream code — the master panel assembler, the API, the dashboard — uses these
# canonical names. Never reference the raw UNDESA names outside this module.
COLUMN_NAMES: dict[str, str] = {
    "Year": "year",
    "PopTotal": "population",
    "PopUrban": "population_urban",
    "PopRural": "population_rural",
}

# UN DESA publishes population in thousands — a longstanding demography convention.
# Multiplying by 1,000 converts to absolute person counts, which is the GFIP standard.
# This conversion is applied to all three population columns (total, urban, rural) after
# renaming, ensuring the master panel always contains whole-person counts.
_THOUSANDS = 1_000


def _to_iso3(name: str) -> str | None:
    """Convert a country name string to its ISO 3166-1 alpha-3 code.

    Uses the pycountry library to map country names to the ISO standard three-letter codes
    that GFIP uses as the universal country identifier. This ensures that UNDESA population
    data can be joined cleanly with every other dataset in the master panel on the "iso3"
    column.

    UNDESA uses its own country naming conventions for some territories and regions,
    particularly sub-national entities (e.g. Channel Islands, Réunion) that are tracked
    as separate population units but may not have standard ISO codes. Such cases will
    return None and surface as unmapped names during validation.

    Args:
        name: A country name string as it appears in the raw UNDESA data, e.g. "Germany".

    Returns:
        The ISO 3166-1 alpha-3 code (e.g. "DEU"), or None if pycountry cannot find a match.
    """
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


def load_undesa(path) -> pd.DataFrame:
    """Load a UNDESA World Population Prospects CSV export and standardise it for GFIP.

    The raw UNDESA file has one row per country-year with population figures in thousands
    (UNDESA's standard unit). This function validates the file, standardises country
    identifiers, renames columns to GFIP canonical names, and converts population figures
    from thousands to absolute person counts.

    Processing steps:
      1. Validate that required columns are present; raise immediately if not.
      2. Map each "Country" name to its ISO 3166-1 alpha-3 code using pycountry.
      3. Raise if any country name cannot be mapped.
      4. Drop the original "Country" column and rename remaining columns to GFIP canonical
         snake_case names using the COLUMN_NAMES mapping.
      5. Multiply population columns by 1,000 to convert from UNDESA's "thousands" unit
         to absolute person counts (the GFIP standard throughout the master panel).

    Args:
        path: Path to the UNDESA WPP CSV file, or a file-like object (e.g. io.StringIO).
              The file must contain at minimum the columns "Country", "Year", and
              "PopTotal". Population values are expected in thousands (UNDESA convention).

    Returns:
        A pandas DataFrame with one row per country-year, containing columns:
          - ``iso3`` (str): ISO 3166-1 alpha-3 country code.
          - ``year`` (int): Calendar year.
          - ``population`` (float): Total population in absolute persons (converted from
            UNDESA's thousands by multiplying by 1,000).
          - ``population_urban`` (float, if present): Urban population in absolute persons.
            Present only if the source file contained a "PopUrban" column.
          - ``population_rural`` (float, if present): Rural population in absolute persons.
            Present only if the source file contained a "PopRural" column.

    Raises:
        ValueError: If any required columns are absent from the input file, with message
            "Missing required columns: [...]".
        ValueError: If any country name in the "Country" column cannot be matched to an
            ISO 3166-1 alpha-3 code, with message
            "unmapped countries — no ISO3 code found: [...]".
    """
    df = pd.read_csv(path)

    # Validate required columns immediately. "PopTotal" being missing typically means
    # the wrong WPP variant was exported (e.g. fertility or mortality tables instead of
    # the population size table).
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Map country names to ISO 3166-1 alpha-3 codes. UNDESA includes some sub-national
    # and aggregate regions (e.g. "Less developed regions") alongside countries; these
    # will not map to ISO codes and must be excluded before ingest by filtering the
    # source file, or they will surface here as unmapped names.
    df["iso3"] = df["Country"].map(_to_iso3)

    # Fail immediately on any unmapped name. The population column is used as a
    # denominator throughout the master panel — missing or wrong population data for a
    # country would silently corrupt per-capita calculations for every variable.
    unmapped = df.loc[df["iso3"].isna(), "Country"].unique().tolist()
    if unmapped:
        raise ValueError(f"unmapped countries — no ISO3 code found: {unmapped}")

    # Drop the raw "Country" text column and apply the COLUMN_NAMES mapping to rename
    # UNDESA-specific column names to GFIP canonical snake_case names.
    df = df.drop(columns=["Country"]).rename(columns=COLUMN_NAMES)

    # Convert UNDESA's thousands-unit population figures to absolute person counts.
    #
    # UNDESA publishes population in thousands: a value of 83,200 means 83.2 million
    # people, not 83,200 people. This is a common source of off-by-1000 errors when
    # working with WPP data. GFIP standardises all counts to whole persons, so we
    # multiply by 1,000 here once and never again. All downstream code can assume
    # absolute counts in the master panel.
    #
    # We check "if col in df.columns" before multiplying to handle files that contain
    # only total population and not the urban/rural split (a valid subset export).
    for col in ["population", "population_urban", "population_rural"]:
        if col in df.columns:
            df[col] = df[col] * _THOUSANDS  # convert thousands → absolute persons

    return df

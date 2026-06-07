"""Ingest module for the UN Office on Drugs and Crime (UNODC) Homicide Statistics.

The UNODC publishes the Global Study on Homicide, which compiles intentional homicide
data from national police, justice, and public health records for approximately 200
countries. It is the most comprehensive cross-national dataset on lethal violence
available, updated roughly annually.

Why homicide data matters for freshwater research
--------------------------------------------------
Homicide rate — intentional killings per 100,000 population — is a widely used proxy
for organised violence and societal breakdown in contexts where formal armed conflict
(as defined by UCDP) has not occurred. A country can have a very high homicide rate due
to criminal networks, gang violence, or state repression without meeting the UCDP
threshold of 25 battle-deaths in armed conflict per year.

For GFIP, homicide rate serves two analytical roles:

  1. *Secondary stability indicator*: In regions where resource scarcity drives violence
     that falls below the UCDP armed-conflict threshold — interpersonal water disputes,
     criminal competition over water infrastructure, gang violence in water-stressed urban
     areas — homicide rate may pick up an effect that UCDP misses.

  2. *Outcome variable for future hypotheses*: H3 uses UCDP conflict as the outcome
     variable. Homicide provides a broader lens on violence that could anchor additional
     hypotheses, particularly for urban water stress and criminal violence.

Coverage caveats
----------------
Homicide data quality varies enormously across countries. High-income countries with
mature civil registration systems (e.g. Western Europe, Japan) have very reliable data.
Many low-income countries — particularly in sub-Saharan Africa and parts of South and
South-East Asia — have patchy or no systematic records, meaning their reported homicide
rates may be severe undercounts. UNODC uses statistical imputation for some gaps, but
researchers should treat low-income country data with extra caution.

For GFIP we use two variables:

  - ``homicide_rate``: intentional homicides per 100,000 population. This is the
    normalised measure comparable across countries of very different sizes.

  - ``homicide_count``: absolute number of intentional homicides. Useful as a
    denominator check and for countries where population estimates are uncertain.

Source: https://dataunodc.un.org/dp-intentional-homicide-victims

Canonical GFIP column names produced:
  - ``iso3``           — ISO 3166-1 alpha-3 country code
  - ``year``           — calendar year (integer)
  - ``homicide_rate``  — intentional homicides per 100,000 population
  - ``homicide_count`` — absolute intentional homicide count (may be missing for some
                         country-years if only rate is reported)
"""

import pandas as pd
import pycountry

# The minimum set of columns that must be present in the raw UNODC CSV.
# "Country" — the country name (UNODC's own naming convention)
# "Year"    — calendar year of the observation
# "Rate"    — intentional homicides per 100,000 population
# Note: "Count" (absolute homicides) is also renamed if present but is not required,
# since some UNODC exports include only the rate.
_REQUIRED_COLUMNS = ["Country", "Year", "Rate"]


def _to_iso3(name: str) -> str | None:
    """Convert a country name string to its ISO 3166-1 alpha-3 code.

    Uses the pycountry library to map country names to the ISO standard three-letter codes
    that GFIP uses as the universal country identifier. This ensures that UNODC data can
    be joined cleanly with every other data source in the master panel, regardless of the
    name spelling each source uses.

    Args:
        name: A country name string as it appears in the raw UNODC data, e.g. "Mexico".

    Returns:
        The ISO 3166-1 alpha-3 code (e.g. "MEX"), or None if pycountry cannot find a match.
    """
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


def load_unodc(path) -> pd.DataFrame:
    """Load a UNODC homicide CSV export and standardise it for the GFIP master panel.

    The raw UNODC file typically has one row per country-year, containing the homicide
    rate and optionally the absolute count. This function validates the file, standardises
    country identifiers to ISO 3166-1 alpha-3 codes, and renames columns to GFIP canonical
    snake_case names.

    Processing steps:
      1. Validate that required columns are present; raise immediately if not.
      2. Map each "Country" name to its ISO 3166-1 alpha-3 code using pycountry.
      3. Raise if any country name cannot be mapped.
      4. Drop the original "Country" column (replaced by "iso3") and rename remaining
         columns to GFIP canonical names.

    Note on missing "Count" column: if the raw file does not include the "Count" column,
    the rename will simply not produce a "homicide_count" column in the output. Downstream
    pipeline code handles optional columns gracefully.

    Args:
        path: Path to the UNODC homicide CSV file, or a file-like object (e.g. io.StringIO).
              The file must contain at minimum the columns "Country", "Year", and "Rate".

    Returns:
        A pandas DataFrame with one row per country-year, containing columns:
          - ``iso3`` (str): ISO 3166-1 alpha-3 country code.
          - ``year`` (int): Calendar year.
          - ``homicide_rate`` (float): Intentional homicides per 100,000 population.
          - ``homicide_count`` (float, optional): Absolute intentional homicide count.
            Present only if the source file contained a "Count" column. Many country-years
            have NaN here due to incomplete national reporting systems.

    Raises:
        ValueError: If any required columns are absent from the input file, with message
            "Missing required columns: [...]".
        ValueError: If any country name in the "Country" column cannot be matched to an
            ISO 3166-1 alpha-3 code, with message
            "unmapped countries — no ISO3 code found: [...]".
    """
    df = pd.read_csv(path)

    # Validate required columns immediately so failures are obvious and actionable.
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Map country names to ISO 3166-1 alpha-3 codes. UNODC country names are generally
    # standard English names, but some historical country names (e.g. former Soviet
    # republics using older spellings) or territories may need manual pre-processing.
    df["iso3"] = df["Country"].map(_to_iso3)

    # Surface any unmapped names immediately. Data quality issues are much easier to
    # resolve at ingest time than after they propagate silently into the master panel.
    unmapped = df.loc[df["iso3"].isna(), "Country"].unique().tolist()
    if unmapped:
        raise ValueError(f"unmapped countries — no ISO3 code found: {unmapped}")

    # Drop the original "Country" text column (now superseded by the standardised "iso3"
    # code) and rename the remaining columns to GFIP canonical snake_case names.
    # "Count" is renamed to "homicide_count" if the column exists; if not, the rename
    # dict simply has no effect on a non-existent column.
    return df.drop(columns=["Country"]).rename(
        columns={"Year": "year", "Rate": "homicide_rate", "Count": "homicide_count"}
    )

"""Ingest module for the Uppsala Conflict Data Program (UCDP) Georeferenced Event Dataset (GED).

The UCDP GED, produced by the Uppsala University Department of Peace and Conflict Research,
is the most widely used academic dataset for tracking organised armed conflict worldwide.
It records individual conflict events (battles, one-sided violence, non-state conflict)
from 1989 to the present, with each row representing a single violent episode including
information on the location, date, actor, and estimated deaths.

UCDP applies a strict definitional threshold: an "armed conflict" qualifies only if it
involves at least 25 battle-related deaths in a calendar year. This means the dataset
captures *organised* political violence — standing armies, rebel groups, paramilitary
organisations — rather than isolated riots, police incidents, or protests. For researchers,
this threshold makes UCDP conservative but highly reliable. Conflicts that fall below the
threshold (low-intensity violence) will not appear.

For GFIP, we aggregate from the raw event-level data to the country-year level and construct
two variables:

  - ``ucdp_conflict_binary``: 1 if any qualifying armed conflict occurred in that country
    in that year, 0 otherwise. This is the primary conflict outcome variable for Hypothesis 3
    (freshwater scarcity increases the probability of armed conflict).

  - ``ucdp_conflict_count``: the number of distinct conflict episodes recorded in that
    country-year. This captures intensity — a country with five simultaneous active conflicts
    is meaningfully different from one with a single contained insurgency.

Coverage: global, 1989-present. Data are released annually; there is typically a one-year
lag before a given calendar year's events are fully validated and published.

Source: https://ucdp.uu.se/downloads/

Canonical GFIP column names produced:
  - ``iso3``               — ISO 3166-1 alpha-3 country code
  - ``year``               — calendar year (integer)
  - ``ucdp_conflict_count``   — number of conflict events in this country-year
  - ``ucdp_conflict_binary``  — 1 if any conflict, 0 otherwise (always 1 in output rows;
                                countries with no conflict simply have no row)
"""

import pandas as pd
import pycountry

# The minimum set of columns that must be present in the raw UCDP CSV.
# "location" is UCDP's name for the country/region where the conflict occurred.
# "year" is the calendar year of the conflict event.
# If either is missing, the file is not a valid UCDP GED export and we fail immediately.
_REQUIRED_COLUMNS = ["location", "year"]


def _to_iso3(name: str) -> str | None:
    """Convert a country name string to its ISO 3166-1 alpha-3 code.

    Uses the pycountry library, which maps a wide variety of common and official country
    name spellings to ISO codes. Returns None when no match can be found, which triggers
    downstream validation to surface the unmapped names for manual review.

    All GFIP data uses ISO 3166-1 alpha-3 codes as the universal country identifier so that
    datasets from different sources (UCDP, World Bank, UNHCR, etc.) can be joined cleanly
    on a single key column. Without this standardisation, country name variations — "United
    States", "USA", "US", "United States of America" — would prevent joins across sources.

    Args:
        name: A country name string as it appears in the raw UCDP data, e.g. "Ethiopia".

    Returns:
        The ISO 3166-1 alpha-3 code (e.g. "ETH"), or None if pycountry cannot find a match.
    """
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


def load_ucdp(path) -> pd.DataFrame:
    """Load and aggregate a UCDP GED CSV export to the country-year level.

    The raw UCDP GED file has one row per conflict *event* — a single battle, massacre,
    or organised violence episode. A single country in a single year may have dozens or
    hundreds of such rows if it is experiencing active conflict. This function collapses
    that event-level data to one row per country-year, producing a count of events and a
    binary indicator of whether any conflict occurred at all.

    Processing steps:
      1. Validate that the required columns are present; raise immediately if not.
      2. Map each "location" country name to its ISO 3166-1 alpha-3 code using pycountry.
      3. Raise if any country name cannot be mapped — unmapped names indicate either a
         data quality issue or a new UCDP naming convention that needs a lookup update.
      4. Aggregate: group by (iso3, year), count rows to get ucdp_conflict_count, and
         assign ucdp_conflict_binary = 1 for all rows (every row in the output represents
         at least one recorded conflict event in that country-year).

    Important note on the binary indicator: the output only contains rows for country-years
    that had at least one UCDP-qualifying conflict. Country-years with zero conflict are
    *absent* from this dataframe. Downstream in the master panel assembly, a left join
    against the full country-year grid fills these gaps with 0, producing the correct
    binary indicator for all observations including peaceful country-years.

    Args:
        path: Path to the UCDP GED CSV file, or a file-like object (e.g. io.StringIO).
              The file must contain at minimum the columns "location" and "year".

    Returns:
        A pandas DataFrame with one row per country-year that had at least one recorded
        armed conflict, containing columns:
          - ``iso3`` (str): ISO 3166-1 alpha-3 country code.
          - ``year`` (int): Calendar year.
          - ``ucdp_conflict_count`` (int): Number of conflict events in this country-year.
          - ``ucdp_conflict_binary`` (int): Always 1 in this output; rows for country-years
            with no conflict are not present (they will become 0 after the master panel join).

    Raises:
        ValueError: If any required columns are absent from the input file, with message
            "Missing required columns: [...]".
        ValueError: If any country name in the "location" column cannot be matched to an
            ISO 3166-1 alpha-3 code, with message
            "unmapped countries — no ISO3 code found: [...]".
    """
    df = pd.read_csv(path)

    # Validate that the expected columns exist before doing anything else.
    # Failing early with a clear message is much more useful than a cryptic KeyError later.
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Standardise country identifiers. The UCDP "location" column contains country names
    # in UCDP's own naming convention, which mostly matches ISO standard names but has
    # some differences (e.g. historical names, contested territories). pycountry handles
    # most of these; edge cases must be pre-processed or mapped manually before ingest.
    df["iso3"] = df["location"].map(_to_iso3)

    # Any country that could not be mapped to an ISO3 code must be surfaced immediately.
    # Silently dropping unmapped rows would introduce systematic bias — conflicts in
    # countries that happen to have unusual name spellings would disappear from the analysis.
    unmapped = df.loc[df["iso3"].isna(), "location"].unique().tolist()
    if unmapped:
        raise ValueError(f"unmapped countries — no ISO3 code found: {unmapped}")

    # Aggregate from event-level to country-year level.
    #
    # The raw UCDP GED has one row per conflict event (a battle, massacre, etc.).
    # A country experiencing civil war may have hundreds of events in a single year.
    # We collapse to one row per (country, year) pair and compute:
    #
    #   ucdp_conflict_count  — the number of distinct events recorded. Using "size"
    #                          counts all rows in the group, which equals the number
    #                          of events since each row is one event.
    #
    #   ucdp_conflict_binary — always 1 for rows present in this output, because if a
    #                          country-year appears at all, at least one event occurred.
    #                          The 0 values are introduced later when this table is joined
    #                          to the full country-year panel.
    return (
        df.groupby(["iso3", "year"], as_index=False)
        .agg(ucdp_conflict_count=("location", "size"))
        .assign(ucdp_conflict_binary=1)
    )

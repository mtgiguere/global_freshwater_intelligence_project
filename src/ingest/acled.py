"""Ingest module for the Armed Conflict Location and Event Data (ACLED) Project.

ACLED is a disaggregated data collection, analysis, and crisis mapping project that
records the dates, actors, locations, fatalities, and types of all reported political
violence and protest events worldwide. It is produced by a non-governmental organisation
of the same name (acleddata.com) and covers events from 1997 to the present.

How ACLED differs from UCDP
----------------------------
UCDP (see ucdp.py) is GFIP's primary conflict dataset. ACLED serves a complementary role.
The key differences are:

  - *Scope*: UCDP captures only organised armed conflicts that exceed 25 battle-deaths per
    year — it deliberately excludes smaller-scale violence and non-violent events. ACLED
    captures a much wider spectrum: battles, explosions, violence against civilians,
    protests, riots, and even peaceful demonstrations. ACLED will record events that UCDP
    would classify as below-threshold.

  - *Geographic resolution*: ACLED records a specific location (city, village, coordinates)
    for each event, whereas UCDP records the country and a broad region. This gives ACLED
    finer spatial granularity, though for GFIP we aggregate to the country-year level anyway.

  - *Real-time updates*: ACLED updates weekly; UCDP updates annually with validation lag.
    ACLED can therefore reflect very recent events that have not yet been validated into UCDP.

  - *Actor definitions*: ACLED includes non-state actors such as militias, criminal groups,
    and mobs alongside state military forces. UCDP focuses specifically on organised armed
    groups with a political objective.

The broader ACLED definition makes it useful as:
  (a) a robustness check for UCDP-based findings (if an effect holds in both datasets it
      is not an artefact of UCDP's threshold);
  (b) a feature in the instability prediction model (Phase 4), where the wider event
      spectrum may improve predictive accuracy even if it is less theoretically clean.

For GFIP we construct two country-year variables:

  - ``acled_events_count``: total number of political violence and protest events in a
    country in a given year. This counts *all* ACLED event types.

  - ``acled_fatalities``: total number of fatalities reported across all events in that
    country-year. Fatality data is less reliable for protests and riots than for battles;
    treat as a directional indicator rather than a precise count.

Coverage: global, 1997-present (Africa from 1997; Asia from 2010; MENA, Latin America,
and South and Southeast Asia at varying start dates). Early years for non-African countries
may have significantly lower event counts due to less systematic data collection, not
necessarily lower actual conflict.

Source: https://acleddata.com/data-export-tool/

Canonical GFIP column names produced:
  - ``iso3``               — ISO 3166-1 alpha-3 country code
  - ``year``               — calendar year (integer, extracted from event_date)
  - ``acled_events_count`` — total political violence/protest events in this country-year
  - ``acled_fatalities``   — total reported fatalities in this country-year
"""

import pandas as pd
import pycountry

# The minimum set of columns that must be present in the raw ACLED CSV.
# "country"    — the country where the event occurred (ACLED's own naming)
# "event_date" — the date of the event (ISO format; we extract the year from this)
# "fatalities" — number of reported fatalities (0 for protests/non-violent events)
_REQUIRED_COLUMNS = ["country", "event_date", "fatalities"]


def _to_iso3(name: str) -> str | None:
    """Convert a country name string to its ISO 3166-1 alpha-3 code.

    Uses the pycountry library to map country names to the ISO standard three-letter codes
    that GFIP uses as its universal country identifier throughout the master panel. All
    datasets — ACLED, UCDP, World Bank, UNHCR, etc. — are joined on this single key.

    ACLED uses a mix of standard and non-standard country names. Most map cleanly via
    pycountry, but some edge cases (disputed territories, historical names, sub-national
    entities that ACLED treats as separate) may require pre-processing before ingest.

    Args:
        name: A country name string as it appears in the raw ACLED data, e.g. "Nigeria".

    Returns:
        The ISO 3166-1 alpha-3 code (e.g. "NGA"), or None if pycountry cannot find a match.
    """
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


def load_acled(path) -> pd.DataFrame:
    """Load and aggregate an ACLED CSV export to the country-year level.

    The raw ACLED file has one row per *event* — an individual battle, protest, riot,
    explosion, or other political violence or demonstration episode. Events are timestamped
    at the day level. This function extracts the year from the event date, then collapses
    all events to one row per country-year, summing fatalities and counting events.

    Processing steps:
      1. Validate that required columns are present; raise immediately if not.
      2. Extract the calendar year from the "event_date" column (ACLED stores full dates,
         e.g. "2019-04-07"; we need just the year for the GFIP annual panel).
      3. Map each "country" name to its ISO 3166-1 alpha-3 code using pycountry.
      4. Raise if any country name cannot be mapped.
      5. Aggregate: group by (iso3, year), count rows (acled_events_count), and sum
         fatalities (acled_fatalities).

    Args:
        path: Path to the ACLED CSV file, or a file-like object (e.g. io.StringIO).
              The file must contain at minimum the columns "country", "event_date",
              and "fatalities".

    Returns:
        A pandas DataFrame with one row per country-year, containing columns:
          - ``iso3`` (str): ISO 3166-1 alpha-3 country code.
          - ``year`` (int): Calendar year, extracted from "event_date".
          - ``acled_events_count`` (int): Total number of political violence and protest
            events recorded for this country in this year.
          - ``acled_fatalities`` (int): Total number of reported fatalities across all
            events in this country-year. Note: fatality counts are more reliable for
            armed clashes than for protests or riots.

    Raises:
        ValueError: If any required columns are absent from the input file, with message
            "Missing required columns: [...]".
        ValueError: If any country name in the "country" column cannot be matched to an
            ISO 3166-1 alpha-3 code, with message
            "unmapped countries — no ISO3 code found: [...]".
    """
    df = pd.read_csv(path)

    # Validate required columns upfront. Failing here with a clear error saves a
    # confusing KeyError later during aggregation.
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ACLED stores event dates as full calendar dates ("YYYY-MM-DD" or similar formats).
    # The GFIP master panel is annual, so we extract just the year for grouping.
    # pandas.to_datetime handles the various date formats ACLED has used across versions.
    df["year"] = pd.to_datetime(df["event_date"]).dt.year

    # Standardise country identifiers to ISO 3166-1 alpha-3 codes.
    # ACLED country names are generally standard English names, but some territories and
    # disputed regions may not map cleanly — these will surface as unmapped and must be
    # investigated and resolved before the data can enter the master panel.
    df["iso3"] = df["country"].map(_to_iso3)

    # Any unmapped country name must cause an immediate failure. Silently dropping rows
    # would mean conflict events in those places disappear from the analysis entirely,
    # introducing systematic undercounting for specific regions.
    unmapped = df.loc[df["iso3"].isna(), "country"].unique().tolist()
    if unmapped:
        raise ValueError(f"unmapped countries — no ISO3 code found: {unmapped}")

    # Aggregate from event-level to country-year level.
    #
    # acled_events_count: "size" counts the number of rows in each group. Each row in
    #   the raw ACLED file is one event, so this is the total number of political violence
    #   or protest events in that country that year — including battles, riots, explosions,
    #   attacks on civilians, and peaceful demonstrations.
    #
    # acled_fatalities: "sum" totals the fatality column across all events. Fatalities
    #   are 0 for non-violent events (e.g. peaceful protests), so the sum is dominated
    #   by armed clashes. A country with many protests but no armed conflict will have
    #   high acled_events_count and near-zero acled_fatalities.
    return df.groupby(["iso3", "year"], as_index=False).agg(
        acled_events_count=("fatalities", "size"), acled_fatalities=("fatalities", "sum")
    )

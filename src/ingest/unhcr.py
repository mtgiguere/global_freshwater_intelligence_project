"""Ingest module for the UN High Commissioner for Refugees (UNHCR) Refugee Statistics.

The UNHCR Refugee Statistics database is the world's authoritative source of data on
forced displacement. It tracks the number of refugees, internally displaced persons,
asylum seekers, and stateless persons at the country level, and is updated annually
based on registration data submitted by national governments and UNHCR field offices.
Coverage extends from 1951 (the year of the UN Refugee Convention) to the present,
though systematic country-level data becomes far more consistent from the 1990s onward.

Understanding forced displacement — key definitions
----------------------------------------------------
Three distinct categories of forced displacement appear in this dataset:

  1. *Refugees* (``refugee_outflow``): People who have crossed an international border
     to flee persecution, armed conflict, violence, or serious human rights violations,
     and who have been recognised or are seeking recognition under the 1951 Refugee
     Convention. For GFIP, we track *outflow* — people who LEFT a given country as
     refugees. This is the variable of interest for Hypothesis 5 (freshwater scarcity
     increases refugee outflow). Higher water stress → more people flee → higher outflow
     is the hypothesised causal chain.

  2. *Internally Displaced Persons* (``idp_count``): People who have been forced to flee
     their home but have remained within their own country's borders. IDPs are often in
     similar danger as refugees but lack the legal protections of the Refugee Convention
     because they have not crossed an international border. Water stress-driven displacement
     (e.g. from communities near dried-up rivers or failing agricultural systems) often
     produces IDPs before it produces refugees, because crossing a border requires more
     resources and opportunity than moving to the nearest city. IDP data is less reliable
     than refugee data, especially before 2000, because many countries do not systematically
     track internal displacement.

  3. *Asylum applications* (``asylum_applications_origin``): The number of asylum claims
     filed in foreign countries by people originating from this country. This is a leading
     indicator of displacement pressure — people may apply for asylum in large numbers even
     before formal refugee recognition. It also captures flows to high-income countries
     that are tracked through formal asylum systems, which may be underrepresented in raw
     refugee counts.

Directionality note
--------------------
UNHCR tracks displacement from both sides: the *origin* country (where people are fleeing
from) and the *host* country (where they are seeking safety). For GFIP, all three variables
are measured from the *origin country* perspective:
  - refugee_outflow    → people LEAVING country X as refugees
  - idp_count          → people displaced WITHIN country X
  - asylum_applications_origin → people FROM country X filing asylum claims abroad

This is the appropriate perspective for our research question: we want to know whether
water stress in a country causes more people to flee *from* it.

Coverage caveats
-----------------
  - Refugee and asylum data: generally reliable from ~1990 onward for most countries;
    pre-1990 data is sparse and should be used with caution.
  - IDP data: UNHCR IDP data has improved significantly since the 2010s but remains
    patchy for earlier years and for countries with limited government cooperation.
  - Stateless persons: not included in GFIP (no clear freshwater linkage).
  - Some conflict-affected countries may be systematically undercounted because UNHCR
    field operations have limited access.

Source: https://www.unhcr.org/refugee-statistics/download/

Canonical GFIP column names produced:
  - ``iso3``                       — ISO 3166-1 alpha-3 country code (origin country)
  - ``year``                       — calendar year (integer)
  - ``refugee_outflow``            — number of people who left this country as refugees
  - ``idp_count``                  — number of internally displaced persons within this country
  - ``asylum_applications_origin`` — asylum applications filed abroad by people from this country
"""

import pandas as pd
import pycountry

# The minimum set of columns that must be present in the raw UNHCR CSV.
# "Country"  — the origin country (where people are fleeing from)
# "Year"     — calendar year
# "Refugees" — refugee outflow count (the primary forced displacement outcome variable)
# IDPs and AsylumSeekers are optional columns; their absence does not raise an error,
# but they will be renamed if present.
_REQUIRED_COLUMNS = ["Country", "Year", "Refugees"]

# Mapping from raw UNHCR column names to GFIP canonical snake_case names.
# All downstream code — the master panel assembler, the API, the dashboard — uses these
# canonical names. Never reference the raw UNHCR names outside this module.
COLUMN_NAMES: dict[str, str] = {
    "Year": "year",
    "Refugees": "refugee_outflow",
    "IDPs": "idp_count",
    "AsylumSeekers": "asylum_applications_origin",
}


def _to_iso3(name: str) -> str | None:
    """Convert a country name string to its ISO 3166-1 alpha-3 code.

    Uses the pycountry library to map country names to the ISO standard three-letter codes
    that GFIP uses as the universal country identifier. This enables the UNHCR displacement
    data to join cleanly with every other dataset in the master panel on the "iso3" column.

    UNHCR uses its own country naming conventions for some regions — particularly for
    countries that have changed names, territories under UNHCR's mandate, or "stateless"
    situations not tied to a specific country. Such cases will return None and surface as
    unmapped names during validation.

    Args:
        name: A country name string as it appears in the raw UNHCR data, e.g. "Somalia".

    Returns:
        The ISO 3166-1 alpha-3 code (e.g. "SOM"), or None if pycountry cannot find a match.
    """
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


def load_unhcr(path) -> pd.DataFrame:
    """Load a UNHCR Refugee Statistics CSV export and standardise it for the GFIP master panel.

    The raw UNHCR file typically has one row per origin-country-year, with columns for
    different categories of forced displacement. This function validates the file,
    standardises country identifiers to ISO 3166-1 alpha-3 codes, and renames columns
    to GFIP canonical names.

    All three displacement variables (refugee outflow, IDPs, asylum applications) are
    measured from the *origin country* perspective — i.e., they answer the question
    "how many people fled FROM this country in this year?" This is the correct orientation
    for GFIP's Hypothesis 5, which asks whether water stress in a country drives people
    to flee from it.

    Processing steps:
      1. Validate that required columns are present; raise immediately if not.
      2. Map each "Country" name to its ISO 3166-1 alpha-3 code using pycountry.
      3. Raise if any country name cannot be mapped.
      4. Drop the original "Country" column (replaced by "iso3") and rename remaining
         columns to GFIP canonical snake_case names using the COLUMN_NAMES mapping.

    Args:
        path: Path to the UNHCR CSV file, or a file-like object (e.g. io.StringIO).
              The file must contain at minimum the columns "Country", "Year",
              and "Refugees".

    Returns:
        A pandas DataFrame with one row per origin-country-year, containing columns:
          - ``iso3`` (str): ISO 3166-1 alpha-3 code for the *origin* country
            (the country people are fleeing from).
          - ``year`` (int): Calendar year.
          - ``refugee_outflow`` (float): Number of people who left this country as
            internationally recognised refugees. This is the primary outcome variable
            for GFIP Hypothesis 5.
          - ``idp_count`` (float, if present): Number of people internally displaced
            within this country. Often missing or zero for years before ~2000.
          - ``asylum_applications_origin`` (float, if present): Number of asylum claims
            filed abroad by people originating from this country. A leading indicator of
            displacement pressure before formal refugee recognition occurs.

    Raises:
        ValueError: If any required columns are absent from the input file, with message
            "Missing required columns: [...]".
        ValueError: If any country name in the "Country" column cannot be matched to an
            ISO 3166-1 alpha-3 code, with message
            "unmapped countries — no ISO3 code found: [...]".
    """
    df = pd.read_csv(path)

    # Validate required columns upfront. A missing "Refugees" column is a strong signal
    # that the wrong UNHCR export was loaded (e.g. host-country data rather than origin).
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Map country names to ISO 3166-1 alpha-3 codes. UNHCR sometimes uses non-standard
    # names for territories under special mandate or contested jurisdictions. These will
    # surface as unmapped names and must be resolved before the data can enter the panel.
    df["iso3"] = df["Country"].map(_to_iso3)

    # Any unmapped country name must cause an immediate failure. Silent row-dropping
    # would mean entire displacement crises disappear from the analysis — exactly the
    # kind of systematic error that would bias any hypothesis test.
    unmapped = df.loc[df["iso3"].isna(), "Country"].unique().tolist()
    if unmapped:
        raise ValueError(f"unmapped countries — no ISO3 code found: {unmapped}")

    # Drop the raw "Country" text column and apply the COLUMN_NAMES mapping.
    # Optional columns (IDPs, AsylumSeekers) are renamed only if they exist; the rename
    # call silently skips keys that are not present in the dataframe.
    return df.drop(columns=["Country"]).rename(columns=COLUMN_NAMES)

"""Ingest module for the World Health Organization (WHO) Global Health Observatory data.

The WHO Global Health Observatory (GHO) is the WHO's primary data repository for
monitoring global health, covering mortality, disease burden, healthcare access, and
risk factors for every WHO member state. GFIP draws three indicators from the GHO that
directly connect water availability and sanitation to human health outcomes.

The three health indicators used in GFIP
-----------------------------------------
1. ``life_expectancy``  — Life expectancy at birth, in years. This is the expected number
   of years a newborn would live if current mortality rates remained constant throughout
   their life. It is the single most widely used summary measure of population health.
   Life expectancy is influenced by many factors — income, healthcare quality, nutrition,
   violence — but safe water access is among the most powerful determinants, particularly
   in low-income countries where waterborne disease drives a large share of mortality.

2. ``u5mr``  — Under-5 mortality rate, expressed as deaths per 1,000 live births. This
   is the probability that a child born in a given year will die before reaching their
   fifth birthday. U5MR is extremely sensitive to water and sanitation conditions: the
   leading causes of child death globally (diarrhoea, pneumonia) are closely tied to
   contaminated water and poor hygiene. U5MR fell dramatically worldwide between 1990
   and 2020 largely because of improved water access, sanitation, and oral rehydration
   therapy. It remains elevated in water-stressed, low-income countries.

3. ``diarrhoeal_daly``  — Disability-Adjusted Life Years (DALYs) lost to diarrhoeal
   disease per 100,000 population. See the note on DALYs below.

A note on DALYs for non-technical readers
------------------------------------------
A DALY is a composite measure of disease burden that combines two components:
  - *Years of Life Lost (YLL)*: the years lost due to premature death from a disease.
  - *Years Lived with Disability (YLD)*: the years spent living with illness or disability
    caused by a disease, weighted by the severity of that disability.

One DALY equals one year of healthy life lost. A country where diarrhoeal disease causes
1,000 DALYs per 100,000 people per year has lost the equivalent of one year of perfect
health for every hundred people in its population to diarrhoea alone.

DALYs are produced by the Global Burden of Disease (GBD) study, which the WHO incorporates
into the GHO. They are the standard metric in global health economics for comparing the
impact of different diseases across countries and time periods.

Why diarrhoeal DALYs specifically?
------------------------------------
Diarrhoeal disease is the most direct, well-documented pathway from contaminated water to
human harm. Faecal contamination of drinking water — caused by inadequate sanitation, open
defecation near water sources, or flooding — is the primary transmission vector for
diarrhoeal pathogens (cholera, rotavirus, E. coli, etc.). A higher diarrhoeal DALY burden
in a country is therefore a strong signal that water quality and sanitation are inadequate.
For GFIP Hypothesis 4 (safe water access improves health outcomes), diarrhoeal DALYs are
the most causally proximate health outcome variable we could choose.

Coverage: global, ~1990-present for life expectancy and U5MR; GBD DALY estimates from
1990 onwards. Data are based on a combination of civil registration records, household
surveys, and statistical modelling; quality varies substantially across countries.

Source: https://www.who.int/data/gho

Canonical GFIP column names produced:
  - ``iso3``              — ISO 3166-1 alpha-3 country code
  - ``year``              — calendar year (integer)
  - ``life_expectancy``   — life expectancy at birth, in years
  - ``u5mr``              — under-5 mortality rate per 1,000 live births
  - ``diarrhoeal_daly``   — DALYs lost to diarrhoeal disease per 100,000 population
"""

import pandas as pd
import pycountry

# The minimum set of columns that must be present in the raw WHO CSV.
# "Country"       — country name (WHO's own naming convention)
# "Year"          — calendar year
# "LifeExpectancy" — life expectancy at birth, in years (the anchor health variable)
# U5MR and DiarrhoeaDALY are optional in the sense that they may not be present in all
# WHO GHO exports, but they are expected and documented in COLUMN_NAMES below.
_REQUIRED_COLUMNS = ["Country", "Year", "LifeExpectancy"]

# Mapping from raw WHO column names to GFIP canonical snake_case names.
# All downstream code — the master panel assembler, the API, the dashboard — expects
# these canonical names. Never use the raw WHO names outside this module.
COLUMN_NAMES: dict[str, str] = {
    "Year": "year",
    "LifeExpectancy": "life_expectancy",
    "U5MR": "u5mr",
    "DiarrhoeaDALY": "diarrhoeal_daly",
}


def _to_iso3(name: str) -> str | None:
    """Convert a country name string to its ISO 3166-1 alpha-3 code.

    Uses the pycountry library to map country names to the ISO standard three-letter codes
    that GFIP uses as the universal country identifier. This ensures that WHO health data
    can be joined cleanly with every other data source in the master panel.

    WHO country names are generally standard English names, but the WHO occasionally uses
    its own naming conventions for territories, disputed regions, or former countries.
    Such cases will return None and surface as unmapped names during validation.

    Args:
        name: A country name string as it appears in the raw WHO data, e.g. "Bangladesh".

    Returns:
        The ISO 3166-1 alpha-3 code (e.g. "BGD"), or None if pycountry cannot find a match.
    """
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


def load_who(path) -> pd.DataFrame:
    """Load a WHO Global Health Observatory CSV export and standardise it for GFIP.

    The raw WHO file typically has one row per country-year, containing health indicators
    for that country in that year. This function validates the file, standardises country
    identifiers to ISO 3166-1 alpha-3 codes, and renames columns to GFIP canonical names.

    Processing steps:
      1. Validate that required columns are present; raise immediately if not.
      2. Map each "Country" name to its ISO 3166-1 alpha-3 code using pycountry.
      3. Raise if any country name cannot be mapped.
      4. Drop the original "Country" column (replaced by "iso3") and rename remaining
         columns to GFIP canonical snake_case names using the COLUMN_NAMES mapping.

    Args:
        path: Path to the WHO GHO CSV file, or a file-like object (e.g. io.StringIO).
              The file must contain at minimum the columns "Country", "Year", and
              "LifeExpectancy".

    Returns:
        A pandas DataFrame with one row per country-year, containing columns:
          - ``iso3`` (str): ISO 3166-1 alpha-3 country code.
          - ``year`` (int): Calendar year.
          - ``life_expectancy`` (float): Life expectancy at birth, in years.
          - ``u5mr`` (float, if present in source): Under-5 mortality rate per 1,000
            live births.
          - ``diarrhoeal_daly`` (float, if present in source): DALYs lost to diarrhoeal
            disease per 100,000 population. One DALY = one year of healthy life lost.

    Raises:
        ValueError: If any required columns are absent from the input file, with message
            "Missing required columns: [...]".
        ValueError: If any country name in the "Country" column cannot be matched to an
            ISO 3166-1 alpha-3 code, with message
            "unmapped countries — no ISO3 code found: [...]".
    """
    df = pd.read_csv(path)

    # Validate required columns immediately. A missing "LifeExpectancy" column is more
    # likely to be a sign that the wrong file was passed in than a genuine data issue.
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Map country names to ISO 3166-1 alpha-3 codes. WHO name variants that pycountry
    # cannot resolve will appear as None and surface in the unmapped check below.
    df["iso3"] = df["Country"].map(_to_iso3)

    # Fail immediately on any unmapped country. Silently dropping rows would mean some
    # countries' health data never enters the master panel — a serious analysis flaw.
    unmapped = df.loc[df["iso3"].isna(), "Country"].unique().tolist()
    if unmapped:
        raise ValueError(f"unmapped countries — no ISO3 code found: {unmapped}")

    # Drop the raw "Country" text column (now superseded by "iso3") and apply the
    # COLUMN_NAMES mapping to rename all WHO-specific column names to the GFIP standard.
    # Optional columns (U5MR, DiarrhoeaDALY) are renamed only if they are present in
    # the dataframe; the rename call silently skips keys that don't exist.
    return df.drop(columns=["Country"]).rename(columns=COLUMN_NAMES)

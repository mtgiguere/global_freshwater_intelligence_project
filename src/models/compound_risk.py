"""Compound Risk Score (CRS) — the headline indicator for the GFIP dashboard.

Combines the three model outputs into a single 0-100 index per country-year.
Higher = more at risk. Designed to be immediately legible to a non-technical
audience: a country scoring 80+ is in a critical situation.

Components and weights (from GFIP project plan):
  Water Scarcity Index:    30%  (how severe is current and projected water stress?)
  Instability Risk:        35%  (what is the probability of political collapse?)
  Migration Pressure:      35%  (how much forced displacement is the country generating?)

All three component inputs must be normalised to [0, 1] before combining.
The final score is multiplied by 100 to produce a human-readable 0-100 scale.

Uncertainty bands (not yet implemented — Phase 5):
  The dashboard will display CRS ± confidence interval, derived from
  the model's prediction intervals for each component.
"""

import numpy as np

WEIGHTS: dict[str, float] = {
    "scarcity": 0.30,
    "instability": 0.35,
    "migration": 0.35,
}
# Weight rationale (from the GFIP project plan):
#
# Instability and migration are weighted slightly higher than scarcity (35% vs 30%)
# because they represent acute, fast-moving human impacts — political collapse and
# forced displacement — that demand immediate attention from policymakers and
# humanitarian responders. Water scarcity, while serious, is a slower-developing
# structural condition that allows more lead time for adaptation and investment.
#
# This does NOT mean scarcity matters less; it means the urgency of response differs.
# A country can be chronically water-scarce (like many in the Middle East) and
# manage it for decades, whereas a sudden instability onset or refugee crisis
# compresses the response window to months.
#
# To adjust the weights (e.g. for a scarcity-focused policy brief), change ONLY
# this dictionary. The rest of the code will pick up the new values automatically.
# Ensure the three values still sum to 1.0 or the CRS will no longer be on a 0-100 scale.


def normalise_to_unit_interval(values: np.ndarray) -> np.ndarray:
    """Scale values to [0, 1] using min-max normalisation.

    Maps the observed range of a component score onto [0, 1] so that all three
    components are on a comparable scale before being combined into the CRS.
    The country with the worst score on this component maps to 1.0; the country
    with the best score maps to 0.0; all others fall in between proportionally.

    This is a cross-sectional normalisation — it compares countries to each other
    within the same prediction run, not to some fixed external standard. Scores
    should therefore be interpreted relative to the current set of countries, not
    as absolute physical quantities.

    Zero-variance case: if all countries have an identical value on this component
    (e.g. all instability probabilities happen to be 0.50), the denominator
    (hi - lo) would be zero, producing a division-by-zero NaN. Instead we return
    an array of zeros. The interpretation is: when a component does not discriminate
    between countries at all, it carries no information about which countries are
    more at risk — so it contributes zero to the CRS for all of them equally. This
    is mathematically equivalent to removing the component from the index entirely
    for that particular prediction run.

    Args:
        values: 1-D NumPy array of raw component scores (any numeric range).
            Typically the output of ``predict_scarcity``, ``predict_instability``,
            or ``predict_migration`` after collecting predictions for all countries.

    Returns:
        A NumPy array of the same shape as ``values``, with all entries in [0, 1].
        Returns an array of zeros if all input values are identical.
    """
    lo, hi = values.min(), values.max()
    if hi == lo:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def compute_compound_risk_score(
    scarcity: np.ndarray,
    instability: np.ndarray,
    migration: np.ndarray,
) -> np.ndarray:
    """Combine three normalised risk components into the Compound Risk Score (CRS).

    Each input must already be normalised to [0, 1] (use ``normalise_to_unit_interval``
    before calling this function):
        0 = no risk on this component relative to all countries in the current run
        1 = maximum risk on this component relative to all countries in the current run

    The output is scaled to [0, 100] for dashboard display.
    A score of 70+ is considered HIGH risk, 50-70 ELEVATED, <30 LOW.

    Example calculation:
        A country with scarcity=0.8 (very water-stressed), instability=0.9 (near-certain
        conflict onset), and migration=0.85 (heavy forced displacement) would score:
            (0.30 * 0.8) + (0.35 * 0.9) + (0.35 * 0.85) = 0.24 + 0.315 + 0.2975 = 0.8525
            * 100 = 85.25 out of 100 -- a critical situation requiring urgent attention.
        By contrast, a country with all three components at 0.1 would score 10 — low risk.

    Args:
        scarcity: 1-D NumPy array of normalised water scarcity scores in [0, 1].
            Higher = more severe projected water stress.
        instability: 1-D NumPy array of normalised instability risk scores in [0, 1].
            Higher = greater probability of political collapse or conflict onset.
        migration: 1-D NumPy array of normalised migration pressure scores in [0, 1].
            Higher = greater predicted forced displacement from this country.
        All three arrays must have the same length (one entry per country).

    Returns:
        A NumPy array of CRS values in [0, 100] (one per country). The ordering
        matches the ordering of the three input arrays — element i of the output
        corresponds to element i of each input.
    """
    return (
        WEIGHTS["scarcity"] * scarcity
        + WEIGHTS["instability"] * instability
        + WEIGHTS["migration"] * migration
    ) * 100.0

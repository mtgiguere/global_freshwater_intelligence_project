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


def normalise_to_unit_interval(values: np.ndarray) -> np.ndarray:
    """Scale values to [0, 1] using min-max normalisation.

    If all values are identical (zero variance), returns zeros — a country
    that is identical to all others on this dimension contributes zero risk
    differentiation on this component.
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
    """Combine three normalised risk components into the Compound Risk Score.

    Each input must already be normalised to [0, 1]:
      0 = no risk on this component
      1 = maximum risk on this component

    The output is scaled to [0, 100] for dashboard display.
    A score of 70+ is considered HIGH risk, 50-70 ELEVATED, <30 LOW.
    """
    return (
        WEIGHTS["scarcity"] * scarcity
        + WEIGHTS["instability"] * instability
        + WEIGHTS["migration"] * migration
    ) * 100.0

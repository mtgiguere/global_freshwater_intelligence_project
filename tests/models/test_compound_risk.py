"""Compound Risk Score — strict TDD.

The Compound Risk Score (CRS) combines the three model outputs into a single
0-100 index per country, designed for the public-facing dashboard.

Components and weights (from project plan):
  Water Scarcity Index:    30%
  Instability Risk:        35%
  Migration Pressure:      35%

Properties that MUST hold:
  - Score is always in [0, 100]
  - Higher score = more at risk
  - A country with max risk on all components scores 100
  - A country with zero risk on all components scores 0
  - Weights sum to 1.0
"""

import numpy as np
import pytest

from src.models.compound_risk import (
    WEIGHTS,
    compute_compound_risk_score,
    normalise_to_unit_interval,
)

# ---------------------------------------------------------------------------
# normalise_to_unit_interval
# ---------------------------------------------------------------------------


def test_normalise_returns_values_in_0_1():
    values = np.array([1.0, 3.0, 5.0, 2.0, 4.0])
    result = normalise_to_unit_interval(values)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_normalise_min_becomes_0_max_becomes_1():
    values = np.array([2.0, 4.0, 6.0])
    result = normalise_to_unit_interval(values)
    assert result[0] == pytest.approx(0.0)
    assert result[2] == pytest.approx(1.0)


def test_normalise_constant_array_returns_zeros():
    """All-same values have no variance — normalise to 0 rather than divide by 0."""
    values = np.array([5.0, 5.0, 5.0])
    result = normalise_to_unit_interval(values)
    assert (result == 0).all()


# ---------------------------------------------------------------------------
# WEIGHTS
# ---------------------------------------------------------------------------


def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_weights_all_positive():
    assert all(v > 0 for v in WEIGHTS.values())


# ---------------------------------------------------------------------------
# compute_compound_risk_score
# ---------------------------------------------------------------------------


def test_compound_risk_score_returns_series():
    scarcity = np.array([0.3, 0.7, 0.5])
    instability = np.array([0.4, 0.8, 0.2])
    migration = np.array([0.2, 0.6, 0.4])
    result = compute_compound_risk_score(scarcity, instability, migration)
    assert isinstance(result, np.ndarray)
    assert len(result) == 3


def test_compound_risk_score_is_in_0_100():
    """Score must always be in [0, 100] regardless of input values."""
    rng = np.random.default_rng(42)
    scarcity = rng.random(50)
    instability = rng.random(50)
    migration = rng.random(50)
    scores = compute_compound_risk_score(scarcity, instability, migration)
    assert scores.min() >= 0.0
    assert scores.max() <= 100.0


def test_compound_risk_score_all_zero_inputs_give_zero():
    """Zero risk on every component must produce a score of 0."""
    scores = compute_compound_risk_score(np.zeros(3), np.zeros(3), np.zeros(3))
    assert np.allclose(scores, 0.0)


def test_compound_risk_score_all_one_inputs_give_100():
    """Maximum risk on every component must produce a score of 100."""
    scores = compute_compound_risk_score(np.ones(3), np.ones(3), np.ones(3))
    assert np.allclose(scores, 100.0)


def test_compound_risk_score_higher_components_give_higher_score():
    """A country with worse conditions on all components must score higher."""
    low = compute_compound_risk_score(np.array([0.1]), np.array([0.1]), np.array([0.1]))
    high = compute_compound_risk_score(np.array([0.9]), np.array([0.9]), np.array([0.9]))
    assert high[0] > low[0]

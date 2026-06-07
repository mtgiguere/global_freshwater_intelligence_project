"""GFIP API endpoints — strict TDD.

The FastAPI backend serves the Master Panel data and ML model predictions
to the React dashboard. Every endpoint is tested against a synthetic data
fixture so tests run without the real parquet file.

Note: UI rendering is not tested here (see TDD contract). These tests
cover the API contract: correct status codes, response schemas, and
data transformations. They are the guarantee that the frontend can
depend on a stable interface.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def test_health_check_returns_200():
    """API must respond to health checks — used by deployment monitoring."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_check_returns_status_ok():
    response = client.get("/health")
    assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Global risk endpoint
# ---------------------------------------------------------------------------


def test_global_risk_returns_200():
    """GET /api/v1/global/risk must return 200."""
    response = client.get("/api/v1/global/risk")
    assert response.status_code == 200


def test_global_risk_returns_list():
    """Response must be a list of country risk objects."""
    response = client.get("/api/v1/global/risk")
    data = response.json()
    assert isinstance(data, list)


def test_global_risk_each_item_has_required_fields():
    """Each country risk object must have iso3, year, and compound_risk_score."""
    response = client.get("/api/v1/global/risk")
    data = response.json()
    if data:  # only check if data is non-empty
        item = data[0]
        assert "iso3" in item
        assert "year" in item
        assert "compound_risk_score" in item


def test_global_risk_scores_are_in_0_100():
    """Compound risk scores must be in [0, 100]."""
    response = client.get("/api/v1/global/risk")
    for item in response.json():
        assert 0 <= item["compound_risk_score"] <= 100


# ---------------------------------------------------------------------------
# Country detail endpoint
# ---------------------------------------------------------------------------


def test_country_detail_valid_iso3_returns_200():
    """A valid ISO3 code must return 200 with country data."""
    response = client.get("/api/v1/country/AFG")
    assert response.status_code == 200


def test_country_detail_invalid_iso3_returns_404():
    """An unknown ISO3 code must return 404."""
    response = client.get("/api/v1/country/ZZZ")
    assert response.status_code == 404


def test_country_detail_response_schema():
    """Country detail must include iso3, name, and time-series data."""
    response = client.get("/api/v1/country/AFG")
    data = response.json()
    assert "iso3" in data
    assert "timeseries" in data
    assert isinstance(data["timeseries"], list)


# ---------------------------------------------------------------------------
# Hypotheses summary endpoint
# ---------------------------------------------------------------------------


def test_hypotheses_summary_returns_200():
    """GET /api/v1/hypotheses must return all H1-H7 results."""
    response = client.get("/api/v1/hypotheses")
    assert response.status_code == 200


def test_hypotheses_summary_contains_all_hypotheses():
    """Response must include results for H1 through H7."""
    data = client.get("/api/v1/hypotheses").json()
    ids = {h["id"] for h in data}
    for expected in ["H1", "H2", "H3", "H4", "H4b", "H5", "H6", "H7"]:
        assert expected in ids


# ---------------------------------------------------------------------------
# Country prediction endpoint
# ---------------------------------------------------------------------------


def test_predict_country_returns_200_for_known_iso3():
    """GET /api/v1/predict/{iso3} must return 200 for a known country."""
    response = client.get("/api/v1/predict/AFG")
    assert response.status_code == 200


def test_predict_country_returns_200_with_neutral_fallback_for_unknown_iso3():
    """In synthetic mode (no panel/models), any ISO3 returns 200 with a neutral
    50/50/50 placeholder rather than 404.  The user may have just clicked any
    country on the map; a graceful response with is_trained=False is more useful
    than an error.  404 is reserved for the real-data path when the country is
    genuinely absent from the Master Panel."""
    response = client.get("/api/v1/predict/UGA")
    assert response.status_code == 200
    data = response.json()
    assert data["iso3"] == "UGA"
    assert data["is_trained"] is False
    assert data["scarcity_score"] == 0.50
    assert data["instability_probability"] == 0.50
    assert data["compound_risk_score"] == 50.0


def test_predict_country_response_has_required_fields():
    """Response must contain all CountryPrediction schema fields."""
    data = client.get("/api/v1/predict/AFG").json()
    for field in (
        "iso3",
        "country_name",
        "year",
        "scarcity_score",
        "instability_probability",
        "migration_score",
        "compound_risk_score",
        "is_trained",
    ):
        assert field in data, f"Missing field: {field}"


def test_predict_country_scores_are_in_valid_ranges():
    """Component scores must be [0,1] and compound_risk_score must be [0,100]."""
    data = client.get("/api/v1/predict/IND").json()
    assert 0 <= data["scarcity_score"] <= 1
    assert 0 <= data["instability_probability"] <= 1
    assert 0 <= data["migration_score"] <= 1
    assert 0 <= data["compound_risk_score"] <= 100


def test_predict_country_is_trained_is_false_without_model_file():
    """is_trained must be False when serving synthetic CI fallback data.

    The dashboard uses this flag to display a 'no trained model' caveat
    so users know the scores are illustrative, not real forecasts.
    """
    data = client.get("/api/v1/predict/USA").json()
    assert data["is_trained"] is False


def test_predict_country_iso3_is_case_insensitive():
    """Lowercase and mixed-case ISO3 codes must resolve the same as uppercase."""
    upper = client.get("/api/v1/predict/NGA").json()
    lower = client.get("/api/v1/predict/nga").json()
    assert upper["iso3"] == lower["iso3"]
    assert upper["compound_risk_score"] == lower["compound_risk_score"]


def test_predict_country_is_trained_true_when_models_loaded():
    """is_trained must be True when the endpoint uses real trained models.

    When both the Master Panel and all three model files are present, the
    endpoint must run the real models and set is_trained=True so the dashboard
    removes the synthetic-data warning banner.
    """
    # Build a minimal panel with all columns the feature builders need
    fake_panel = pd.DataFrame(
        {
            "iso3": ["AFG"] * 5,
            "year": [2017, 2018, 2019, 2020, 2021],
            "renewable_freshwater_percap": [1500.0] * 5,
            "gdp_pc_ppp": [500.0] * 5,
            "population": [38_000_000.0] * 5,
            "safe_water_access_pct": [55.0] * 5,
            "ucdp_conflict_binary": [1] * 5,
            "grace_lwe_anomaly_cm": [-2.0] * 5,
            "fsi_score": [108.0] * 5,
            "refugee_outflow": [2_500_000.0] * 5,
        }
    )

    # Instability model: XGBoost classifier — returns probability array
    fake_instability = MagicMock()
    fake_instability.predict_proba.return_value = np.array([[0.1, 0.9]])

    # Scarcity model: GBR — returns predicted log(freshwater_percap)
    fake_scarcity = MagicMock()
    fake_scarcity.predict.return_value = np.array([6.5])

    # Migration model: RF — returns predicted log(refugee_outflow+1)
    fake_migration = MagicMock()
    fake_migration.predict.return_value = np.array([14.7])

    fake_models = {
        "instability": fake_instability,
        "scarcity": fake_scarcity,
        "migration": fake_migration,
        "norm": {"scarcity": {"min": 0.0, "max": 10.0}, "migration": {"min": 0.0, "max": 20.0}},
    }

    with (
        patch("src.api.main._load_panel", return_value=fake_panel),
        patch("src.api.main._load_models", return_value=fake_models),
    ):
        response = client.get("/api/v1/predict/AFG")

    assert response.status_code == 200
    assert response.json()["is_trained"] is True

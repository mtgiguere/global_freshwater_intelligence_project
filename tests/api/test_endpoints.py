"""GFIP API endpoints — strict TDD.

The FastAPI backend serves the Master Panel data and ML model predictions
to the React dashboard. Every endpoint is tested against a synthetic data
fixture so tests run without the real parquet file.

Note: UI rendering is not tested here (see TDD contract). These tests
cover the API contract: correct status codes, response schemas, and
data transformations. They are the guarantee that the frontend can
depend on a stable interface.
"""

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


def test_predict_country_returns_404_for_unknown_iso3():
    """An ISO3 code not in the panel must return 404, not a 500."""
    response = client.get("/api/v1/predict/ZZZ")
    assert response.status_code == 404


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

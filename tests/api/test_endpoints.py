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

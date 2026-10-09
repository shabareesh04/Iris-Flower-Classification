"""
Integration and functional tests for the Iris Classification Flask web application.
"""

import pytest
import sys
import os

# Ensure project root is in Python module search path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app


@pytest.fixture
def client():
    """Create Flask test client with testing mode activated."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_homepage_loads(client):
    """Test that the homepage renders successfully with HTTP 200 and expected titles."""
    response = client.get("/")
    assert response.status_code == 200
    html_text = response.data.decode("utf-8")
    assert "Iris Flower Classification" in html_text
    assert "Predict Flower" in html_text
    assert "Sepal Length" in html_text


def test_predict_valid_form_submission(client):
    """Test standard form POST with valid Iris Setosa measurements."""
    form_data = {
        "sepal_length": "5.1",
        "sepal_width": "3.5",
        "petal_length": "1.4",
        "petal_width": "0.2",
    }
    response = client.post("/predict", data=form_data)
    assert response.status_code == 200
    html_text = response.data.decode("utf-8")
    assert "Prediction Result" in html_text
    assert "Iris Setosa" in html_text


def test_predict_valid_json_payload(client):
    """Test REST API JSON POST endpoint."""
    json_data = {
        "sepal_length": 6.5,
        "sepal_width": 3.0,
        "petal_length": 5.5,
        "petal_width": 1.8,
    }
    response = client.post("/predict", json=json_data)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["species"] in ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]
    assert "measurements" in data


def test_predict_missing_fields(client):
    """Test form POST with missing measurements."""
    incomplete_data = {
        "sepal_length": "5.1",
        "sepal_width": "3.5",
        # petal_length is missing
        "petal_width": "0.2",
    }
    response = client.post("/predict", data=incomplete_data)
    assert response.status_code == 400
    html_text = response.data.decode("utf-8")
    assert "Validation Error" in html_text


def test_predict_non_numeric_input(client):
    """Test form POST with non-numeric text input."""
    bad_data = {
        "sepal_length": "five",
        "sepal_width": "3.5",
        "petal_length": "1.4",
        "petal_width": "0.2",
    }
    response = client.post("/predict", data=bad_data)
    assert response.status_code == 400
    html_text = response.data.decode("utf-8")
    assert "numeric" in html_text.lower()


def test_predict_negative_values(client):
    """Test form POST rejecting negative values."""
    negative_data = {
        "sepal_length": "-5.1",
        "sepal_width": "3.5",
        "petal_length": "1.4",
        "petal_width": "0.2",
    }
    response = client.post("/predict", data=negative_data)
    assert response.status_code == 400
    html_text = response.data.decode("utf-8")
    assert "greater than zero" in html_text.lower()


def test_predict_unrealistic_values(client):
    """Test form POST rejecting biologically unrealistic values."""
    unrealistic_data = {
        "sepal_length": "50.0",
        "sepal_width": "3.5",
        "petal_length": "1.4",
        "petal_width": "0.2",
    }
    response = client.post("/predict", data=unrealistic_data)
    assert response.status_code == 400
    html_text = response.data.decode("utf-8")
    assert "unrealistic" in html_text.lower()


def test_api_info_endpoint(client):
    """Test GET /api/info returns model information."""
    response = client.get("/api/info")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "model_info" in data
    assert "bounds" in data

"""Automated tests for validation, prediction API, and presets."""

import json
import os
import sys

import pytest

# Ensure project root is on the path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import app, MODEL, THRESHOLD, MEANS, LIMITS, parse_application


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_model_loaded():
    assert MODEL is not None
    assert 0 < THRESHOLD < 1
    assert MEANS is not None
    assert len(MEANS) == 28


def test_metrics_endpoint(client):
    r = client.get("/api/metrics")
    assert r.status_code == 200
    data = r.get_json()
    assert "test" in data
    t = data["test"]
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert key in t
        assert 0 <= t[key] <= 1
    # Verify published numbers within tolerance of saved metrics
    assert abs(t["accuracy"] - 0.8624) < 0.01
    assert abs(t["roc_auc"] - 0.8695) < 0.01
    assert abs(data["threshold"] - 0.37) < 1e-6


def test_parse_valid_application():
    row = parse_application(
        {
            "person_age": 30,
            "person_income": 65000,
            "person_emp_length": 5,
            "loan_amnt": 10000,
            "loan_int_rate": "",
            "cb_person_cred_hist_length": 6,
            "person_home_ownership": "RENT",
            "loan_intent": "PERSONAL",
            "loan_grade": "B",
            "cb_person_default_on_file": "N",
        }
    )
    assert abs(row["loan_percent_income"] - round(10000 / 65000, 4)) < 1e-9
    assert row["loan_int_rate"] != row["loan_int_rate"]  # NaN


def test_missing_required_field():
    with pytest.raises(ValueError, match="Age is required"):
        parse_application(
            {
                "person_income": 65000,
                "person_emp_length": 5,
                "loan_amnt": 10000,
                "cb_person_cred_hist_length": 6,
                "person_home_ownership": "RENT",
                "loan_intent": "PERSONAL",
                "loan_grade": "B",
                "cb_person_default_on_file": "N",
            }
        )


def test_out_of_range_loan():
    with pytest.raises(ValueError, match="Loan amount must be between"):
        parse_application(
            {
                "person_age": 30,
                "person_income": 65000,
                "person_emp_length": 5,
                "loan_amnt": 200000,
                "cb_person_cred_hist_length": 6,
                "person_home_ownership": "RENT",
                "loan_intent": "PERSONAL",
                "loan_grade": "B",
                "cb_person_default_on_file": "N",
            }
        )


def test_out_of_range_income():
    with pytest.raises(ValueError, match="Annual income must be between"):
        parse_application(
            {
                "person_age": 30,
                "person_income": 3000000,
                "person_emp_length": 5,
                "loan_amnt": 10000,
                "cb_person_cred_hist_length": 6,
                "person_home_ownership": "RENT",
                "loan_intent": "PERSONAL",
                "loan_grade": "B",
                "cb_person_default_on_file": "N",
            }
        )


def test_loan_exceeds_income():
    with pytest.raises(ValueError, match="cannot exceed annual income"):
        parse_application(
            {
                "person_age": 30,
                "person_income": 10000,
                "person_emp_length": 5,
                "loan_amnt": 15000,
                "cb_person_cred_hist_length": 6,
                "person_home_ownership": "RENT",
                "loan_intent": "PERSONAL",
                "loan_grade": "B",
                "cb_person_default_on_file": "N",
            }
        )


def test_invalid_categorical():
    with pytest.raises(ValueError, match="Home ownership must be one of"):
        parse_application(
            {
                "person_age": 30,
                "person_income": 65000,
                "person_emp_length": 5,
                "loan_amnt": 10000,
                "cb_person_cred_hist_length": 6,
                "person_home_ownership": "LEASE",
                "loan_intent": "PERSONAL",
                "loan_grade": "B",
                "cb_person_default_on_file": "N",
            }
        )


def test_predict_default_form(client):
    payload = {
        "person_age": 30,
        "person_income": 65000,
        "person_emp_length": 5,
        "loan_amnt": 10000,
        "loan_int_rate": "",
        "cb_person_cred_hist_length": 6,
        "person_home_ownership": "RENT",
        "loan_intent": "PERSONAL",
        "loan_grade": "B",
        "cb_person_default_on_file": "N",
    }
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 200
    data = r.get_json()
    assert "probability" in data
    assert 0 <= data["probability"] <= 1
    assert data["threshold"] == pytest.approx(THRESHOLD)
    assert data["tier"] in ("Low", "Moderate", "High", "Very high")
    assert "drivers" in data
    assert isinstance(data["drivers"], list)
    assert len(data["drivers"]) > 0  # means key fixed → drivers available


def test_predict_low_risk_preset(client):
    payload = {
        "person_age": 38,
        "person_income": 95000,
        "person_emp_length": 10,
        "cb_person_cred_hist_length": 12,
        "loan_amnt": 8000,
        "loan_int_rate": "",
        "person_home_ownership": "MORTGAGE",
        "loan_intent": "HOMEIMPROVEMENT",
        "loan_grade": "A",
        "cb_person_default_on_file": "N",
    }
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 200
    data = r.get_json()
    assert data["probability"] < THRESHOLD
    assert data["high_risk"] is False


def test_predict_high_risk_preset(client):
    payload = {
        "person_age": 23,
        "person_income": 28000,
        "person_emp_length": 1,
        "cb_person_cred_hist_length": 2,
        "loan_amnt": 15000,
        "loan_int_rate": "",
        "person_home_ownership": "RENT",
        "loan_intent": "MEDICAL",
        "loan_grade": "E",
        "cb_person_default_on_file": "Y",
    }
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 200
    data = r.get_json()
    assert data["probability"] >= THRESHOLD
    assert data["high_risk"] is True


def test_predict_rejects_ood_loan(client):
    payload = {
        "person_age": 30,
        "person_income": 65000,
        "person_emp_length": 5,
        "loan_amnt": 200000,
        "cb_person_cred_hist_length": 6,
        "person_home_ownership": "RENT",
        "loan_intent": "PERSONAL",
        "loan_grade": "B",
        "cb_person_default_on_file": "N",
    }
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 400
    assert "Loan amount" in r.get_json()["error"]


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Credit Risk" in r.data


def test_unknown_route_keeps_404_status(client):
    response = client.get("/this-route-does-not-exist")
    assert response.status_code == 404
    assert "error" in response.get_json()


def test_wrong_method_keeps_405_status(client):
    response = client.get("/api/predict")
    assert response.status_code == 405
    assert "error" in response.get_json()

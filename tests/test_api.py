from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def _make_body(overrides: dict | None = None) -> dict:
    """Full Telco payload matching training-time columns."""
    data = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",  # or "Fiber optic" or "No"
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }
    if overrides:
        data.update(overrides)
    return {"customer_id": "0001", "data": data}


def test_predict_route():
    resp = client.post("/predict", json=_make_body())
    assert resp.status_code == 200
    js = resp.json()
    assert "churn_probability" in js
    assert 0.0 <= js["churn_probability"] <= 1.0


def test_top_features_len():
    resp = client.post("/predict", json=_make_body())
    assert resp.status_code == 200
    js = resp.json()
    assert "top_features" in js
    assert len(js["top_features"]) == 3

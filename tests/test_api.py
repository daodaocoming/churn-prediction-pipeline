from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_predict_route():
    body = {"customer_id":"0001",
            "data": {"gender":"Male","SeniorCitizen":0,"tenure":1,
                     "PhoneService":"Yes","Contract":"Month-to-month",
                     "MonthlyCharges":29.85,"TotalCharges":29.85}}
    resp = client.post("/predict", json=body)
    assert resp.status_code == 200
    assert 0 <= resp.json()["churn_probability"] <= 1

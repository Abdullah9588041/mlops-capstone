"""API contract tests (in-process TestClient)."""

import pandas as pd
from fastapi.testclient import TestClient

from capstone import api
from capstone.api import app

TELCO_CSV = "/home/hatch/workspace/portfolio-projects/customer-churn-mlops/data/Telco-Customer-Churn.csv"

PAYLOAD_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def _payload(i: int = 0) -> dict:
    row = pd.read_csv(TELCO_CSV).iloc[i]
    out = {}
    for col in PAYLOAD_COLS:
        val = row[col]
        if col == "TotalCharges":
            s = str(val).strip()
            out[col] = None if s == "" else float(s)
        elif col in ("SeniorCitizen", "tenure"):
            out[col] = int(val)
        elif col == "MonthlyCharges":
            out[col] = float(val)
        else:
            out[col] = str(val)
    return out


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "xgboost_tuned"
    assert body["threshold"] == 0.2


def test_predict_valid():
    client = TestClient(app)
    r = client.post("/predict", json=_payload(0))
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)
    assert body["threshold"] == 0.2


def test_predict_invalid_rejected():
    client = TestClient(app)
    bad = _payload(0)
    bad["tenure"] = 9999  # violates Field(le=200)
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_metrics_endpoint_counts_requests():
    api.metrics.reset()
    client = TestClient(app)
    client.post("/predict", json=_payload(1))
    client.post("/predict", json=_payload(2))
    r = client.get("/metrics")
    assert r.status_code == 200
    text = r.text
    assert "capstone_requests_total 2" in text
    assert "capstone_request_latency_seconds_bucket" in text
    assert 'capstone_predictions_total{class="churn"}' in text

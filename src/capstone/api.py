"""Production FastAPI service for churn prediction.

Loads the champion bundle (``models/champion.pkl``, lineage in
``models/lineage.json``) and serves:

- ``GET /health``   — liveness + model metadata
- ``POST /predict`` — single customer -> churn probability + decision
- ``GET /metrics``  — Prometheus-style counters (requests, errors, latency
  histogram, prediction distribution)

Structured JSON logging on every request; latency recorded per request.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from .preprocess import prepare_rows

BUNDLE_PATH = os.environ.get("CHURN_MODEL_PATH", "models/champion.pkl")

# ---------------------------------------------------------------- logging ---
logger = logging.getLogger("capstone.api")
_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def _log(event: str, **fields: object) -> None:
    logger.info(json.dumps({"event": event, **fields}))


# ---------------------------------------------------------------- metrics ---
_LAT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]


class Metrics:
    """Tiny in-process Prometheus-style metrics (single worker)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.requests_total = 0
        self.errors_total = 0
        self.latency_bucket = {b: 0 for b in _LAT_BUCKETS}
        self.latency_bucket[float("inf")] = 0
        self.latency_sum = 0.0
        self.pred_churn = 0
        self.pred_no_churn = 0

    def observe(self, latency_s: float, prediction: int | None, error: bool = False) -> None:
        self.requests_total += 1
        self.latency_sum += latency_s
        for b in self.latency_bucket:
            if latency_s <= b:
                self.latency_bucket[b] += 1
        if error:
            self.errors_total += 1
        elif prediction == 1:
            self.pred_churn += 1
        else:
            self.pred_no_churn += 1

    def render(self) -> str:
        lines = [
            "# HELP capstone_requests_total Total prediction requests",
            "# TYPE capstone_requests_total counter",
            f"capstone_requests_total {self.requests_total}",
            "# HELP capstone_errors_total Total failed requests",
            "# TYPE capstone_errors_total counter",
            f"capstone_errors_total {self.errors_total}",
            "# HELP capstone_request_latency_seconds Request latency histogram",
            "# TYPE capstone_request_latency_seconds histogram",
        ]
        for b, c in self.latency_bucket.items():
            le = "+Inf" if b == float("inf") else str(b)
            lines.append(f'capstone_request_latency_seconds_bucket{{le="{le}"}} {c}')
        lines.append(f"capstone_request_latency_seconds_sum {self.latency_sum:.6f}")
        lines.append(f"capstone_request_latency_seconds_count {self.requests_total}")
        lines += [
            "# HELP capstone_predictions_total Predictions by class",
            "# TYPE capstone_predictions_total counter",
            f'capstone_predictions_total{{class="churn"}} {self.pred_churn}',
            f'capstone_predictions_total{{class="no_churn"}} {self.pred_no_churn}',
        ]
        return "\n".join(lines) + "\n"


metrics = Metrics()

# ------------------------------------------------------------------ model ---
app = FastAPI(title="Churn Prediction API (MLOps capstone)", version="0.1.0")

_bundle: dict | None = None


def get_bundle() -> dict:
    """Lazy-load the model bundle (works under TestClient too)."""
    global _bundle
    if _bundle is None:
        if not os.path.exists(BUNDLE_PATH):
            raise HTTPException(
                status_code=503,
                detail=f"Model bundle not found at {BUNDLE_PATH}. "
                       "Run scripts/fetch_model.py first.",
            )
        with open(BUNDLE_PATH, "rb") as f:
            _bundle = pickle.load(f)
    return _bundle


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=200)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float | None = Field(default=None, ge=0)


def _predict_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    bundle = get_bundle()
    df = prepare_rows(df_raw)
    X = bundle["preprocessor"].transform(df)
    proba = bundle["model"].predict_proba(X)[:, 1]
    threshold = bundle["threshold"]
    return pd.DataFrame({
        "churn_probability": proba.round(4),
        "churn_prediction": (proba >= threshold).astype(int),
        "threshold": threshold,
    })


@app.get("/health")
def health():
    bundle = get_bundle()
    return {
        "status": "ok",
        "model": bundle.get("model_name"),
        "threshold": bundle.get("threshold"),
        "trained_at": bundle.get("trained_at"),
    }


@app.post("/predict")
def predict(customer: Customer):
    start = time.perf_counter()
    try:
        df = pd.DataFrame([customer.model_dump()])
        row = _predict_df(df).iloc[0]
        latency = time.perf_counter() - start
        metrics.observe(latency, int(row["churn_prediction"]))
        _log("predict", latency_s=round(latency, 4),
             proba=float(row["churn_probability"]),
             prediction=int(row["churn_prediction"]))
        return {
            "churn_probability": float(row["churn_probability"]),
            "churn_prediction": int(row["churn_prediction"]),
            "threshold": float(row["threshold"]),
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — 500s must still be counted
        latency = time.perf_counter() - start
        metrics.observe(latency, None, error=True)
        _log("predict_error", latency_s=round(latency, 4), error=str(exc))
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc


@app.get("/metrics")
def prometheus_metrics():
    return Response(content=metrics.render(), media_type="text/plain")

"""Concurrent load test against a running API.

Fires N POST /predict requests from a thread pool, then reports p50/p95/p99
latency, throughput, and error count. All numbers are measured, never invented.

Usage:
    # terminal 1: PYTHONPATH=src uvicorn capstone.api:app --port 8000
    # terminal 2: python scripts/load_test.py [--n 500] [--workers 20]
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TELCO_CSV = Path("/home/hatch/workspace/portfolio-projects/customer-churn-mlops/data/Telco-Customer-Churn.csv")
OUT = ROOT / "results" / "load_test.json"

PAYLOAD_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def row_to_payload(row: pd.Series) -> dict:
    payload = {}
    for col in PAYLOAD_COLS:
        val = row[col]
        if col == "TotalCharges":
            s = str(val).strip()
            payload[col] = None if s == "" else float(s)
        elif col in ("SeniorCitizen", "tenure"):
            payload[col] = int(val)
        elif col == "MonthlyCharges":
            payload[col] = float(val)
        else:
            payload[col] = str(val)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()

    df = pd.read_csv(TELCO_CSV)
    rng = np.random.default_rng(args.seed)
    payloads = [row_to_payload(df.iloc[int(rng.integers(len(df)))]) for _ in range(args.n)]

    latencies: list[float] = []
    errors = 0

    def one(payload: dict) -> None:
        nonlocal errors
        t0 = time.perf_counter()
        try:
            # trust_env=False: ignore sandbox proxy env vars (malformed here;
            # localhost traffic needs no proxy anyway).
            with httpx.Client(timeout=30.0, trust_env=False) as client:
                r = client.post(f"{args.api_url}/predict", json=payload)
                r.raise_for_status()
            latencies.append(time.perf_counter() - t0)
        except Exception:  # noqa: BLE001 — count, don't crash
            errors += 1

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(one, payloads))
    total = time.perf_counter() - t_start

    lat = np.array(latencies) * 1000  # ms
    result = {
        "n_requests": args.n,
        "workers": args.workers,
        "seed": args.seed,
        "errors": errors,
        "latency_ms": {
            "p50": round(float(np.percentile(lat, 50)), 2),
            "p95": round(float(np.percentile(lat, 95)), 2),
            "p99": round(float(np.percentile(lat, 99)), 2),
            "mean": round(float(lat.mean()), 2),
            "max": round(float(lat.max()), 2),
        },
        "throughput_rps": round(len(latencies) / total, 1),
        "total_s": round(total, 2),
        "note": "Single uvicorn worker, localhost, model = calibrated XGBClassifier.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

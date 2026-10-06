"""Synthetic production-traffic generator (the `monitor` compose service).

Replays historical Telco rows as stand-in live traffic: samples seeded rows,
POSTs them to the API's /predict, and appends one JSON line per request to
``monitoring/traffic_logs/traffic.jsonl`` (timestamp, latency, probability,
prediction). This is SIMULATED traffic — see README "What is real vs simulated".

Usage:
    python monitoring/traffic.py --api-url http://localhost:8000 --n-requests 200 --interval 0.5 --seed 11
    (--n-requests 0 = run forever)
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TELCO_CSV = Path("/home/hatch/workspace/portfolio-projects/customer-churn-mlops/data/Telco-Customer-Churn.csv")
LOG_DIR = ROOT / "monitoring" / "traffic_logs"

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
    ap.add_argument("--n-requests", type=int, default=200)
    ap.add_argument("--interval", type=float, default=0.5,
                    help="seconds between requests (0 = as fast as possible)")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "traffic.jsonl"

    df = pd.read_csv(TELCO_CSV)
    rng = np.random.default_rng(args.seed)

    ok, err = 0, 0
    i = 0
    # trust_env=False: ignore sandbox proxy env vars (malformed here;
    # localhost traffic needs no proxy anyway).
    with httpx.Client(timeout=30.0, trust_env=False) as client, open(log_path, "a") as f:
        while args.n_requests == 0 or i < args.n_requests:
            row = df.iloc[int(rng.integers(len(df)))]
            t0 = time.perf_counter()
            try:
                r = client.post(f"{args.api_url}/predict",
                                json=row_to_payload(row))
                r.raise_for_status()
                body = r.json()
                ok += 1
                status = "ok"
            except Exception as exc:  # noqa: BLE001 — traffic gen must not die
                body = {}
                err += 1
                status = f"error: {exc}"
            latency = time.perf_counter() - t0
            f.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "latency_s": round(latency, 4),
                "status": status,
                "churn_probability": body.get("churn_probability"),
                "churn_prediction": body.get("churn_prediction"),
            }) + "\n")
            f.flush()
            i += 1
            if args.interval > 0:
                time.sleep(args.interval)
    print(f"traffic done: {ok} ok, {err} errors, log -> {log_path}")


if __name__ == "__main__":
    main()

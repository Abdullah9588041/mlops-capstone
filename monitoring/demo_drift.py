"""Demonstrate the drift detector: control (no drift) vs injected drift.

- CONTROL: seeded sample from the training data -> expect overall "ok"
  (a few "warning"s are acceptable sampling noise; zero "drift" expected).
- DRIFTED: same sample with synthetic shifts applied ->
  tenure +15 months (clipped), MonthlyCharges x1.35, and 40% of
  "Two year" contracts flipped to "Month-to-month" -> expect "drift".

Usage:
    python monitoring/demo_drift.py [--n 3000] [--seed 7]

Writes results/demo_drift.json and prints a summary. All numbers are real.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from capstone.drift import check_drift, load_baseline, send_alert

ROOT = Path(__file__).resolve().parents[1]
TELCO_CSV = Path("/home/hatch/workspace/portfolio-projects/customer-churn-mlops/data/Telco-Customer-Churn.csv")
OUT = ROOT / "results" / "demo_drift.json"


def inject_drift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    df["tenure"] = (df["tenure"] + 15).clip(upper=200)
    df["MonthlyCharges"] = (df["MonthlyCharges"] * 1.35).round(2)
    mask = (df["Contract"] == "Two year") & (rng.random(len(df)) < 0.40)
    df.loc[mask, "Contract"] = "Month-to-month"
    # keep TotalCharges roughly consistent with the new tenure/charges
    df["TotalCharges"] = (df["MonthlyCharges"] * df["tenure"]).round(2)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    baseline = load_baseline()
    raw = pd.read_csv(TELCO_CSV)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(raw), size=min(args.n, len(raw)), replace=False)
    sample = raw.iloc[idx].reset_index(drop=True)

    control = check_drift(sample, baseline)
    drifted_df = inject_drift(sample, rng)
    drifted = check_drift(drifted_df, baseline)

    summary = {
        "seed": args.seed,
        "n": int(len(sample)),
        "control": {
            "overall": control["overall"],
            "drifted_features": control["drifted_features"],
            "warned_features": control["warned_features"],
        },
        "drifted": {
            "overall": drifted["overall"],
            "drifted_features": drifted["drifted_features"],
            "warned_features": drifted["warned_features"],
            "psi": {c: drifted["features"][c]["psi"]
                    for c in drifted["drifted_features"]},
        },
        "expectation": "control -> ok (no drift flags); drifted -> drift",
        "met": control["overall"] != "drift"
               and not control["drifted_features"]
               and drifted["overall"] == "drift",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if drifted["overall"] == "drift":
        send_alert(f"demo: drift detected in {drifted['drifted_features']}")
    print("expectation met:", summary["met"])


if __name__ == "__main__":
    main()

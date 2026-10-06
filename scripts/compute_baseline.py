"""Compute the training baseline for drift detection.

Reads the same Telco CSV used to train the P5 model, applies the vendored
clean/engineer pipeline, and stores per-feature reference distributions in
``monitoring/baseline_stats.json``:

- numeric: quantile bin edges + bin proportions + a baseline sample (for KS)
- categorical: category list + proportions

Usage:
    python scripts/compute_baseline.py [--n-sample 2000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from capstone.preprocess import (
    ENGINEERED_CATEGORICAL,
    ENGINEERED_NUMERIC,
    prepare_rows,
)

ROOT = Path(__file__).resolve().parents[1]
TELCO_CSV = Path("/home/hatch/workspace/portfolio-projects/customer-churn-mlops/data/Telco-Customer-Churn.csv")
OUT = ROOT / "monitoring" / "baseline_stats.json"
N_BINS = 10
SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sample", type=int, default=2000)
    args = ap.parse_args()

    raw = pd.read_csv(TELCO_CSV)
    df = prepare_rows(raw)
    rng = np.random.default_rng(SEED)

    numeric: dict = {}
    for col in ENGINEERED_NUMERIC:
        vals = df[col].dropna().to_numpy(dtype=float)
        edges = np.quantile(vals, np.linspace(0, 1, N_BINS + 1)).tolist()
        # guard against duplicate edges on discrete columns (is_new etc.)
        edges = sorted(set(edges))
        if len(edges) < 3:
            edges = [float(vals.min() - 0.5), float(vals.max() + 0.5)]
        counts, _ = np.histogram(vals, bins=np.asarray(edges))
        props = (counts / counts.sum()).tolist()
        sample = rng.choice(vals, size=min(args.n_sample, len(vals)), replace=False).tolist()
        numeric[col] = {
            "bin_edges": edges,
            "bin_proportions": props,
            "baseline_sample": sample,
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "n": int(len(vals)),
        }

    categorical: dict = {}
    for col in ENGINEERED_CATEGORICAL:
        vc = df[col].astype(str).value_counts(normalize=True)
        categorical[col] = {
            "categories": vc.index.tolist(),
            "proportions": {k: float(v) for k, v in vc.items()},
            "n": int(len(df)),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": str(TELCO_CSV),
        "n_rows": len(df),
        "seed": SEED,
        "numeric": numeric,
        "categorical": categorical,
    }, indent=1))
    print(f"Baseline written to {OUT} ({len(df)} rows, "
          f"{len(numeric)} numeric + {len(categorical)} categorical features)")


if __name__ == "__main__":
    main()

"""Drift detector unit tests (seeded RNG throughout)."""

import numpy as np
import pandas as pd
import pytest

from capstone.drift import categorical_drift, check_drift, load_baseline, numeric_drift, psi

TELCO_CSV = "/home/hatch/workspace/portfolio-projects/customer-churn-mlops/data/Telco-Customer-Churn.csv"


def test_psi_zero_on_identical():
    p = np.array([0.5, 0.3, 0.2])
    assert psi(p, p) == 0.0


def test_psi_positive_on_shift():
    # (0.9-0.5)*ln(0.9/0.5) + (0.1-0.5)*ln(0.1/0.5) = 0.8789
    assert psi(np.array([0.5, 0.5]), np.array([0.9, 0.1])) == pytest.approx(0.8789, abs=1e-4)


def test_numeric_drift_quiet_on_baseline_sample():
    baseline = load_baseline()
    rng = np.random.default_rng(0)
    sample = pd.Series(rng.choice(
        np.asarray(baseline["numeric"]["tenure"]["baseline_sample"]), size=2000))
    res = numeric_drift(sample, baseline["numeric"]["tenure"])
    assert res["status"] in ("ok", "warning")


def test_numeric_drift_fires_on_shift():
    baseline = load_baseline()
    shifted = pd.Series(
        np.asarray(baseline["numeric"]["tenure"]["baseline_sample"]) + 60.0)
    res = numeric_drift(shifted, baseline["numeric"]["tenure"])
    assert res["status"] == "drift"
    assert res["psi"] > 0.25


def test_categorical_drift_quiet_and_fires():
    baseline = load_baseline()
    stats = baseline["categorical"]["Contract"]
    cats, props = stats["categories"], stats["proportions"]
    rng = np.random.default_rng(2)
    same = pd.Series(rng.choice(cats, size=3000, p=[props[c] for c in cats]))
    assert categorical_drift(same, stats)["status"] != "drift"
    flipped = pd.Series(["Month-to-month"] * 3000)
    assert categorical_drift(flipped, stats)["status"] == "drift"


def test_check_drift_end_to_end():
    baseline = load_baseline()
    raw = pd.read_csv(TELCO_CSV)
    rng = np.random.default_rng(3)
    sample = raw.iloc[rng.choice(len(raw), size=800, replace=False)]
    control = check_drift(sample, baseline)
    assert control["overall"] != "drift"
    assert control["drifted_features"] == []

    drifted = sample.copy()
    drifted["tenure"] = (drifted["tenure"] + 40).clip(upper=200)
    out = check_drift(drifted, baseline)
    assert out["overall"] == "drift"
    assert "tenure" in out["drifted_features"]

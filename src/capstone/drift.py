"""Feature drift detection: PSI + Kolmogorov-Smirnov vs. a training baseline.

Method
------
- Numeric features: binned with the baseline's quantile edges; Population
  Stability Index (PSI) between baseline and current bin proportions.
  Rule of thumb: PSI < 0.1 no significant change, 0.1-0.25 small change,
  > 0.25 significant change (flag).
- Numeric features (secondary): two-sample KS test vs. a stored baseline
  sample; flag if p < 0.01.
- Categorical features: PSI over category proportions (epsilon smoothing).

Alerting is a stub by design (see README): ``send_alert`` appends to
``monitoring/alerts.log`` and logs. In production this would page PagerDuty /
post to Slack.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from capstone.preprocess import (
    ENGINEERED_CATEGORICAL,
    ENGINEERED_NUMERIC,
    prepare_rows,
)

logger = logging.getLogger("capstone.drift")

PSI_FLAG = 0.25   # significant change
PSI_WARN = 0.10   # small change
KS_P_FLAG = 0.01
_EPS = 1e-6
ROOT = Path(__file__).resolve().parents[2]
ALERT_LOG = ROOT / "monitoring" / "alerts.log"


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """Population Stability Index between two proportion vectors."""
    expected = np.clip(np.asarray(expected, dtype=float), _EPS, None)
    actual = np.clip(np.asarray(actual, dtype=float), _EPS, None)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _binned_proportions(values: pd.Series, edges: list[float]) -> np.ndarray:
    counts, _ = np.histogram(values.to_numpy(dtype=float), bins=np.asarray(edges))
    return counts / max(counts.sum(), 1)


def numeric_drift(values: pd.Series, feature_stats: dict) -> dict:
    """PSI + KS drift check for one numeric feature."""
    edges = feature_stats["bin_edges"]
    base_props = np.asarray(feature_stats["bin_proportions"])
    cur_props = _binned_proportions(values.dropna(), edges)
    psi_val = psi(base_props, cur_props)
    ks_stat, ks_p = stats.ks_2samp(
        np.asarray(feature_stats["baseline_sample"], dtype=float),
        values.dropna().to_numpy(dtype=float),
    )
    status = "ok"
    if psi_val > PSI_FLAG or ks_p < KS_P_FLAG:
        status = "drift"
    elif psi_val > PSI_WARN:
        status = "warning"
    return {
        "type": "numeric", "psi": round(psi_val, 4),
        "ks_stat": round(float(ks_stat), 4), "ks_p": float(ks_p),
        "status": status, "n": int(values.notna().sum()),
    }


def categorical_drift(values: pd.Series, feature_stats: dict) -> dict:
    """PSI drift check for one categorical feature."""
    cats = feature_stats["categories"]
    base_props = np.asarray([feature_stats["proportions"][c] for c in cats])
    cur_counts = values.astype(str).value_counts()
    cur_props = np.asarray([cur_counts.get(c, 0) for c in cats], dtype=float)
    cur_props = cur_props / max(cur_props.sum(), 1)
    psi_val = psi(base_props, cur_props)
    status = "ok"
    if psi_val > PSI_FLAG:
        status = "drift"
    elif psi_val > PSI_WARN:
        status = "warning"
    return {
        "type": "categorical", "psi": round(psi_val, 4),
        "status": status, "n": int(values.notna().sum()),
    }


def check_drift(df_raw: pd.DataFrame, baseline: dict) -> dict:
    """Run all drift checks. ``df_raw`` = raw customer rows (pre-engineering)."""
    df = prepare_rows(df_raw)
    results: dict[str, dict] = {}
    for col in ENGINEERED_NUMERIC:
        if col in baseline["numeric"]:
            results[col] = numeric_drift(df[col], baseline["numeric"][col])
    for col in ENGINEERED_CATEGORICAL:
        if col in baseline["categorical"]:
            results[col] = categorical_drift(df[col], baseline["categorical"][col])
    drifted = [c for c, r in results.items() if r["status"] == "drift"]
    warned = [c for c, r in results.items() if r["status"] == "warning"]
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(df),
        "features": results,
        "drifted_features": drifted,
        "warned_features": warned,
        "overall": "drift" if drifted else ("warning" if warned else "ok"),
    }


def send_alert(message: str) -> None:
    """Alert stub: append to alerts.log + log. Production would page/post here."""
    ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()} ALERT {message}\n"
    with open(ALERT_LOG, "a") as f:
        f.write(line)
    logger.warning("drift alert (stub): %s", message)


def load_baseline(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "monitoring" / "baseline_stats.json"
    return json.loads(path.read_text())

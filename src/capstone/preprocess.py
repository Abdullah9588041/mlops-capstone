"""Vendored preprocessing for serving.

Slim adaptation of ``customer-churn-mlops/src/churn/features.py`` (P5):
only ``clean`` + ``engineer`` are needed at inference time, because the fitted
``ColumnTransformer`` ships inside ``models/champion.pkl``.

Kept in sync with P5 by ``tests/test_preprocess.py`` (parity test on sample rows).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "Churn"
ID_COL = "customerID"

SERVICE_NA_MAP = {
    "MultipleLines": "No phone service",
    "OnlineSecurity": "No internet service",
    "OnlineBackup": "No internet service",
    "DeviceProtection": "No internet service",
    "TechSupport": "No internet service",
    "StreamingTV": "No internet service",
    "StreamingMovies": "No internet service",
}

ADDON_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Numeric columns used by the drift baseline (post-engineering).
ENGINEERED_NUMERIC = [
    "tenure", "MonthlyCharges", "TotalCharges", "avg_monthly",
    "service_count", "is_new", "is_monthly_contract",
    "is_electronic_check", "monthly_x_tenure",
]

# Categorical columns used by the drift baseline (post-engineering).
ENGINEERED_CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "tenure_bucket",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw customer rows: fix TotalCharges blanks, normalize service NAs."""
    df = df.copy()
    df["TotalCharges"] = df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_total = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_total)
    for col, na_val in SERVICE_NA_MAP.items():
        df[col] = df[col].replace(na_val, "No")
    df = df.drop(columns=[ID_COL], errors="ignore")
    if TARGET in df.columns:
        df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features. Expects the output of :func:`clean`."""
    df = df.copy()
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 200],
        labels=["0-12", "12-24", "24-48", "48+"],
    ).astype(str)
    df["avg_monthly"] = df["TotalCharges"] / df["tenure"].replace(0, np.nan)
    df["avg_monthly"] = df["avg_monthly"].fillna(df["MonthlyCharges"])
    df["service_count"] = (df[ADDON_SERVICES] == "Yes").sum(axis=1).astype(int)
    df["is_new"] = (df["tenure"] <= 12).astype(int)
    df["is_monthly_contract"] = (df["Contract"] == "Month-to-month").astype(int)
    df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["monthly_x_tenure"] = df["MonthlyCharges"] * df["tenure"]
    return df


def prepare_rows(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Full inference-time transform: clean -> engineer (no target needed)."""
    return engineer(clean(df_raw))

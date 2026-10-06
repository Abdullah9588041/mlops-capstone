"""Parity: vendored preprocess.py must match P5's features.py on sample rows."""

import sys

import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, "/home/hatch/workspace/portfolio-projects/customer-churn-mlops/src")
from churn import features as p5  # noqa: E402

from capstone import preprocess as cap  # noqa: E402

TELCO_CSV = "/home/hatch/workspace/portfolio-projects/customer-churn-mlops/data/Telco-Customer-Churn.csv"


def test_clean_parity():
    raw = pd.read_csv(TELCO_CSV).head(200)
    assert_frame_equal(cap.clean(raw), p5.clean(raw))


def test_engineer_parity():
    raw = pd.read_csv(TELCO_CSV).head(200)
    assert_frame_equal(cap.engineer(cap.clean(raw)), p5.engineer(p5.clean(raw)))


def test_engineered_columns_match_p5_lists():
    assert cap.ENGINEERED_NUMERIC == p5.ENGINEERED_NUMERIC
    assert cap.ENGINEERED_CATEGORICAL == p5.ENGINEERED_CATEGORICAL

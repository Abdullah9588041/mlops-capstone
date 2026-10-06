"""Fetch the champion model bundle from the sibling P5 project.

Lineage: the model was trained, tuned, calibrated, and MLflow-tracked in
``customer-churn-mlops`` (sibling project). This capstone is about OPERATIONS,
not modeling, so we reuse that artifact verbatim and record its provenance.

Usage:
    python scripts/fetch_model.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P5_MODEL = Path("/home/hatch/workspace/portfolio-projects/customer-churn-mlops/models/champion.pkl")
DEST = ROOT / "models" / "champion.pkl"
LINEAGE = ROOT / "models" / "lineage.json"


def main() -> None:
    if not P5_MODEL.exists():
        sys.exit(f"P5 model not found at {P5_MODEL}. Build customer-churn-mlops first.")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(P5_MODEL, DEST)
    sha = hashlib.sha256(DEST.read_bytes()).hexdigest()
    LINEAGE.write_text(json.dumps({
        "artifact": "models/champion.pkl",
        "source_project": "customer-churn-mlops (P5)",
        "source_path": str(P5_MODEL),
        "sha256": sha,
        "copied_at": datetime.now(timezone.utc).isoformat(),
        "note": "Model trained/tuned/calibrated and MLflow-tracked in P5; "
                "this project reuses it unchanged for serving + operations.",
    }, indent=2))
    print(f"Copied {P5_MODEL} -> {DEST}")
    print(f"sha256: {sha}")
    print(f"Lineage written to {LINEAGE}")


if __name__ == "__main__":
    main()

# MLOps Capstone: Production Operations for a Churn Model

The production-grade companion to the churn model: serving, monitoring, drift
detection, CI/CD, and an incident runbook. This project is about **operations**,
not modeling — the model itself was built, tuned, and calibrated in the sibling
project `customer-churn-mlops` (P5) and is reused here unchanged.

## Architecture

```
                    ┌──────────────────────────────┐
                    │        docker compose        │
                    │                              │
  synthetic traffic │  ┌────────┐    ┌───────────┐  │
 ───────────────────▶ │ monitor│───▶│    api    │  │
  (monitor service) │  │(traffic│    │ (FastAPI) │  │
                    │  │ gen.)  │    │ /predict  │  │
                    │  └────────┘    │ /health   │  │
                    │       │        │ /metrics  │  │
                    │       │        └─────┬─────┘  │
                    │       ▼              │        │
                    │  traffic.jsonl  champion.pkl  │
                    │       │        (1.9MB, from P5)│
                    │       ▼                      │
                    │  ┌────────────┐   ┌────────┐  │
                    │  │ drift check│   │ mlflow │  │
                    │  │ PSI + KS   │   │registry│  │
                    │  └────────────┘   └────────┘  │
                    └──────────────────────────────┘
```

## What is real vs. simulated

This section is deliberate — MLOps portfolios often blur the line. Here it is,
explicitly:

| Component | Status |
|---|---|
| FastAPI serving (`/predict`, `/health`, `/metrics`) | **Real** — tested with TestClient + live load test |
| Model artifact (`models/champion.pkl`) | **Real** — the actual P5 bundle (sha256 in `models/lineage.json`) |
| Prometheus-style metrics, structured JSON logging | **Real** — emitted by the running API |
| Load test (500 requests, p50/p95/p99) | **Real** — measured against the local API (see Results) |
| Drift detector (PSI + KS) + demo | **Real** — runs against real data; demo injects synthetic shifts on purpose |
| `pytest` suite (17 tests), `ruff` lint, GitHub Actions CI | **Real** — green |
| `docker-compose.yml` + Dockerfiles | **Written, YAML-validated only** — no Docker daemon was available in the build environment, so images were never built or run |
| CD workflow push steps | **Guarded** — build + smoke-test always run; push requires `DOCKERHUB_*` secrets |
| MLflow registry promotion commands (`docs/registry.md`) | **Documented, not executed** — no tracking server was started |
| Traffic generator | **Simulated by design** — replays historical Telco rows as stand-in live traffic |
| Alerting (`send_alert`) | **Stub** — writes to `monitoring/alerts.log`; production would page/post |

## Quickstart

**Option A — Docker Compose** (needs a Docker daemon):
```bash
docker compose up --build
# API: http://localhost:8000  |  MLflow UI: http://localhost:5000
```

**Option B — local** (no Docker needed):
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/fetch_model.py        # copies champion.pkl from P5
.venv/bin/python scripts/compute_baseline.py   # builds monitoring/baseline_stats.json
PYTHONPATH=src CHURN_MODEL_PATH=models/champion.pkl \
  .venv/bin/python -m uvicorn capstone.api:app --port 8000
# then: .venv/bin/python scripts/load_test.py
```

## Results

### Load test (measured, single uvicorn worker, localhost)

500 requests, 20 concurrent workers, 0 errors, 18.0 req/s:

| p50 | p95 | p99 | mean | max |
|---|---|---|---|---|
| 1060ms | 1747ms | 2240ms | 1105ms | 3181ms |

Full record: `results/load_test.json`. Honest note: latency is dominated by
per-request pandas preprocessing + a fresh HTTP connection per request in the
test harness. The runbook documents the scaling path (workers, batching, ONNX).

### Drift demo (measured, seed 7, n=3000)

| Arm | Overall | Detail |
|---|---|---|
| Control (same distribution) | **ok** | zero drift flags, zero warnings |
| Injected drift (tenure +15, MonthlyCharges ×1.35, contract flips) | **drift** | 8 features flagged |

Flagged features (PSI): `tenure_bucket` 4.06, `tenure` 3.62, `TotalCharges` 2.46,
`monthly_x_tenure` 2.46, `avg_monthly` 2.22, `MonthlyCharges` 2.14
(`is_new`, `is_monthly_contract` flagged by the KS test). Full record:
`results/demo_drift.json`. The demo also wrote one entry to
`monitoring/alerts.log` via the alert stub.

### Tests & lint

`pytest`: **17 passed**. `ruff check`: clean. CI runs both on every push.

### Model lineage

- Source: `customer-churn-mlops` (P5) — `models/champion.pkl`
- Algorithm: `CalibratedClassifierCV` over tuned `XGBClassifier`
- Threshold 0.2; test ROC-AUC 0.836, recall 0.864, precision 0.474
- sha256: `a74b66c66adc2423ff437839d4ae18748fd76d03c1c139ab8a4641b9dd542d77`
- Copied verbatim by `scripts/fetch_model.py`; provenance in `models/lineage.json`

## Project structure

```
├── src/capstone/
│   ├── api.py          # FastAPI: /predict, /health, /metrics (Prometheus-style)
│   ├── drift.py        # PSI + KS drift detection vs baseline
│   └── preprocess.py   # vendored P5 clean/engineer (parity-tested)
├── models/
│   ├── champion.pkl        # the real P5 bundle (1.9MB, committed deliberately)
│   └── lineage.json        # sha256 + provenance
├── monitoring/
│   ├── baseline_stats.json # training reference distributions (7043 rows)
│   ├── demo_drift.py       # control vs injected-drift demonstration
│   ├── traffic.py          # synthetic traffic generator (compose `monitor`)
│   ├── Dockerfile / Dockerfile.mlflow
├── scripts/
│   ├── fetch_model.py      # copy bundle from P5 + record lineage
│   ├── compute_baseline.py # build the drift baseline from Telco data
│   └── load_test.py        # concurrent load test -> results/load_test.json
├── docs/
│   ├── registry.md         # MLflow staging->production promotion (documented)
│   └── runbook.md          # incident runbook: symptoms -> diagnosis -> rollback
├── .github/workflows/
│   ├── ci.yml              # pytest + ruff
│   └── cd.yml              # build + smoke-test; push guarded by secrets
├── tests/                  # 17 tests: API contract, drift, preprocess parity, compose
├── docker-compose.yml      # api + mlflow + monitor (YAML-validated only)
├── Dockerfile              # API image
└── results/
    ├── load_test.json      # real measured latencies (committed)
    └── demo_drift.json     # real drift demo outcome (committed)
```

## Reproducibility

- Pinned dependencies (`requirements.txt`), Python 3.12
- Fixed seeds: baseline (42), drift demo (7), load test (21), traffic gen (11)
- `PYTHONPATH=src` for scripts/tests (also set in `pyproject.toml` for pytest)

## Limitations & future work

- Docker images were never built here (no daemon) — first real validation step
  is `docker compose up --build` on a machine with Docker.
- Single-worker API; no autoscaling, no canary deploys, no K8s manifests.
- Drift alerting is a log stub, not a paging integration.
- No online feature store; the traffic generator replays history rather than
  producing truly live features.
- Model monitoring covers input drift, not label/concept drift (needs labeled
  outcomes with delay — the hard part, honestly noted).

## References

- P5 sibling project: `customer-churn-mlops` (model training, tuning, calibration)
- MLflow model registry docs; Prometheus exposition format
- Baeldung-style PSI rule of thumb: <0.1 none, 0.1–0.25 small, >0.25 significant

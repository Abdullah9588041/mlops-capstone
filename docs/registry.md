# Model registry workflow (MLflow)

The champion model was trained, tuned, calibrated, and **experiment-tracked in
MLflow in the P5 project** (`customer-churn-mlops/mlruns.db`). This capstone
reuses that artifact for serving. The registry promotion workflow below is the
production pattern for moving a model from experimentation to serving.

## Status: documented, not executed here

The commands below were **not run** in this environment (no MLflow server was
started during the build; see README "What is real vs simulated"). They are the
exact steps an operator would run against the compose `mlflow` service.

## 1. Register the champion

```bash
# Point at the tracking server from docker-compose.yml
export MLFLOW_TRACKING_URI=http://localhost:5000

python - <<'EOF'
import mlflow, pickle

with open("models/champion.pkl", "rb") as f:
    bundle = pickle.load(f)

# Log the bundle as an MLflow model with its metadata
with mlflow.start_run(run_name="champion-xgboost-registration"):
    mlflow.log_param("model_name", bundle["model_name"])
    mlflow.log_param("threshold", bundle["threshold"])
    mlflow.log_metric("test_roc_auc", bundle["test_metrics"]["roc_auc"])
    mlflow.log_metric("test_recall", bundle["test_metrics"]["recall"])
    mlflow.sklearn.log_model(
        bundle["model"],
        artifact_path="model",
        registered_model_name="churn-xgboost-champion",
    )
print("registered: churn-xgboost-champion")
EOF
```

## 2. Promote Staging -> Production

```bash
python - <<'EOF'
from mlflow.tracking import MlflowClient

client = MlflowClient()
# After offline validation of version N in Staging:
client.transition_model_version_stage(
    name="churn-xgboost-champion",
    version="1",
    stage="Production",
    archive_existing_versions=True,   # previous Production -> Archived
)
print("version 1 is now Production")
EOF
```

Promotion checklist (each box must be true before promoting):
- [ ] Champion metrics reproduced from `models/champion.pkl` (ROC-AUC 0.836, recall 0.864 @ threshold 0.2)
- [ ] `pytest` green on the serving repo at the pinned commit
- [ ] Load test meets the latency SLO (see `results/load_test.json`)
- [ ] Drift baseline (`monitoring/baseline_stats.json`) regenerated for this version
- [ ] Rollback bundle (previous Production version) retained in the registry

## 3. Rollback

```bash
python - <<'EOF'
from mlflow.tracking import MlflowClient

client = MlflowClient()
# Re-promote the previous version; the bad version goes to Archived.
client.transition_model_version_stage(
    name="churn-xgboost-champion", version="1",
    stage="Production", archive_existing_versions=True,
)
EOF
```

Then redeploy the API pointing at the rolled-back artifact and follow
`docs/runbook.md`.

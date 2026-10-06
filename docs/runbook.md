# Incident runbook: churn prediction API

How to think like an operator when this service misbehaves. Severity levels:
**SEV-3** (degraded) -> investigate in hours; **SEV-2** (broken) -> drop
everything; **SEV-1** (wrong predictions at scale) -> rollback first, ask later.

## 1. "API is down / 5xx" (SEV-2)

**Symptoms:** `/health` fails; compose shows `api` restarting; alerts on
`capstone_errors_total`.

1. `docker compose logs api --tail 100` — read the actual traceback.
2. Common causes and checks:
   - `Model bundle not found` (503): the image was built without `models/`.
     Rebuild: `docker compose build api`.
   - OOM on startup: the XGBoost bundle needs ~300MB RSS; check
     `docker stats`. Raise the container memory limit.
   - Port conflict: something else on 8000 (`lsof -i :8000`).
3. If the container is crash-looping and the cause is not obvious in 15
   minutes: roll back to the last good image tag
   (`churn-capstone-api:<previous-sha>`) and investigate offline.

## 2. "Latency p95 blowing past SLO" (SEV-3)

**Symptoms:** `capstone_request_latency_seconds` histogram shifts right;
load-test baseline is in `results/load_test.json` (p50 ~1060ms, p95 ~1750ms
single-worker — yes, that is the honest baseline; see "known limits").

1. Check `docker stats`: CPU saturated? The model does full pandas
   preprocessing per request on one worker.
2. Short-term relief: raise `--workers` to 2-4 (memory cost ~300MB each).
3. Real fix (future work): batch inference endpoint, pre-warmed connection
   pool, or ONNX export. Do NOT "fix" by raising the timeout.

## 3. "Drift detector firing" (SEV-3, can become SEV-1)

**Symptoms:** `monitoring/alerts.log` entries; `demo_drift.py`-style PSI > 0.25
on live traffic.

1. Run the detector manually on the recent traffic window:
   `PYTHONPATH=src python monitoring/demo_drift.py` is the demo; for live
   data, point `check_drift` at the logged requests.
2. Diagnose WHICH features drifted:
   - Upstream schema change (new category values)? -> fix the producer,
     update the baseline.
   - Genuine population shift (e.g., new pricing plans)? -> this is the
     interesting case: schedule a retrain evaluation in P5, compare champion
     vs challenger on recent labeled data.
3. Do NOT silently retrain on drifted data and redeploy. Drift is a signal
   to investigate, not an auto-retrain trigger (auto-retraining on a broken
   upstream feed bakes the breakage into the model).

## 4. "Predictions look wrong" (SEV-1)

**Symptoms:** prediction distribution in `/metrics` flips suddenly
(`capstone_predictions_total` churn share jumps), or business reports
nonsense decisions.

1. **Rollback first:** redeploy the previous Production model version from
   the MLflow registry (`docs/registry.md` section 3). Target: < 15 minutes.
2. Then investigate: compare `/metrics` distributions before/after, check the
   drift report, verify the bundle sha256 matches `models/lineage.json`.

## 5. Rollback procedure (any SEV)

1. Identify the last good version: `models/lineage.json` (sha256) or the
   MLflow registry's previous Production version.
2. `docker compose build api` is NOT needed — retag:
   `docker tag churn-capstone-api:<good-sha> churn-capstone-api:rollback`
   and update the compose image, or re-run the CD workflow at the good commit.
3. `docker compose up -d api`; verify `/health` model metadata matches the
   expected `trained_at`.
4. Announce in the incident channel: what broke, what was rolled back to,
   link to the postmortem doc.

## Contacts / ownership

| Role | Responsibility |
|---|---|
| On-call engineer | Triage, rollback, incident channel |
| Model owner (me) | Drift diagnosis, retrain decisions |
| Data engineering | Upstream feed fixes |

Postmortem template: timeline, root cause, what went well, action items with
owners and dates. No-blame.

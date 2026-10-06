"""docker-compose.yml validity (parsed, not executed — no daemon here)."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text())


def test_compose_parses_and_has_services():
    d = _compose()
    assert set(d["services"]) == {"api", "mlflow", "monitor"}


def test_api_service():
    api = _compose()["services"]["api"]
    assert api["ports"] == ["8000:8000"]
    assert "CHURN_MODEL_PATH" in api["environment"]
    assert "healthcheck" in api


def test_monitor_depends_on_healthy_api():
    mon = _compose()["services"]["monitor"]
    assert mon["depends_on"]["api"]["condition"] == "service_healthy"


def test_dockerfiles_exist():
    assert (ROOT / "Dockerfile").exists()
    assert (ROOT / "monitoring" / "Dockerfile").exists()
    assert (ROOT / "monitoring" / "Dockerfile.mlflow").exists()

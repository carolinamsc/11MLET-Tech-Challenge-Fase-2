"""Shared settings, file paths, parameter loading and logging setup."""

import json
import logging.config
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

PARAMS_PATH = Path("params.yaml")
LOGGING_CONFIG_PATH = Path("configs/logging.yaml")
INTERIM_DIR = Path("data/interim")
MODEL_PATH = Path("models/purchase_intent.joblib")
BEST_RUN_PATH = Path("models/best_run.json")
METRICS_PATH = Path("reports/metrics.json")
REGISTRY_REPORT_PATH = Path("reports/registry.json")


class Settings(BaseSettings):
    """Runtime configuration read from environment variables or .env."""

    mlflow_tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "11mlet-purchase-intent"
    registered_model_name: str = "11mlet-purchase-intent"
    champion_alias: str = "champion"

    model_config = SettingsConfigDict(env_prefix="CHALLENGE_", env_file=".env", extra="ignore")


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    """Load pipeline parameters from the project YAML file."""
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def setup_logging(path: Path = LOGGING_CONFIG_PATH) -> None:
    """Configure logging from the YAML file, falling back to basic INFO logging."""
    if not path.exists():
        logging.basicConfig(level=logging.INFO)
        return
    with path.open(encoding="utf-8") as stream:
        logging.config.dictConfig(yaml.safe_load(stream))


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Write a dictionary as indented UTF-8 JSON, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")


def read_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON file into a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))

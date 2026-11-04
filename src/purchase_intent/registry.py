"""Register the selected model in the MLflow Model Registry and promote it."""

import logging

from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from purchase_intent.config import (
    BEST_RUN_PATH,
    METRICS_PATH,
    REGISTRY_REPORT_PATH,
    Settings,
    read_json,
    setup_logging,
    write_json,
)

logger = logging.getLogger(__name__)


def ensure_registered_model(client: MlflowClient, name: str) -> None:
    """Create the registered model entry when it does not exist yet."""
    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(name, description="Propensão de compra (UCI)")


def register_run(client: MlflowClient, settings: Settings, run_id: str) -> ModelVersion:
    """Create a model version from the run artifact and point the champion alias at it."""
    name = settings.registered_model_name
    ensure_registered_model(client, name)
    version = client.create_model_version(name, f"runs:/{run_id}/model", run_id)
    client.set_registered_model_alias(name, settings.champion_alias, version.version)
    return version


def tag_version(client: MlflowClient, version: ModelVersion, metrics: dict[str, float]) -> None:
    """Store the held-out test metrics on the registered model version."""
    for key in ("f1", "roc_auc", "recall", "precision"):
        client.set_model_version_tag(version.name, version.version, f"test_{key}", metrics[key])


def main() -> None:
    """Register the model chosen by the training stage as the champion."""
    setup_logging()
    settings = Settings()
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    best_run = read_json(BEST_RUN_PATH)
    version = register_run(client, settings, best_run["run_id"])
    tag_version(client, version, read_json(METRICS_PATH))
    report = {
        "registered_model": version.name,
        "version": int(version.version),
        "alias": settings.champion_alias,
        "model_type": best_run["model_type"],
    }
    write_json(report, REGISTRY_REPORT_PATH)
    logger.info("Modelo %s v%s promovido a %s", version.name, version.version, report["alias"])


if __name__ == "__main__":
    main()

"""Evaluate the selected model on the held-out test partition."""

import logging

import joblib
import mlflow
import pandas as pd

from purchase_intent.config import (
    BEST_RUN_PATH,
    INTERIM_DIR,
    METRICS_PATH,
    MODEL_PATH,
    Settings,
    load_params,
    read_json,
    setup_logging,
    write_json,
)
from purchase_intent.training.train import evaluate_partition

logger = logging.getLogger(__name__)


def log_test_metrics(run_id: str, metrics: dict[str, float], settings: Settings) -> None:
    """Attach test metrics to the MLflow run that produced the selected model."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("selected", "true")
        mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})


def main() -> None:
    """Compute test metrics, log them to MLflow and save them for DVC."""
    setup_logging()
    target = load_params()["data"]["target"]
    test = pd.read_csv(INTERIM_DIR / "test.csv")
    best_run = read_json(BEST_RUN_PATH)
    metrics = evaluate_partition(joblib.load(MODEL_PATH), test, target)
    log_test_metrics(best_run["run_id"], metrics, Settings())
    report = {"model_type": best_run["model_type"], **metrics, "test_rows": len(test)}
    write_json(report, METRICS_PATH)
    logger.info("Test F1=%.4f; métricas salvas em %s", metrics["f1"], METRICS_PATH)


if __name__ == "__main__":
    main()

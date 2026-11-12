"""Train candidate classifiers, track each one in MLflow and keep the best."""

import logging
from dataclasses import dataclass
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from purchase_intent.config import (
    BEST_RUN_PATH,
    INTERIM_DIR,
    MODEL_PATH,
    Settings,
    load_params,
    setup_logging,
    write_json,
)
from purchase_intent.data.prepare import read_sessions
from purchase_intent.metrics import calculate_metrics
from purchase_intent.models.pipeline import build_pipeline

logger = logging.getLogger(__name__)
SELECTION_METRIC = "f1"


@dataclass(frozen=True)
class TrainingResult:
    """Outcome of training one candidate model."""

    name: str
    run_id: str
    metrics: dict[str, float]
    model: Pipeline


def evaluate_partition(model: Pipeline, frame: pd.DataFrame, target: str) -> dict[str, float]:
    """Calculate classification metrics for a labeled partition."""
    features = frame.drop(columns=[target])
    return calculate_metrics(
        frame[target].astype(int).to_numpy(),
        model.predict(features),
        model.predict_proba(features)[:, 1],
    )


def build_input_example(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return a small float-safe input sample for the MLflow model signature."""
    example = frame.drop(columns=[target]).head(5).copy()
    integer_columns = example.select_dtypes(include=["integer"]).columns
    example[integer_columns] = example[integer_columns].astype("float64")
    return example


def train_candidate(
    name: str, parameters: dict[str, Any], data: dict[str, pd.DataFrame], target: str, seed: int
) -> TrainingResult:
    """Fit one candidate and log its parameters, metrics and model in an MLflow run."""
    model = build_pipeline(data["train"], target, name, parameters, seed)
    model.fit(data["train"].drop(columns=[target]), data["train"][target].astype(int))
    metrics = evaluate_partition(model, data["validation"], target)
    example = build_input_example(data["validation"], target)
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"run_type": "training", "model_type": name})
        mlflow.log_params({**parameters, "model_type": name, "random_seed": seed})
        mlflow.log_metrics({f"validation_{key}": value for key, value in metrics.items()})
        signature = infer_signature(example, model.predict(example))
        mlflow.sklearn.log_model(
            model, artifact_path="model", signature=signature, input_example=example
        )
    logger.info("%s validation F1=%.4f", name, metrics[SELECTION_METRIC])
    return TrainingResult(name, run.info.run_id, metrics, model)


def save_best(results: list[TrainingResult]) -> TrainingResult:
    """Persist the candidate with the highest validation selection metric."""
    best = max(results, key=lambda result: result.metrics[SELECTION_METRIC])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.model, MODEL_PATH)
    summary = {
        "model_type": best.name,
        "run_id": best.run_id,
        "validation": best.metrics,
        "candidates": {result.name: result.metrics for result in results},
    }
    write_json(summary, BEST_RUN_PATH)
    return best


def main() -> None:
    """Train every configured candidate and keep the best one for evaluation."""
    setup_logging()
    settings = Settings()
    params = load_params()
    seed = params["experiment"]["random_seed"]
    data = {name: read_sessions(INTERIM_DIR / f"{name}.csv") for name in ("train", "validation")}
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)
    results = [
        train_candidate(name, candidate, data, params["data"]["target"], seed)
        for name, candidate in params["model"]["candidates"].items()
    ]
    best = save_best(results)
    logger.info("Melhor modelo: %s (run %s) salvo em %s", best.name, best.run_id, MODEL_PATH)


if __name__ == "__main__":
    main()

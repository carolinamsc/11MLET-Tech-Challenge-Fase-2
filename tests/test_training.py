"""Integration test: train candidates with a throwaway MLflow backend."""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from purchase_intent.training import train


@pytest.fixture
def partitions() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "PageValues": rng.exponential(5, 200),
            "VisitorType": rng.choice(["New_Visitor", "Returning_Visitor"], 200),
        }
    )
    frame["Revenue"] = (frame["PageValues"] > 5).astype(int)
    return {"train": frame.iloc[:150], "validation": frame.iloc[150:]}


def test_candidates_are_tracked_and_best_is_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, partitions: dict[str, pd.DataFrame]
) -> None:
    monkeypatch.setattr(train, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(train, "BEST_RUN_PATH", tmp_path / "best_run.json")
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.create_experiment("test", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment("test")
    candidates = {
        "logistic_regression": {"C": 1.0, "max_iter": 200},
        "random_forest": {"n_estimators": 10, "max_depth": 3},
    }
    results = [
        train.train_candidate(name, params, partitions, "Revenue", seed=42)
        for name, params in candidates.items()
    ]
    best = train.save_best(results)

    runs = mlflow.search_runs(experiment_names=["test"])
    assert len(runs) == 2
    assert {"validation_f1", "validation_roc_auc"} <= {
        column.removeprefix("metrics.") for column in runs.columns
    }
    assert best.metrics["f1"] == max(result.metrics["f1"] for result in results)
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "best_run.json").exists()

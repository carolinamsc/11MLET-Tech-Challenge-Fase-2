import numpy as np

from purchase_intent.metrics import calculate_metrics


def test_metrics_report_perfect_predictions() -> None:
    result = calculate_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
    )
    assert result == {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "roc_auc": 1.0,
    }

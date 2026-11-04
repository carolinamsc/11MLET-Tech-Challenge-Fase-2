import pandas as pd
import pytest

from purchase_intent.models.pipeline import build_pipeline


def test_pipeline_fits_numeric_and_categorical_features() -> None:
    frame = pd.DataFrame(
        {
            "pages": [1, 2, 3, 4, 5, 6],
            "visitor": ["new", "returning", "new", "new", "returning", "returning"],
            "Revenue": [0, 0, 0, 1, 1, 1],
        }
    )
    params = {"C": 1.0, "max_iter": 100, "class_weight": "balanced"}
    model = build_pipeline(frame, "Revenue", "logistic_regression", params, seed=42)
    model.fit(frame.drop(columns="Revenue"), frame["Revenue"])
    probabilities = model.predict_proba(frame.drop(columns="Revenue"))
    assert probabilities.shape == (6, 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_random_forest_candidate_is_reproducible() -> None:
    frame = pd.DataFrame({"pages": range(20), "Revenue": [0, 1] * 10})
    params = {"n_estimators": 10, "max_depth": 3}
    predictions = []
    for _ in range(2):
        model = build_pipeline(frame, "Revenue", "random_forest", params, seed=7)
        model.fit(frame.drop(columns="Revenue"), frame["Revenue"])
        predictions.append(model.predict_proba(frame.drop(columns="Revenue"))[:, 1].tolist())
    assert predictions[0] == predictions[1]


def test_unknown_model_is_rejected() -> None:
    frame = pd.DataFrame({"pages": [1, 2], "Revenue": [0, 1]})
    with pytest.raises(ValueError, match="Unsupported model"):
        build_pipeline(frame, "Revenue", "svm", {}, seed=1)

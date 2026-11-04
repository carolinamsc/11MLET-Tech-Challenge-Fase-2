import pandas as pd

from purchase_intent.models.pipeline import build_pipeline
from purchase_intent.predict import score_sessions


def test_score_sessions_adds_probability_and_label() -> None:
    frame = pd.DataFrame({"pages": [1, 2, 3, 8, 9, 10], "Revenue": [0, 0, 0, 1, 1, 1]})
    model = build_pipeline(frame, "Revenue", "logistic_regression", {"max_iter": 100}, seed=1)
    model.fit(frame[["pages"]], frame["Revenue"])
    scored = score_sessions(model, frame, "Revenue")
    assert scored["purchase_probability"].between(0, 1).all()
    assert set(scored["purchase_prediction"]) <= {0, 1}
    assert len(scored) == len(frame)

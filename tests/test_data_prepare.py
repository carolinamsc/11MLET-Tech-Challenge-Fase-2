from pathlib import Path

import pandas as pd
import pytest

from purchase_intent.data.prepare import encode_target, read_sessions, split_data
from purchase_intent.models.pipeline import build_preprocessor


def test_read_sessions_keeps_coded_categoricals_as_text(tmp_path: Path) -> None:
    path = tmp_path / "sessions.csv"
    path.write_text("PageValues,Browser,Weekend\n1.5,2,True\n0.0,10,False\n", encoding="utf-8")
    frame = read_sessions(path)
    assert frame["Browser"].tolist() == ["2", "10"]
    assert frame["Weekend"].tolist() == ["True", "False"]
    _, _, numeric = build_preprocessor(frame).transformers[0]
    _, _, categorical = build_preprocessor(frame).transformers[1]
    assert numeric == ["PageValues"]
    assert categorical == ["Browser", "Weekend"]


def test_encode_target_accepts_boolean_strings() -> None:
    result = encode_target(pd.Series(["TRUE", "FALSE", "True"]))
    assert result.tolist() == [1, 0, 1]


def test_encode_target_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Revenue"):
        encode_target(pd.Series(["yes", "no"]))


def test_split_data_is_disjoint_and_stratified() -> None:
    frame = pd.DataFrame({"feature": range(100), "Revenue": [True, False] * 50})
    train, validation, test = split_data(frame, "Revenue", 0.2, 0.2, seed=42)
    assert len(train) == 60
    assert len(validation) == 20
    assert len(test) == 20
    assert set(train.index).isdisjoint(validation.index)
    assert set(train.index).isdisjoint(test.index)

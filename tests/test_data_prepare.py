import pandas as pd
import pytest

from purchase_intent.data.prepare import encode_target, split_data


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

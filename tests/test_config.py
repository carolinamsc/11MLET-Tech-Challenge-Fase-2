from pathlib import Path

from purchase_intent.config import Settings, load_params, read_json, write_json


def test_params_define_candidates_and_seed() -> None:
    params = load_params()
    assert params["experiment"]["random_seed"] == 42
    assert {"logistic_regression", "random_forest"} <= set(params["model"]["candidates"])


def test_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "file.json"
    write_json({"f1": 0.5}, path)
    assert read_json(path) == {"f1": 0.5}


def test_settings_read_prefixed_environment(monkeypatch) -> None:
    monkeypatch.setenv("CHALLENGE_EXPERIMENT_NAME", "exp-test")
    assert Settings(_env_file=None).experiment_name == "exp-test"

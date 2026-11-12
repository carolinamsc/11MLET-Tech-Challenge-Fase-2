"""Score sessions with the champion model loaded from the MLflow Model Registry.

Usage: ``python -m purchase_intent.predict data/interim/test.csv reports/predictions.csv``
"""

import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline

from purchase_intent.config import Settings, load_params, setup_logging
from purchase_intent.data.prepare import read_sessions

logger = logging.getLogger(__name__)


def load_champion(settings: Settings) -> Pipeline:
    """Load the model version currently pointed to by the champion alias."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    uri = f"models:/{settings.registered_model_name}@{settings.champion_alias}"
    logger.info("Carregando %s", uri)
    return mlflow.sklearn.load_model(uri)


def score_sessions(model: Pipeline, sessions: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return the sessions with purchase probability and predicted label columns."""
    features = sessions.drop(columns=[target], errors="ignore")
    scored = sessions.copy()
    scored["purchase_probability"] = model.predict_proba(features)[:, 1]
    scored["purchase_prediction"] = model.predict(features)
    return scored


def parse_args() -> argparse.Namespace:
    """Read input and output CSV paths from the command line."""
    parser = argparse.ArgumentParser(description="Prevê propensão de compra por sessão.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    return parser.parse_args()


def main() -> None:
    """Score a CSV of sessions with the champion model and save the result."""
    setup_logging()
    args = parse_args()
    model = load_champion(Settings())
    scored = score_sessions(model, read_sessions(args.input_csv), load_params()["data"]["target"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_csv, index=False)
    logger.info("%d sessões pontuadas em %s", len(scored), args.output_csv)


if __name__ == "__main__":
    main()

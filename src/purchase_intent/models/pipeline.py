"""Build preprocessing and classification pipelines for the candidate models."""

from collections.abc import Callable
from typing import Any

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_DTYPES = ["object", "category", "bool", "string"]

CLASSIFIERS: dict[str, Callable[..., ClassifierMixin]] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create imputation, scaling and one-hot encoding steps from feature dtypes."""
    categorical = features.select_dtypes(include=CATEGORICAL_DTYPES).columns.tolist()
    numeric = features.select_dtypes(exclude=CATEGORICAL_DTYPES).columns.tolist()
    numeric_steps = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_steps = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_steps, numeric), ("categorical", categorical_steps, categorical)]
    )


def build_classifier(name: str, parameters: dict[str, Any], seed: int) -> ClassifierMixin:
    """Instantiate a supported Scikit-Learn classifier with a fixed random seed."""
    if name not in CLASSIFIERS:
        raise ValueError(f"Unsupported model '{name}'. Options: {sorted(CLASSIFIERS)}")
    return CLASSIFIERS[name](**parameters, random_state=seed)


def build_pipeline(
    frame: pd.DataFrame, target: str, name: str, parameters: dict[str, Any], seed: int
) -> Pipeline:
    """Create a leakage-safe preprocessing and classifier pipeline."""
    preprocessing = build_preprocessor(frame.drop(columns=[target]))
    classifier = build_classifier(name, parameters, seed)
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])

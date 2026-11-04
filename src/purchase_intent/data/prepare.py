"""Prepare reproducible train, validation, and test partitions."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from purchase_intent.config import INTERIM_DIR, load_params

CATEGORICAL_COLUMNS = (
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
)


def encode_target(values: pd.Series) -> pd.Series:
    """Convert UCI boolean-like target values to integer labels."""
    if pd.api.types.is_bool_dtype(values):
        return values.astype(int)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0}
    encoded = normalized.map(mapping)
    if encoded.isna().any():
        raise ValueError("Revenue must contain only boolean or 0/1 values.")
    return encoded.astype(int)


def split_data(
    frame: pd.DataFrame, target: str, test_size: float, validation_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows into stratified train, validation, and test data."""
    if target not in frame.columns:
        raise ValueError(f"Target column '{target}' is missing.")
    frame = frame.dropna(subset=[target]).copy()
    frame[target] = encode_target(frame[target])
    for column in CATEGORICAL_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype("string")
    train_validation, test = train_test_split(
        frame, test_size=test_size, random_state=seed, stratify=frame[target]
    )
    relative_validation_size = validation_size / (1 - test_size)
    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation_size,
        random_state=seed,
        stratify=train_validation[target],
    )
    return train, validation, test


def main() -> None:
    """Write the configured dataset partitions."""
    params = load_params()
    raw_path = Path(params["data"]["raw_path"])
    frame = pd.read_csv(raw_path)
    partitions = split_data(
        frame,
        params["data"]["target"],
        params["data"]["test_size"],
        params["data"]["validation_size"],
        params["experiment"]["random_seed"],
    )
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    for name, partition in zip(("train", "validation", "test"), partitions, strict=True):
        partition.to_csv(INTERIM_DIR / f"{name}.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    main()

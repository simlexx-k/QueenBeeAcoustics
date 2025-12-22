#!/usr/bin/env python3
"""
Train a second-stage model that ingests the unified hive/weather/acoustic dataset
and predicts whether a hive is experiencing a stress event.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "content" / "main-data"
DATASET_PATH = DATA_DIR / "hive_weather_acoustic.parquet"
MODEL_PATH = DATA_DIR / "hive_unified_model.pkl"
METRICS_PATH = DATA_DIR / "hive_unified_model_metrics.json"


def _load_dataset() -> pd.DataFrame:
    if DATASET_PATH.suffix == ".parquet" and DATASET_PATH.exists():
        df = pd.read_parquet(DATASET_PATH)
    elif DATASET_PATH.with_suffix(".csv").exists():
        df = pd.read_csv(DATASET_PATH.with_suffix(".csv"))
    else:
        raise FileNotFoundError(
            f"Unified dataset not found. Run scripts/merge_hive_acoustic.py first ({DATASET_PATH})."
        )

    df["date"] = pd.to_datetime(df["date"])
    df["stress_event"] = df["stress_event"].fillna("none")
    df["stress_target"] = (df["stress_event"] != "none").astype(int)
    return df


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    exclude_cols = {
        "date",
        "hive_id",
        "note",
        "stress_event",
        "stress_target",
        "created_at",
        "recorded_at",
    }
    numeric_cols = df.select_dtypes(include=["number"]).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    if not feature_cols:
        raise ValueError("No numeric features found for training.")

    X = df[feature_cols].copy()
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median())
    y = df["stress_target"]
    return X, y


def main() -> None:
    df = _load_dataset()
    X, y = _prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(X.columns)}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(report, indent=2))

    print(f"[train] Saved model -> {MODEL_PATH}")
    print(f"[train] Metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()

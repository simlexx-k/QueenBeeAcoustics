#!/usr/bin/env python3
"""
Build a unified hive dataset that blends hive logs, weather/NDVI data,
and acoustic predictions captured by the FastAPI service.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "content" / "main-data"
HIVE_LOG_PATH = DATA_DIR / "hive_logs_2008_2025.csv"
WEATHER_NDVI_PATH = DATA_DIR / "makueni_weather_ndvi_2008_2025.csv"
WEATHER_FALLBACK_PATH = DATA_DIR / "makueni_weather_2008_2025.csv"
ACOUSTIC_LOG_PATH = DATA_DIR / "acoustic_predictions.csv"
OUTPUT_PATH = DATA_DIR / "hive_weather_acoustic.parquet"


def _load_hive_logs() -> pd.DataFrame:
    if not HIVE_LOG_PATH.exists():
        raise FileNotFoundError(f"Hive log not found at {HIVE_LOG_PATH}")
    hive = pd.read_csv(HIVE_LOG_PATH)
    hive["date"] = pd.to_datetime(hive["date"]).dt.date
    return hive


def _load_weather() -> pd.DataFrame:
    if WEATHER_NDVI_PATH.exists():
        path = WEATHER_NDVI_PATH
    elif WEATHER_FALLBACK_PATH.exists():
        path = WEATHER_FALLBACK_PATH
    else:
        raise FileNotFoundError(
            "No weather file found. Expected makueni_weather_ndvi_2008_2025.csv "
            "or makueni_weather_2008_2025.csv in content/main-data."
        )
    weather = pd.read_csv(path)
    weather["date"] = pd.to_datetime(weather["date"]).dt.date
    return weather


def _expand_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    if "probabilities_json" not in df.columns:
        return df
    prob_records = df["probabilities_json"].fillna("{}").apply(json.loads)
    prob_df = pd.json_normalize(prob_records)
    prob_df.columns = [f"acoustic_prob_{c}" for c in prob_df.columns]
    return pd.concat([df.drop(columns=["probabilities_json"]), prob_df], axis=1)


def _load_acoustic() -> pd.DataFrame | None:
    if not ACOUSTIC_LOG_PATH.exists():
        print(f"[merge] Acoustic log {ACOUSTIC_LOG_PATH} not found. Skipping acoustic merge.")
        return None
    acoustic = pd.read_csv(ACOUSTIC_LOG_PATH)
    if acoustic.empty:
        print("[merge] Acoustic log exists but is empty.")
        return None

    for col in ("recorded_at", "created_at"):
        if col in acoustic.columns:
            acoustic[col] = pd.to_datetime(acoustic[col], errors="coerce")

    acoustic["date"] = (
        acoustic["recorded_at"]
        .fillna(acoustic["created_at"])
        .dt.tz_localize(None)
        .dt.date
    )
    acoustic = acoustic.dropna(subset=["date"])
    acoustic = _expand_probabilities(acoustic)

    # Keep the latest entry per hive/date.
    acoustic.sort_values(by=["date", "created_at"], inplace=True)
    latest = acoustic.groupby(["hive_id", "date"], as_index=False).tail(1)
    return latest


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    hive_df = _load_hive_logs()
    weather_df = _load_weather()

    merged = pd.merge(hive_df, weather_df, on="date", how="left", suffixes=("", "_weather"))

    acoustic_df = _load_acoustic()
    if acoustic_df is not None:
        merged = pd.merge(
            merged,
            acoustic_df.drop(columns=["audio_filename"]),
            how="left",
            on=["hive_id", "date"],
        )

    merged.to_parquet(OUTPUT_PATH, index=False)
    print(f"[merge] Saved unified dataset -> {OUTPUT_PATH} ({merged.shape[0]} rows, {merged.shape[1]} columns)")


if __name__ == "__main__":
    main()

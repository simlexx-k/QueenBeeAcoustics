import csv
import io
import json
import os
import tempfile
from datetime import datetime
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from app.wards import WARD_LOOKUP, WARD_REGISTRY

matplotlib.use("Agg")

APP_ROOT = Path(__file__).resolve().parent.parent
CONTENT_ROOT = APP_ROOT / "content"
BEES_ROOT = CONTENT_ROOT / "beehive_audio" / "7" / "Dataset"
DATA_ROOT = CONTENT_ROOT / "main-data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_PATH = APP_ROOT / "models" / "queenbee_final_tuned_model.keras"
SPECTROGRAM_DIR = BEES_ROOT / "Spectograms"
ACOUSTIC_LOG_PATH = DATA_ROOT / "acoustic_predictions.csv"
UNIFIED_MODEL_PATH = DATA_ROOT / "hive_unified_model.pkl"
UNIFIED_DATASET_PARQUET = DATA_ROOT / "hive_weather_acoustic.parquet"
UNIFIED_DATASET_CSV = DATA_ROOT / "hive_weather_acoustic.csv"
KAGGLE_EXPORT_ROOT = APP_ROOT / "kaggle_exports" / "beeunity"
KAGGLE_DATA_ROOT = KAGGLE_EXPORT_ROOT / "content" / "main-data"
WEATHER_NDVI_FILE = "makueni_weather_ndvi_2008_2025.csv"
YIELD_FORECAST_FILE = "makueni_climate_yield_forecast.csv"
ACOUSTIC_LOG_FIELDS = [
    "created_at",
    "recorded_at",
    "hive_id",
    "ward_id",
    "latitude",
    "longitude",
    "audio_filename",
    "predicted_label",
    "confidence",
    "probabilities_json",
]

SR = 22050
IMG_SIZE = (128, 128)
CLIMATE_COLUMNS = [
    "temp_max",
    "temp_min",
    "temp_mean",
    "humidity_mean",
    "rainfall_mm",
    "wind_speed_max",
    "cloud_cover_percent",
    "ndvi_mean",
]
MONTHLY_AGG_MAP = {
    "temp_max": "mean",
    "temp_min": "mean",
    "temp_mean": "mean",
    "humidity_mean": "mean",
    "rainfall_mm": "sum",
    "wind_speed_max": "mean",
    "cloud_cover_percent": "mean",
    "ndvi_mean": "mean",
}


@tf.keras.utils.register_keras_serializable(package="queenbee")
class SparseClassRecall(tf.keras.metrics.Metric):
    def __init__(self, class_id: int = 0, name: str = "sparse_class_recall", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.class_id = int(class_id)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        class_mask = tf.cast(tf.equal(y_true, self.class_id), self.dtype)
        pred_mask = tf.cast(tf.equal(y_pred, self.class_id), self.dtype)
        if sample_weight is None:
            weights = tf.ones_like(class_mask, dtype=self.dtype)
        else:
            weights = tf.cast(tf.reshape(sample_weight, [-1]), self.dtype)
            weights = tf.broadcast_to(weights, tf.shape(class_mask))
        weighted_mask = class_mask * weights
        tp = tf.reduce_sum(pred_mask * weighted_mask)
        fn = tf.reduce_sum((1.0 - pred_mask) * weighted_mask)
        self.true_positives.assign_add(tp)
        self.false_negatives.assign_add(fn)

    def result(self):
        return tf.math.divide_no_nan(self.true_positives, self.true_positives + self.false_negatives)

    def reset_states(self):
        self.true_positives.assign(0.0)
        self.false_negatives.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"class_id": int(self.class_id)})
        return config


def _infer_class_names() -> List[str]:
    if SPECTROGRAM_DIR.exists():
        class_dirs = sorted(p.name for p in SPECTROGRAM_DIR.iterdir() if p.is_dir())
        if class_dirs:
            return class_dirs
    return ["absent", "external", "present"]


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at {MODEL_PATH}. Please export the .keras file first."
    )

CLASS_NAMES = _infer_class_names()
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}
MODEL = load_model(MODEL_PATH, custom_objects={"SparseClassRecall": SparseClassRecall})
UNIFIED_MODEL = None
UNIFIED_FEATURE_COLUMNS: List[str] = []
UNIFIED_FEATURE_MEDIANS: Dict[str, float] = {}
UNIFIED_CONTEXT_DF: Optional[pd.DataFrame] = None


class WardInfo(BaseModel):
    id: str
    name: str
    subcounty: str
    latitude: float
    longitude: float


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]
    stress_probability: Optional[float] = None
    stress_label: Optional[str] = None
    ward: Optional[WardInfo] = None


class ClimateRecord(BaseModel):
    date: str
    temp_max: Optional[float] = None
    temp_min: Optional[float] = None
    temp_mean: Optional[float] = None
    humidity_mean: Optional[float] = None
    rainfall_mm: Optional[float] = None
    wind_speed_max: Optional[float] = None
    cloud_cover_percent: Optional[float] = None
    ndvi_mean: Optional[float] = None


class ClimateSeriesResponse(BaseModel):
    count: int
    records: List[ClimateRecord]
    ward: Optional[WardInfo] = None


class YieldForecastRecord(BaseModel):
    date: str
    predicted_yield_kg: float


class YieldForecastSummary(BaseModel):
    start_date: Optional[str]
    end_date: Optional[str]
    mean_kg_per_day: Optional[float]
    total_kg: Optional[float]


class YieldForecastResponse(BaseModel):
    summary: YieldForecastSummary
    records: List[YieldForecastRecord]
    ward: Optional[WardInfo] = None


app = FastAPI(title="Queen Bee Detection API", version="1.0.0")

allowed_origins = [
    origin.strip()
    for origin in os.getenv("API_CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def audio_bytes_to_input_tensor(audio_bytes: bytes) -> np.ndarray:
    if not audio_bytes:
        raise ValueError("Empty audio payload.")

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        y, sr = librosa.load(tmp.name, sr=SR)

    if y.size == 0:
        raise ValueError("Unable to decode any samples from the provided audio.")

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(2, 2), dpi=64)
    librosa.display.specshow(S_db, sr=sr, cmap="magma")
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    image = Image.open(buf).convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def run_model_inference(tensor: np.ndarray) -> PredictionResponse:
    probs = MODEL.predict(tensor, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = IDX_TO_CLASS.get(idx, str(idx))
    probabilities = {name: float(probs[idx_]) for idx_, name in IDX_TO_CLASS.items()}
    return PredictionResponse(label=label, confidence=float(probs[idx]), probabilities=probabilities)


def _parse_timestamp(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).isoformat()
        except ValueError:
            continue
    raise HTTPException(status_code=400, detail=f"Invalid recorded_at value: {value}")


def _parse_date_param(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid date value: {value}") from exc


def _safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (float, int, np.floating, np.integer)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, str) and not value.strip():
        return None
    if pd.isna(value):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _load_weather_ndvi_dataframe() -> pd.DataFrame:
    candidate_paths = [
        DATA_ROOT / WEATHER_NDVI_FILE,
        KAGGLE_DATA_ROOT / WEATHER_NDVI_FILE,
    ]
    dataset_path = next((path for path in candidate_paths if path.exists()), None)
    if dataset_path is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Weather and NDVI dataset not found. Regenerate it via the Kaggle notebook "
                f"or copy it into {DATA_ROOT} or {KAGGLE_DATA_ROOT}."
            ),
        )
    try:
        df = pd.read_csv(dataset_path, parse_dates=["date"])
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to load weather dataset: {exc}"
        ) from exc
    return df


def _serialize_climate_records(df: pd.DataFrame) -> List[ClimateRecord]:
    records: List[ClimateRecord] = []
    for _, row in df.iterrows():
        payload = {}
        date_value = row.get("date")
        if pd.isna(date_value):
            continue
        payload["date"] = pd.to_datetime(date_value).date().isoformat()
        for column in CLIMATE_COLUMNS:
            payload[column] = _safe_float(row.get(column))
        records.append(ClimateRecord(**payload))
    return records


def _load_yield_forecast_dataframe() -> pd.DataFrame:
    candidate_paths = [
        DATA_ROOT / YIELD_FORECAST_FILE,
        KAGGLE_DATA_ROOT / YIELD_FORECAST_FILE,
    ]
    forecast_path = next((path for path in candidate_paths if path.exists()), None)
    if forecast_path is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Yield forecast not found. Run the climate forecasting section of the notebook "
                f"or copy it into {DATA_ROOT} or {KAGGLE_DATA_ROOT}."
            ),
        )
    try:
        df = pd.read_csv(forecast_path)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to load yield forecast: {exc}"
        ) from exc

    if "date" not in df.columns:
        # Some exports persist the date as the index column with no header.
        first_column = df.columns[0]
        if first_column not in ("predicted_yield_kg", "date"):
            df.rename(columns={first_column: "date"}, inplace=True)
        elif "index" in df.columns:
            df.rename(columns={"index": "date"}, inplace=True)
        else:
            raise HTTPException(status_code=500, detail="Yield forecast missing date column.")
    if "predicted_yield_kg" not in df.columns:
        raise HTTPException(
            status_code=500, detail="Yield forecast missing 'predicted_yield_kg' column."
        )
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "predicted_yield_kg"]]


def _serialize_yield_records(df: pd.DataFrame) -> List[YieldForecastRecord]:
    records: List[YieldForecastRecord] = []
    for _, row in df.iterrows():
        date_value = pd.to_datetime(row["date"]).date().isoformat()
        value = _safe_float(row["predicted_yield_kg"])
        if value is None:
            continue
        records.append(YieldForecastRecord(date=date_value, predicted_yield_kg=value))
    return records


def _resolve_ward(ward_id: Optional[str]) -> Optional[Dict[str, float | str]]:
    if not ward_id:
        return None
    ward = WARD_LOOKUP.get(ward_id)
    if not ward:
        raise HTTPException(status_code=404, detail=f"Ward '{ward_id}' is not recognized.")
    return ward


def _ward_info_payload(ward: Optional[Dict[str, float | str]]) -> Optional[WardInfo]:
    if not ward:
        return None
    return WardInfo(
        id=str(ward["id"]),
        name=str(ward["name"]),
        subcounty=str(ward["subcounty"]),
        latitude=float(ward["latitude"]),
        longitude=float(ward["longitude"]),
    )


def _ensure_acoustic_log_schema() -> None:
    if not ACOUSTIC_LOG_PATH.exists():
        return
    try:
        df = pd.read_csv(ACOUSTIC_LOG_PATH)
    except Exception:
        return
    missing = [field for field in ACOUSTIC_LOG_FIELDS if field not in df.columns]
    if not missing:
        return
    for field in missing:
        df[field] = "" if field not in {"latitude", "longitude", "confidence"} else np.nan
    df.to_csv(ACOUSTIC_LOG_PATH, index=False, columns=ACOUSTIC_LOG_FIELDS)


def _log_prediction_entry(entry: Dict[str, Optional[str]]) -> None:
    try:
        exists = ACOUSTIC_LOG_PATH.exists()
        with ACOUSTIC_LOG_PATH.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=ACOUSTIC_LOG_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(entry)
    except OSError as exc:
        # Do not block inference if logging fails.
        print(f"[acoustic-log] Failed to append entry: {exc}")


def _load_unified_assets() -> None:
    global UNIFIED_MODEL, UNIFIED_FEATURE_COLUMNS, UNIFIED_FEATURE_MEDIANS, UNIFIED_CONTEXT_DF
    if UNIFIED_MODEL_PATH.exists():
        payload = joblib.load(UNIFIED_MODEL_PATH)
        UNIFIED_MODEL = payload.get("model")
        UNIFIED_FEATURE_COLUMNS = payload.get("features", [])
    else:
        print(f"[unified-model] Model not found at {UNIFIED_MODEL_PATH}; stress scoring disabled.")
        return

    dataset_path = None
    if UNIFIED_DATASET_PARQUET.exists():
        dataset_path = UNIFIED_DATASET_PARQUET
    elif UNIFIED_DATASET_CSV.exists():
        dataset_path = UNIFIED_DATASET_CSV

    if dataset_path is None:
        print("[unified-model] Unified feature dataset not found; stress scoring disabled.")
        UNIFIED_MODEL = None
        return

    try:
        if dataset_path.suffix == ".parquet":
            df = pd.read_parquet(dataset_path)
        else:
            df = pd.read_csv(dataset_path)
    except Exception as exc:
        print(f"[unified-model] Failed to load dataset {dataset_path}: {exc}")
        UNIFIED_MODEL = None
        return

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    UNIFIED_CONTEXT_DF = df
    if UNIFIED_FEATURE_COLUMNS:
        medians = {}
        for col in UNIFIED_FEATURE_COLUMNS:
            if col in df.columns:
                medians[col] = float(df[col].median(skipna=True))
            else:
                medians[col] = 0.0
        UNIFIED_FEATURE_MEDIANS = medians
    print("[unified-model] Loaded unified model and dataset for stress scoring.")


_ensure_acoustic_log_schema()
_load_unified_assets()


@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "classes": ", ".join(CLASS_NAMES),
        "model_path": str(MODEL_PATH),
        "unified_model": bool(UNIFIED_MODEL),
    }


@app.get("/locations/wards", response_model=List[WardInfo])
def list_wards() -> List[WardInfo]:
    return [
        WardInfo(
            id=str(ward["id"]),
            name=str(ward["name"]),
            subcounty=str(ward["subcounty"]),
            latitude=float(ward["latitude"]),
            longitude=float(ward["longitude"]),
        )
        for ward in WARD_REGISTRY
    ]


@app.get("/climate/daily", response_model=ClimateSeriesResponse)
def climate_daily(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 90,
    ward_id: Optional[str] = None,
) -> ClimateSeriesResponse:
    if limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be a positive integer.")
    df = _load_weather_ndvi_dataframe().sort_values("date")
    ward = _resolve_ward(ward_id)
    start_dt = _parse_date_param(start_date) if start_date else None
    end_dt = _parse_date_param(end_date) if end_date else None
    if start_dt and end_dt and end_dt < start_dt:
        raise HTTPException(status_code=400, detail="end_date must be after start_date.")
    if start_dt:
        df = df[df["date"] >= start_dt]
    if end_dt:
        df = df[df["date"] <= end_dt]
    if limit is not None and limit > 0:
        df = df.tail(limit)
    records = _serialize_climate_records(df)
    return ClimateSeriesResponse(count=len(records), records=records, ward=_ward_info_payload(ward))


@app.get("/climate/monthly", response_model=ClimateSeriesResponse)
def climate_monthly(months: int = 12, ward_id: Optional[str] = None) -> ClimateSeriesResponse:
    if months <= 0:
        raise HTTPException(status_code=400, detail="months must be a positive integer.")
    df = _load_weather_ndvi_dataframe()
    ward = _resolve_ward(ward_id)
    monthly = (
        df.set_index("date")
        .resample("MS")
        .agg(MONTHLY_AGG_MAP)
        .reset_index()
        .sort_values("date")
    )
    monthly = monthly.tail(months)
    records = _serialize_climate_records(monthly)
    return ClimateSeriesResponse(count=len(records), records=records, ward=_ward_info_payload(ward))


@app.get("/yield/forecast", response_model=YieldForecastResponse)
def yield_forecast(limit: int = 120, ward_id: Optional[str] = None) -> YieldForecastResponse:
    if limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be a positive integer.")
    df = _load_yield_forecast_dataframe().sort_values("date")
    ward = _resolve_ward(ward_id)
    df = df.tail(limit)
    if df.empty:
        raise HTTPException(status_code=404, detail="Yield forecast is empty.")
    records = _serialize_yield_records(df)
    if not records:
        raise HTTPException(status_code=404, detail="Yield forecast has no usable records.")
    summary = YieldForecastSummary(
        start_date=pd.to_datetime(df["date"].min()).date().isoformat(),
        end_date=pd.to_datetime(df["date"].max()).date().isoformat(),
        mean_kg_per_day=_safe_float(df["predicted_yield_kg"].mean()),
        total_kg=_safe_float(df["predicted_yield_kg"].sum()),
    )
    return YieldForecastResponse(summary=summary, records=records, ward=_ward_info_payload(ward))


def _build_unified_feature_vector(
    hive_id: Optional[str], probs: Dict[str, float]
) -> Optional[np.ndarray]:
    if not UNIFIED_MODEL or UNIFIED_CONTEXT_DF is None or not hive_id:
        return None

    if "hive_id" not in UNIFIED_CONTEXT_DF.columns:
        return None

    hive_rows = UNIFIED_CONTEXT_DF[UNIFIED_CONTEXT_DF["hive_id"] == hive_id]
    if hive_rows.empty:
        return None

    row = hive_rows.sort_values("date").iloc[-1]
    feature_values = []
    for feature in UNIFIED_FEATURE_COLUMNS:
        value = row.get(feature, np.nan)
        if feature.startswith("acoustic_prob_"):
            label = feature.replace("acoustic_prob_", "")
            if label in probs:
                value = probs[label]
        if pd.isna(value):
            value = UNIFIED_FEATURE_MEDIANS.get(feature, 0.0)
        feature_values.append(value)
    return np.array(feature_values, dtype=np.float32)


def _score_unified_model(
    hive_id: Optional[str], probs: Dict[str, float]
) -> Optional[Tuple[float, str]]:
    if not UNIFIED_MODEL:
        return None

    vector = _build_unified_feature_vector(hive_id, probs)
    if vector is None:
        return None

    try:
        stress_prob = float(UNIFIED_MODEL.predict_proba(vector.reshape(1, -1))[0][1])
    except Exception as exc:
        print(f"[unified-model] Prediction failed: {exc}")
        return None

    label = "stress" if stress_prob >= 0.5 else "stable"
    return stress_prob, label


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    hive_id: Optional[str] = Form(None),
    ward_id: Optional[str] = Form(None),
    recorded_at: Optional[str] = Form(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
) -> PredictionResponse:
    if file.content_type not in ("audio/wav", "audio/x-wav", "application/octet-stream"):
        raise HTTPException(status_code=415, detail="Only WAV audio files are supported.")

    audio_bytes = await file.read()
    ward = _resolve_ward(ward_id)
    resolved_lat = (
        float(ward["latitude"])
        if ward and ward.get("latitude") is not None
        else (float(latitude) if latitude is not None else None)
    )
    resolved_lon = (
        float(ward["longitude"])
        if ward and ward.get("longitude") is not None
        else (float(longitude) if longitude is not None else None)
    )
    try:
        tensor = await run_in_threadpool(audio_bytes_to_input_tensor, audio_bytes)
        prediction = await run_in_threadpool(run_model_inference, tensor)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    recorded_timestamp = _parse_timestamp(recorded_at) or datetime.utcnow().isoformat()
    stress_info = await run_in_threadpool(_score_unified_model, hive_id, prediction.probabilities)
    log_entry = {
        "created_at": datetime.utcnow().isoformat(),
        "recorded_at": recorded_timestamp,
        "hive_id": hive_id or "",
        "ward_id": ward["id"] if ward else (ward_id or ""),
        "latitude": resolved_lat,
        "longitude": resolved_lon,
        "audio_filename": file.filename,
        "predicted_label": prediction.label,
        "confidence": prediction.confidence,
        "probabilities_json": json.dumps(prediction.probabilities),
    }
    _log_prediction_entry(log_entry)

    if stress_info:
        stress_prob, stress_label = stress_info
        prediction.stress_probability = stress_prob
        prediction.stress_label = stress_label

    response = PredictionResponse(
        label=prediction.label,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
        stress_probability=prediction.stress_probability,
        stress_label=prediction.stress_label,
        ward=_ward_info_payload(ward),
    )
    return response

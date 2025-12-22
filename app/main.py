import csv
import io
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from PIL import Image

matplotlib.use("Agg")

APP_ROOT = Path(__file__).resolve().parent.parent
CONTENT_ROOT = APP_ROOT / "content"
BEES_ROOT = CONTENT_ROOT / "beehive_audio" / "7" / "Dataset"
DATA_ROOT = CONTENT_ROOT / "main-data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_PATH = BEES_ROOT / "queenbee_final_tuned_model.h5"
SPECTROGRAM_DIR = BEES_ROOT / "Spectograms"
ACOUSTIC_LOG_PATH = DATA_ROOT / "acoustic_predictions.csv"
UNIFIED_MODEL_PATH = DATA_ROOT / "hive_unified_model.pkl"
UNIFIED_DATASET_PARQUET = DATA_ROOT / "hive_weather_acoustic.parquet"
UNIFIED_DATASET_CSV = DATA_ROOT / "hive_weather_acoustic.csv"
ACOUSTIC_LOG_FIELDS = [
    "created_at",
    "recorded_at",
    "hive_id",
    "latitude",
    "longitude",
    "audio_filename",
    "predicted_label",
    "confidence",
    "probabilities_json",
]

SR = 22050
IMG_SIZE = (128, 128)


def _infer_class_names() -> List[str]:
    if SPECTROGRAM_DIR.exists():
        class_dirs = sorted(p.name for p in SPECTROGRAM_DIR.iterdir() if p.is_dir())
        if class_dirs:
            return class_dirs
    return ["absent", "external", "present"]


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Please export the .h5 file first.")

CLASS_NAMES = _infer_class_names()
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}
MODEL = load_model(MODEL_PATH)
UNIFIED_MODEL = None
UNIFIED_FEATURE_COLUMNS: List[str] = []
UNIFIED_FEATURE_MEDIANS: Dict[str, float] = {}
UNIFIED_CONTEXT_DF: Optional[pd.DataFrame] = None


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]
    stress_probability: Optional[float] = None
    stress_label: Optional[str] = None


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


_load_unified_assets()


@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "classes": ", ".join(CLASS_NAMES),
        "model_path": str(MODEL_PATH),
        "unified_model": bool(UNIFIED_MODEL),
    }


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
    recorded_at: Optional[str] = Form(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
) -> PredictionResponse:
    if file.content_type not in ("audio/wav", "audio/x-wav", "application/octet-stream"):
        raise HTTPException(status_code=415, detail="Only WAV audio files are supported.")

    audio_bytes = await file.read()
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
        "latitude": latitude,
        "longitude": longitude,
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

    return prediction

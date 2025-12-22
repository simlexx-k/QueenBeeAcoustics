# QueenBeeAcoustics

## FastAPI Deployment

The repository now bundles a lightweight API for serving the trained CNN (`app/main.py`). It automatically selects a TPU, GPU, or CPU strategy when TensorFlow starts and exposes two endpoints:

- `GET /health` – confirms the service is up and shows the detected classes/model path.
- `POST /predict` – accepts a WAV file upload and returns the predicted label, confidence, and class probabilities. When `content/main-data/hive_unified_model.pkl` (and `hive_weather_acoustic.*`) exist, the response also includes `stress_probability` and `stress_label`, which reflect the Gradient Boosting hive-health model trained from weather + hive + acoustic data.

### Running the API

1. Install dependencies (use `uv`/`pip`/`uv pip sync` as preferred) so FastAPI, TensorFlow, librosa, and friends are available.
2. Ensure the tuned model file exists at `content/beehive_audio/7/Dataset/queenbee_final_tuned_model.h5`.
3. Start the server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Sample request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@example.wav;type=audio/wav"
```

The JSON response includes the predicted `label`, an overall `confidence`, and the probability assigned to each tracked class so downstream clients can make their own decisions.

## Acoustic Logs & Unified Modeling

- Every `/predict` call now appends a row to `content/main-data/acoustic_predictions.csv` (timestamp, hive id, coordinates, class probabilities). Supply optional `hive_id`, `recorded_at`, `latitude`, and `longitude` form fields from the client to capture richer metadata; otherwise the service uses the upload time.
- Use `scripts/merge_hive_acoustic.py` to join hive logs (`hive_logs_2008_2025.csv`), weather/NDVI histories, and the acoustic log into `content/main-data/hive_weather_acoustic.parquet`.
- Train a second-stage hive health model with `scripts/train_unified_model.py`. This fits a Gradient Boosting classifier that predicts whether a hive is under stress using weather, hive metrics, and the latest acoustic probabilities. The model plus metrics land in `content/main-data/hive_unified_model.pkl` and `..._metrics.json`.

### Workflow

1. Generate/refresh hive + weather datasets via the notebook.
2. Run acoustic inference through FastAPI (manually or via the Next.js frontend) to populate `acoustic_predictions.csv`.
3. Build the unified dataset: `python scripts/merge_hive_acoustic.py`.
4. Train the consolidated model: `python scripts/train_unified_model.py`.
5. Restart the FastAPI server so it loads `hive_unified_model.pkl` plus the unified dataset. Each `/predict` call will now return both the acoustic class probabilities and the hive stress verdict derived from the contextual model.

# BeeUnity: Multimodal Bee Health Intelligence

BeeUnity combines acoustic sensing and environmental analytics to help Makueni County beekeepers monitor queen vitality, hive stress, and occupancy risk. The repository hosts the end-to-end Kaggle notebook (`Kaggle_QueenBee_Makueni.ipynb`), auxiliary scripts, and a FastAPI service for serving the tuned acoustic CNN plus the contextual hive-health model.

## Project Structure

- `Kaggle_QueenBee_Makueni.ipynb` – canonical research notebook covering both pipelines: Kaggle audio ingestion, spectrogram generation, CNN/KerasTuner training + calibration, followed by weather/NDVI staging, hive telemetry synthesis, gradient boosting, and PyTorch sequence modeling.
- `content/` – cached datasets (weather, NDVI, hive logs, acoustic predictions) used when Kaggle network access is disabled.
- `scripts/` – helpers for merging acoustic predictions with hive logs and training the consolidated hive stress model.
- `app/` – FastAPI app exposing `/predict` (acoustic inference + optional stress verdict) and `/health`.
- `queenbee/`, `queenbee_tuner/`, `MakueniBeekeepData.ipynb` – earlier experiments retained for reference.

## Notebook Workflows

### 1. Queen Bee Acoustic Detection Pipeline
1. Attach Kaggle dataset `harshkumar1711/beehive-audio-dataset-with-queen-and-without-queen`.
2. Run the notebook section titled **Queen Bee Acoustic Detection Pipeline**. It will:
   - Discover the dataset, cache mel spectrograms under `/kaggle/working/spectrograms/`, and log class counts.
   - Build stratified train/val/test splits, compute class weights, and monitor a custom `SparseClassRecall` on the queen-absent class.
   - Train the baseline CNN, launch KerasTuner Hyperband to optimize filters/dense units/dropout, fine-tune the best trial, and save the `.keras` artifact.
   - Calibrate per-class thresholds using validation precision-recall curves and report argmax vs calibrated metrics (confusion matrix, macro ROC/PR AUC, class-wise precision/recall).
3. The tuned model plus calibrated thresholds feed downstream scripts (`app/`, `scripts/train_unified_model.py`). Figures/metrics are emitted under `artifacts/figures/` for manuscript use.

### 2. Makueni Apiary Intelligence Pipeline
1. Keep `ENABLE_REMOTE_CALLS=False` on Kaggle to load cached weather (`makueni_weather_2008_2025.csv`) and NDVI exports. When running locally with API access, flip the flag to regenerate fresh Open-Meteo/MODIS slices.
2. Execute the **Makueni Apiary Intelligence Pipeline** section to:
   - Merge weather, NDVI, floral calendar, and (optionally synthetic) hive telemetry records.
   - Engineer rolling statistics, calendar features, and imputed numeric matrices for modelling.
   - Train a class-weighted HistGradientBoostingClassifier and plot ROC + classification reports.
   - Slice sequential windows, oversample via `WeightedRandomSampler`, and train the PyTorch sequence model (set `SEQUENCE_MODEL_VARIANT` to `"cnn"` or `"cnn_gru"`). Logged AUC/PR curves + thresholded confusion matrices quantify stress recall.
3. Exported metrics and model weights populate the BeeUnity report and act as inputs for the FastAPI unified predictor. Tabular/sequence evaluation artifacts are written to `artifacts/figures/`.

## FastAPI Deployment

The FastAPI app in `app/main.py` reuses the tuned CNN and, when available, the hive stress model built from `scripts/train_unified_model.py`.

1. Install dependencies (TensorFlow, librosa, FastAPI, scikit-learn, PyTorch as needed).
2. Place the tuned acoustic model at `content/beehive_audio/.../queenbee_final_tuned_model.keras` and the unified hive model inside `content/main-data/hive_unified_model.pkl`.
3. Launch the server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- `GET /health` reports loaded models/classes.
- `POST /predict` accepts a WAV upload plus optional hive metadata and returns acoustic class probabilities, calibrated label, and (if present) the hive stress probability/label from the contextual model.

Acoustic predictions are appended to `content/main-data/acoustic_predictions.csv`. Use `scripts/merge_hive_acoustic.py` followed by `scripts/train_unified_model.py` to refresh the unified dataset + model before restarting the API.

Generate report-ready markdown/figures anytime with:

```bash
python3 scripts/generate_report_markdown.py
```

The script ingests artifacts under `artifacts/figures/` and rewrites `docs/*.md` with the latest metrics/tables for direct inclusion in the manuscript.

## Reproducing Report Figures

1. Run the notebook end-to-end on Kaggle (GPU T4). All intermediate CSVs and artifacts save under `/kaggle/working` or `content/`.
2. Export key plots/tables:
   - Acoustic confusion matrix + ROC/PR AUC (calibrated and argmax modes).
   - HistGradientBoosting classification report + ROC curve.
   - Sequence-model classification report, PR curve, ROC curve.
3. Reference these outputs when updating the manuscript chapters on methodology, results, and discussion.

## Next Steps

- Integrate real hive telemetry as it becomes available to replace synthetic logs.
- Extend the FastAPI service with lightweight dashboards (e.g., Streamlit) mirroring the notebook visualizations.
- Continue hyperparameter exploration for the sequence model (attention pooling, focal loss, richer augmentation) to lift stress recall without collapsing precision.

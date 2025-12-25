# System Integration & Deployment

The tuned acoustic CNN (`queenbee_final_tuned_model.keras`) and the contextual hive-stress model (`hive_unified_model.pkl`) feed the BeeUnity FastAPI service (`app/main.py`). Acoustic predictions append to `content/main-data/acoustic_predictions.csv`, merged with weather/hive data via `scripts/merge_hive_acoustic.py`, and retrained with `scripts/train_unified_model.py`.

FastAPI endpoints:
- `GET /health` – confirms loaded models/classes.
- `POST /predict` – accepts WAV uploads plus optional hive metadata, returning calibrated acoustic probabilities and (if available) the stress probability/label.

Figures/tables referenced in the report should cite the generated assets under `/kaggle/working/figures/` to maintain reproducibility.

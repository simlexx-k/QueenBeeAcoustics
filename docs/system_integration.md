# System Integration & Deployment
- Tuned acoustic model: `queenbee_final_tuned_model.keras`
- Hive stress model: `content/main-data/hive_unified_model.pkl`
- FastAPI service: `app/main.py`
- Unified dataset scripts: `scripts/merge_hive_acoustic.py`, `scripts/train_unified_model.py`

Acoustic predictions feed the unified dataset, which in turn drives the contextual model surfaced via `/predict`. Embed references to the generated figures listed above when discussing the deployment pipeline.

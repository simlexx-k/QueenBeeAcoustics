# Makueni Environmental & Hive Data Fusion

The notebook clamps user-defined date ranges to the latest Open-Meteo and MODIS NDVI availability, using cached CSVs under `content/main-data/` when Kaggle disallows outbound calls. Weather, NDVI, floral calendar, and hive logs (synthetic or real) are merged into `model_df`. Rolling per-hive statistics, median-imputed numeric matrices, and calendar indices form the features consumed by both gradient boosting and sequence models.

Key artifacts for the report:

- `tabular_hgb_classification_report.txt` / `.csv` – precision/recall/F1 by class for the HistGradientBoosting baseline.
- `tabular_hgb_roc.png` – ROC plot cited in Section 4.
- `tabular_hgb_metrics.json` – ROC-AUC value referenced in the results summary.

These files provide the quantitative backbone for the BeeUnity analysis described in Chapters 3–4.

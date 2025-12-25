# Temporal Sequence Modeling (CNN vs CNN-GRU)

Sliding 12-step hive telemetry windows feed two deep models: a pure 1D CNN and a Conv+bi-GRU hybrid (`SEQUENCE_MODEL_VARIANT` flag). WeightedRandomSampler balances the stress class, BCE-with-logits plus class-balanced weights handles loss, and early stopping (patience=5) prevents overfitting.

Each run exports:

- `sequence_cnn_pr_roc.png` and `sequence_cnn_gru_pr_roc.png` – combined PR/ROC plots for use as Figures Y/Z.
- `sequence_<variant>_classification_report.txt` – detailed precision/recall/F1.
- `sequence_<variant>_metrics.json` – ROC-AUC and chosen threshold per variant.

Include the variant that best aligns with your discussion (e.g., higher stress recall) and cite the corresponding JSON values in the manuscript.

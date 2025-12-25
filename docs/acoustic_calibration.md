# Acoustic Probability Calibration & Thresholding

To avoid false negatives on queen-loss events, we calibrate per-class thresholds using validation precision–recall curves. The tuned CNN’s validation logits produce class-specific F1-optimal thresholds that are then applied to the held-out test generator.

Artifacts saved by the notebook:

- `acoustic_confusion_matrix.png` – shows post-calibration confusion matrix used for Figure X.
- `acoustic_classification_report.txt` – textual precision/recall/F1 per class (insert into Appendix/Table).
- `acoustic_metrics_table.csv` – Argmax vs calibrated macro metrics (Accuracy, Precision, Recall, F1).
- `acoustic_auc_summary.json` – macro ROC-AUC and PR-AUC values reported in Section 4.

Quote these files directly when drafting the results chapter; they are generated deterministically each Kaggle run.

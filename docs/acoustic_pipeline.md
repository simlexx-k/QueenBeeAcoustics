# Acoustic Dataset and Spectrogram Engineering

BeeUnity’s queen-state classifier is trained on the Kaggle dataset `harshkumar1711/beehive-audio-dataset-with-queen-and-without-queen`. `_discover_audio_dataset()` validates the mounted `/kaggle/input` structure and exposes canonical paths for the `QueenBee Present`, `QueenBee Absent`, and `External Noise` subsets, guaranteeing dataset traceability. Each WAV undergoes trimming, normalization, mono conversion, and fixed-length padding (3 s at 22.05 kHz) before rendering a 128×128 mel spectrogram. Spectrograms are cached under `/kaggle/working/spectrograms/<class>/` so downstream generators operate purely on PNGs.

After stratified DataFrame splits, the CNN tracks class imbalance via `compute_class_weight`, monitors a custom `SparseClassRecall` on the queen-absent class, and performs Hyperband tuning (`val_recall_absent` objective). Evaluation artifacts land under `/kaggle/working/figures/`:

- `acoustic_metrics_table.csv` – Argmax vs calibrated Accuracy / Macro Precision / Macro Recall / Macro F1.
- `acoustic_confusion_matrix.png` – Calibrated confusion matrix for manuscript Figure X.
- `acoustic_classification_report.txt` – Class-wise precision/recall/F1 for Appendix tables.
- `acoustic_auc_summary.json` – macro ROC-AUC and PR-AUC (cite in Section 4).

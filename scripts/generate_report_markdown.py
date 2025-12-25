#!/usr/bin/env python3
"""Generate report-ready markdown files from notebook artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pandas as pd
try:
    import tabulate  # noqa: F401
    HAVE_TABULATE = True
except ImportError:
    HAVE_TABULATE = False

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "artifacts" / "figures"
DOCS_DIR = REPO_ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)


def ensure_fig_dir() -> bool:
    if not FIG_DIR.exists():
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        return False
    return True


def read_text(path: Path) -> str:
    return path.read_text().strip()


def df_to_table(df: pd.DataFrame) -> str:
    if HAVE_TABULATE:
        return df.to_markdown(index=False, floatfmt=".3f")
    # Fallback simple table
    return df.to_string(index=False)


def write_markdown(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def acoustic_section() -> str:
    metrics_path = FIG_DIR / "acoustic_metrics_table.csv"
    auc_path = FIG_DIR / "acoustic_auc_summary.json"
    report_path = FIG_DIR / "acoustic_classification_report.txt"
    confusion_path = FIG_DIR / "acoustic_confusion_matrix.png"

    if not metrics_path.exists():
        return dedent(
            f"""
            # Acoustic Dataset and Spectrogram Engineering
            Notebook artifacts not found. Run the acoustic pipeline so `{metrics_path.name}` and related files exist, then re-execute `scripts/generate_report_markdown.py`.
            """
        )

    metrics_df = pd.read_csv(metrics_path)
    metrics_md = df_to_table(metrics_df)
    auc_data = json.loads(read_text(auc_path)) if auc_path.exists() else {}
    auc_summary = " | ".join(f"{k}: {v:.4f}" for k, v in auc_data.items()) if auc_data else ""

    report_txt = read_text(report_path) if report_path.exists() else "(classification report missing)"

    return dedent(
        f"""
        # Acoustic Dataset and Spectrogram Engineering
        - **Artifacts directory**: `{FIG_DIR}`
        - **Confusion matrix**: `{confusion_path.name}`
        - **Metrics summary**: `{auc_summary or 'N/A'}`
        
        ## Argmax vs Calibrated Metrics
        {metrics_md}
        
        ## Classification Report (Calibrated)
        ```
        {report_txt}
        ```
        """
    )


def acoustic_calibration_section() -> str:
    confusion_path = FIG_DIR / "acoustic_confusion_matrix.png"
    thresholds_path = FIG_DIR / "acoustic_metrics_table.csv"
    auc_path = FIG_DIR / "acoustic_auc_summary.json"
    if not confusion_path.exists():
        return dedent(
            f"""
            # Acoustic Probability Calibration
            Calibration figures are missing. Ensure `{confusion_path.name}` and `{thresholds_path.name}` exist, then rerun the generator script.
            """
        )
    auc_data = json.loads(read_text(auc_path)) if auc_path.exists() else {}
    auc_summary = " | ".join(f"{k}: {v:.4f}" for k, v in auc_data.items()) if auc_data else ""
    return dedent(
        f"""
        # Acoustic Probability Calibration
        - **Confusion matrix figure**: `{confusion_path.name}`
        - **Metrics table**: `{thresholds_path.name}`
        - **AUC summary**: {auc_summary or 'N/A'}
        
        Calibration thresholds are derived from validation precision–recall curves and then applied to the test generator. Reference the figure/table above in the manuscript.
        """
    )


def makueni_section() -> str:
    report_path = FIG_DIR / "tabular_hgb_classification_report.txt"
    roc_path = FIG_DIR / "tabular_hgb_roc.png"
    summary_path = FIG_DIR / "tabular_hgb_metrics.json"

    if not report_path.exists():
        return dedent(
            f"""
            # Makueni Environmental & Hive Data Fusion
            Run the tabular gradient boosting cell so `{report_path.name}`, `{roc_path.name}`, and `{summary_path.name}` exist before regenerating this markdown.
            """
        )

    report_txt = read_text(report_path)
    summary = json.loads(read_text(summary_path)) if summary_path.exists() else {}
    roc_auc = summary.get("roc_auc")
    roc_text = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
    return dedent(
        f"""
        # Makueni Environmental & Hive Data Fusion
        - **Gradient Boosting ROC figure**: `{roc_path.name}`
        - **ROC-AUC**: {roc_text}
        
        ## Classification Report
        ```
        {report_txt}
        ```
        """
    )


def sequence_section() -> str:
    blocks = []
    for variant in ("cnn", "cnn_gru"):
        report_path = FIG_DIR / f"sequence_{variant}_classification_report.txt"
        metrics_path = FIG_DIR / f"sequence_{variant}_metrics.json"
        fig_path = FIG_DIR / f"sequence_{variant}_pr_roc.png"
        if not report_path.exists():
            blocks.append(
                f"### Sequence Variant: {variant}\nArtifacts missing for this variant. Ensure `{report_path.name}` exists."
            )
            continue
        report_txt = read_text(report_path)
        metrics = json.loads(read_text(metrics_path)) if metrics_path.exists() else {}
        roc_auc = metrics.get("roc_auc")
        threshold = metrics.get("best_threshold")
        roc_text = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
        thresh_text = f"{threshold:.3f}" if threshold is not None else "N/A"
        blocks.append(
            dedent(
                f"""
                ### Sequence Variant: {variant}
                - **PR/ROC figure**: `{fig_path.name}`
                - **ROC-AUC**: {roc_text}
                - **Best threshold**: {thresh_text}
                
                ```
                {report_txt}
                ```
                """
            ).strip()
        )
    return "# Temporal Sequence Modeling\n\n" + "\n\n".join(blocks)


def system_section() -> str:
    return dedent(
        """
        # System Integration & Deployment
        - Tuned acoustic model: `queenbee_final_tuned_model.keras`
        - Hive stress model: `content/main-data/hive_unified_model.pkl`
        - FastAPI service: `app/main.py`
        - Unified dataset scripts: `scripts/merge_hive_acoustic.py`, `scripts/train_unified_model.py`
        
        Acoustic predictions feed the unified dataset, which in turn drives the contextual model surfaced via `/predict`. Embed references to the generated figures listed above when discussing the deployment pipeline.
        """
    )


def main() -> None:
    ensure_fig_dir()
    write_markdown(DOCS_DIR / "acoustic_pipeline.md", acoustic_section())
    write_markdown(DOCS_DIR / "acoustic_calibration.md", acoustic_calibration_section())
    write_markdown(DOCS_DIR / "makueni_data_fusion.md", makueni_section())
    write_markdown(DOCS_DIR / "sequence_model.md", sequence_section())
    write_markdown(DOCS_DIR / "system_integration.md", system_section())
    print("Report markdown updated ->", DOCS_DIR)


if __name__ == "__main__":
    main()

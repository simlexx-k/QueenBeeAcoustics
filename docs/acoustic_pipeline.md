# Acoustic Dataset and Spectrogram Engineering
        - **Artifacts directory**: `/home/kosgei/Projects/ResearchProject1/artifacts/figures`
        - **Confusion matrix**: `acoustic_confusion_matrix.png`
        - **Metrics summary**: `roc_auc: 0.9972 | pr_auc: 0.9961`

        ## Argmax vs Calibrated Metrics
        | Mode       |   Accuracy |   Macro Precision |   Macro Recall |   Macro F1 |
|:-----------|-----------:|------------------:|---------------:|-----------:|
| Argmax     |      0.966 |             0.961 |          0.972 |      0.966 |
| Calibrated |      0.974 |             0.973 |          0.976 |      0.974 |

        ## Classification Report (Calibrated)
        ```
        precision    recall  f1-score   support

      absent       0.97      0.96      0.96       400
    external       0.97      1.00      0.98       400
     present       0.98      0.97      0.97       800

    accuracy                           0.97      1600
   macro avg       0.97      0.98      0.97      1600
weighted avg       0.97      0.97      0.97      1600
        ```

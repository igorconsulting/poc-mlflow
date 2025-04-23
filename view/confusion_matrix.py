import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mlflow.xgboost import log_model
from sklearn.metrics import confusion_matrix


def save_confusion_matrix_plot(y_true, y_pred, labels, path: str):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()

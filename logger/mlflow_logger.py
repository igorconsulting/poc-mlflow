import os

import mlflow
from mlflow.xgboost import log_model

from view.confusion_matrix import save_confusion_matrix_plot


def log_params(params: dict):
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_metrics(metrics: dict):
    to_log = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    mlflow.log_metrics(to_log)


def log_confusion_matrix(y_true, y_pred, labels):
    cm_path = "plots/confusion_matrix.png"
    save_confusion_matrix_plot(y_true, y_pred, labels, cm_path)
    mlflow.log_artifact(cm_path)


def persist_results(model, metrics, y_test, target_names, params) -> dict:
    try:
        with mlflow.start_run():
            run_id = mlflow.active_run().info.run_id
            log_params(params)
            mlflow.log_param("problem_type", "multiclass")
            log_metrics(metrics)
            log_confusion_matrix(y_test, metrics["y_pred"], target_names)
            log_model(model, artifact_path="model")

        return {
            "status": "success",
            "message": "Results logged and model saved successfully",
            "run_id": run_id,
            "logged_metrics": list(metrics.keys()),
        }

    except Exception as e:
        return {
            "status": "failure",
            "message": f"Logging failed: {str(e)}",
            "error_type": type(e).__name__,
        }

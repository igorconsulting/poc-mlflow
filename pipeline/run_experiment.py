import mlflow

from core.evaluator import evaluate_model
from core.trainer import train_model


def run_pipeline(
    X_train, X_test, y_train, y_test, target_names: list[str], params: dict
) -> dict:
    model = train_model(params, X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    return {"model": model, "metrics": metrics, "target_names": target_names}

import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss


def evaluate_model(model, X_test, y_test):
    dtest = xgb.DMatrix(X_test)
    y_proba = model.predict(dtest)
    y_pred = y_proba.argmax(axis=1)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

    return metrics

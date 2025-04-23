import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split

# load dataset
iris = datasets.load_iris()

X = iris.data
y = iris.target

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.21, random_state=31
)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


# log confusion matrix
def log_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    cm_path = "confusion_matrix.png"
    mlflow.log_artifact(cm_path)


with mlflow.start_run():
    # get parameters
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "learning_rate": 0.3,
        "eval_metric": "mlogloss",
        "colsample_bytree": 0.7,
        "subsample": 1,
        "seed": 42,
    }

    # set model
    model = xgb.train(params, dtrain, evals=[(dtrain, "train")], verbose_eval=False)

    # get prediction
    y_proba = model.predict(dtest)
    y_pred = y_proba.argmax(axis=1)

    # get metrics
    loss = log_loss(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)

    # explicitly log parameter
    for key, value in params.items():
        mlflow.log_param(key, value)
    mlflow.log_param("problem_type", "multiclass")

    # log metrics
    mlflow.log_metrics({"log_loss": loss, "accuracy": acc})

    # log confusion matrix
    log_confusion_matrix(y_test, y_pred, iris.target_names)

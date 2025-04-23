import os

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

# print metrics
print(f"Metrics:\nLoss: {loss}\nAccuracy: {acc}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# save model
os.makedirs("model", exist_ok=True)
model.save_model("model/iris_xgboost.json")

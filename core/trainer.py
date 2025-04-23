import xgboost as xgb


def train_model(params: dict, X_train, y_train):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(params, dtrain, evals=[(dtrain, "train")], verbose_eval=False)
    return model

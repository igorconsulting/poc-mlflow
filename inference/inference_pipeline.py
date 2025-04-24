import mlflow

from core.evaluator import evaluate_model
from core.trainer import train_model
from core.preprocessing import preprocessing_pipeline
from core.prediction import predict

def inference_pipeline(
    data, model
) -> dict:
    processed_data = preprocessing_pipeline(data)
    y = predict(model, processed_data)

    return {"prediction": y}

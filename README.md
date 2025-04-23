# MLflow XGBoost Iris Classifier

This repository implements a modular machine learning pipeline for multi-class classification using the Iris dataset and XGBoost, with full experiment tracking through MLflow. The project adheres to Clean Architecture principles, functional programming practices, and software engineering standards such as SOLID and Separation of Concerns.

## Purpose

The goal is to provide a reproducible, maintainable, and extensible pipeline for model training, evaluation, and logging—suitable for use in both research and production-grade machine learning workflows.

## Project Structure

```
mlflow_xgboost_iris/
├── config/
│   └── hyperparams.py              # Model hyperparameters
│
├── core/
│   ├── data_loader.py              # Loads and splits the dataset
│   ├── trainer.py                  # Trains the model
│   ├── evaluator.py                # Evaluates performance
│   └── visualizer.py               # Generates visual artifacts
│
├── pipeline/
│   └── run_experiment.py          # Pipeline orchestration (pure function)
│
├── utils/
│   └── mlflow_logger.py           # Handles MLflow logging and artifacts
│
├── main.py                        # Pipeline entry point
└── README.md
```

## Features

- Functional and modular design
- Clean separation of responsibilities across components
- Parameterized configuration (no hardcoded logic inside pipeline)
- Logging of metrics, parameters, and artifacts to MLflow
- Confusion matrix visualization logged without local disk dependency
- Structured error handling and explicit status returns

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python main.py
```

This will:
- Train an XGBoost model on the Iris dataset
- Evaluate it using accuracy and log loss
- Log all metadata to MLflow
- Store the trained model and confusion matrix plot as artifacts

## MLflow Tracking

To start the MLflow UI locally:

```bash
mlflow ui
```

Then access the interface at:

```
http://localhost:5000
```

## Key Logged Components

- Model parameters (learning rate, subsample, etc.)
- Accuracy and log loss
- Confusion matrix (visualized and logged in memory)
- Trained model (logged via `mlflow.xgboost`)

## Architectural Principles

This project follows the following software design principles:

- **Single Responsibility**: Each module handles one concern
- **Open/Closed**: Logic can be extended (e.g., new models, metrics) without modification
- **Functional Core**: The core pipeline is pure and stateless
- **Isolated Side Effects**: Only explicitly defined logging modules perform I/O
- **Testability**: Each function can be unit tested in isolation

## Future Improvements

- Add test coverage with `pytest`
- Abstract model training into a strategy pattern
- Extend to support additional datasets or model types (e.g., LightGBM, RandomForest)
- Integrate MLflow model registry and serving infrastructure

## License

This project is licensed under the MIT License.
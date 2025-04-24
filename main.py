from config.hyperparams import HYPERPARAMS
from core.data_loader import load_data, prepare_data
from logger.mlflow_logger import persist_results
from pipeline.run_experiment import run_pipeline


def main():
    X, y, target_names = load_data()

    (X_train, X_test, y_train, y_test), target_names = prepare_data(X, y, target_names)

    results = run_pipeline(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        target_names=target_names,
        params=HYPERPARAMS,
    )

    # persist results
    persist_status = persist_results(
        model=results["model"],
        metrics=results["metrics"],
        y_test=y_test,
        target_names=target_names,
        params=PARAMS,
    )

    print("Persist status:", persist_status["status"])
    print("Message:", persist_status["message"])
    print("Run ID", persist_status["run_id"])
    print("Done!")


if __name__ == "__main__":
    main()

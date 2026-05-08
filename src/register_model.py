import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLOps-Experiment")

with mlflow.start_run() as run:
    mlflow.log_artifact("model.pkl")

    result = mlflow.register_model(
        f"runs:/{run.info.run_id}/model",
        "MyModel"
    )

    print("Model registered:", result.name)
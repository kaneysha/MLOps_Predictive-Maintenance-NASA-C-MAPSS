import json
import mlflow
from mlflow import MlflowClient

THRESHOLD = 0.85

with open("metrics.json") as f:
    metrics = json.load(f)

accuracy = metrics["accuracy"]

client = MlflowClient()

if accuracy >= THRESHOLD:
    print("Model lulus threshold, register ke MLflow")

    mlflow.register_model(
        model_uri="runs:/latest/model",
        name="NASA-CMAPSS-Model"
    )

    # set staging
    latest_version = client.get_latest_versions(
        "NASA-CMAPSS-Model",
        stages=["None"]
    )[0].version

    client.transition_model_version_stage(
        name="NASA-CMAPSS-Model",
        version=latest_version,
        stage="Staging"
    )

    print("Model masuk Staging")
else:
    print("Model gagal threshold, tidak diregister")
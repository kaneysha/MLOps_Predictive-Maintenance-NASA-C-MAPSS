import os
from datetime import datetime

import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
import math
import joblib
from dotenv import load_dotenv

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from mlflow_minio import configure_mlflow


load_dotenv()


def format_run_name(base_name: str | None, experiment_name: str, n_estimators: int, max_depth: int) -> str:
    if base_name:
        return base_name

    safe_experiment = experiment_name.replace(" ", "_")
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"{safe_experiment}-{n_estimators}est-{max_depth}depth-{timestamp}"


parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=int(os.getenv("N_ESTIMATORS", "100")))
parser.add_argument("--max_depth", type=int, default=int(os.getenv("MAX_DEPTH", "10")))
parser.add_argument("--min_r2", type=float, default=float(os.getenv("MIN_R2", "0.5")),
                    help="Minimum R2 threshold for the training quality gate")
parser.add_argument("--max_rmse", type=float, default=float(os.getenv("MAX_RMSE", "20")),
                    help="Maximum RMSE threshold for the training quality gate")
parser.add_argument("--register", action="store_true")
parser.add_argument("--mlflow-tracking-uri", type=str, default=os.getenv("MLFLOW_TRACKING_URI"),
                    help="MLflow tracking server URI (e.g. http://mlflow:5000)")
parser.add_argument("--s3-endpoint", type=str, default=os.getenv("MLFLOW_S3_ENDPOINT_URL"),
                    help="S3/MinIO endpoint (e.g. http://minio:9000)")
parser.add_argument("--aws-access-key", type=str, default=os.getenv("AWS_ACCESS_KEY_ID"))
parser.add_argument("--aws-secret-key", type=str, default=os.getenv("AWS_SECRET_ACCESS_KEY"))
parser.add_argument("--experiment", type=str, default=os.getenv("MLFLOW_EXPERIMENT_NAME", "nasa_cmapss"))
parser.add_argument("--run-name", type=str, default=None,
                    help="MLflow run name; if omitted one is built from experiment and hyperparameters")
parser.add_argument("--model-name", type=str, default=os.getenv("MLFLOW_MODEL_NAME", "nasa_cmapss_model"),
                    help="Model registry name for mlflow.register_model")
parser.add_argument("--mlflow-username", type=str, default=os.getenv("MLFLOW_TRACKING_USERNAME"),
                    help="MLflow tracking server username for basic auth")
parser.add_argument("--mlflow-password", type=str, default=os.getenv("MLFLOW_TRACKING_PASSWORD"),
                    help="MLflow tracking server password for basic auth")

args = parser.parse_args()

run_name = format_run_name(
    base_name=args.run_name,
    experiment_name=args.experiment,
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
)

# Configure MLflow + MinIO (sets env vars and tracking URI)
configure_mlflow(
    tracking_uri=args.mlflow_tracking_uri,
    s3_endpoint=args.s3_endpoint,
    aws_access_key=args.aws_access_key,
    aws_secret_key=args.aws_secret_key,
    experiment_name=args.experiment,
    tracking_username=args.mlflow_username,
    tracking_password=args.mlflow_password,
)

# Load data
df = pd.read_csv("data/processed/cleaned_data.csv")

X = df.drop(columns=["RUL"])
y = df["RUL"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name=run_name) as run:

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = math.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    if r2 < args.min_r2 or rmse > args.max_rmse:
        raise ValueError(
            f"Model failed quality gate | R2={r2:.4f} RMSE={rmse:.4f} "
            f"(min_r2={args.min_r2}, max_rmse={args.max_rmse})"
        )

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)

    # SAVE MODEL LOCAL (INI PENTING)
    joblib.dump(model, "model.pkl")

    # SAVE TO MLFLOW
    mlflow.sklearn.log_model(model, "model")

    if args.register:
        mlflow.register_model(
            f"runs:/{run.info.run_id}/model",
            args.model_name,
        )

    print(f"RMSE={rmse:.4f} R2={r2:.4f}")
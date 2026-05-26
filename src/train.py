import os
import math
import joblib
import argparse
import pandas as pd
from datetime import datetime

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from mlflow_minio import configure_mlflow

# CONFIG
parser = argparse.ArgumentParser()

parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=10)
parser.add_argument("--register", action="store_true")
parser.add_argument("--model-name", type=str, default="nasa_cmapss_model")

args = parser.parse_args()

# MLflow config
configure_mlflow(
    tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
    s3_endpoint=os.getenv("MLFLOW_S3_ENDPOINT_URL"),
    aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    experiment_name="nasa_cmapss"
)

# LOAD DATA
df = pd.read_csv("data/processed/cleaned_data.csv")

X = df.drop(columns=["RUL"])
y = df["RUL"]

feature_list = list(X.columns)

# SAVE FEATURE LIST
with open("features.json", "w") as f:
    import json
    json.dump({"features": feature_list}, f)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TRAIN
with mlflow.start_run():

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = math.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)

    mlflow.log_dict({"features": feature_list}, "features.json")

    mlflow.sklearn.log_model(model, "model")

    joblib.dump(model, "models/model.pkl")

    if args.register:
        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            args.model_name
        )
import pandas as pd
import numpy as np
import argparse
import mlflow
import mlflow.sklearn
import math

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# CONFIG MLFLOW
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("NASA_CMAPSS_RUL_Experiment")

# ARGPARSE (VERSION CONTROL)
parser = argparse.ArgumentParser()

parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=10)
parser.add_argument("--register", action="store_true")

args = parser.parse_args()

# LOAD DATA
df = pd.read_csv("data/processed/cleaned_data.csv")

X = df.drop(columns=["RUL"])
y = df["RUL"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TRAIN MODEL
with mlflow.start_run():

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

    if r2 < 0.6:
        raise ValueError(f"Model too weak: R2={r2}")

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)

    mlflow.set_tag("model", "RandomForest")
    mlflow.set_tag("dataset", "C-MAPSS")

    # MODEL LOGGING (FIXED)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="NASA_RUL_Model" if args.register else None
    )

    print(f"RMSE={rmse:.4f} | R2={r2:.4f}")
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import math

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. SETUP MLFLOW
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("NASA_CMAPSS_RUL_Experiment")

# 2. LOAD DATA (DVC OUTPUT)
DATA_PATH = "data/processed/cleaned_data.csv"
df = pd.read_csv(DATA_PATH)

# target column (sesuaikan dataset kamu)
TARGET = "RUL"

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. HYPERPARAMETER SPACE
experiments = [
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 150, "max_depth": 15},
    {"n_estimators": 200, "max_depth": None},
    {"n_estimators": 300, "max_depth": 10},
]

# 4. TRAIN LOOP (MLFLOW TRACKING)
for i, params in enumerate(experiments):

    with mlflow.start_run(run_name=f"RF_Experiment_{i+1}"):

        # model
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # METRICS
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_test, preds)

        # LOG PARAMS
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])

        # LOG METRICS
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # TAG
        mlflow.set_tag("model", "RandomForest")
        mlflow.set_tag("dataset", "C-MAPSS")

        # SAVE MODEL
        mlflow.sklearn.log_model(model, "model")

        print(f"[RUN {i+1}] RMSE={rmse:.4f}, R2={r2:.4f}")

print("SEMUA EXPERIMEN SELESAI")